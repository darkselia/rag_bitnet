import asyncio
import json
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np
from fastapi import HTTPException

_data = tempfile.TemporaryDirectory()
os.environ["DATA_DIR"] = _data.name
os.environ["LOG_REQUEST_CONTENT"] = "false"

from api.ask_service import _stream_answer, ask
from benchmarks.lm_eval_adapter import BitNetLM
from benchmarks.run_medqa import extract_answer
from clients.bitnet_client import BitNetAPI
from common.formatters import html_to_text
from rag import data_store, models
from rag.ingest import chunk_text, ingest_folder
from rag.retrieval import QueryTooLongError, _allowed_indices, _rerank
from schemas.app_schema import AskReq, LLMResp, Message


class FakeEmbedding:
    def encode(self, texts, **kwargs):
        return np.array([[1.0, float(i)] for i, _ in enumerate(texts)], dtype="float32")


class FakeTokenizer:
    model_max_length = 12
    is_fast = True

    def encode(self, text, **kwargs):
        return list(range(len(re.findall(r"\S+", text))))

    def num_special_tokens_to_add(self, pair=False):
        return 3

    def __call__(self, text, text_pair=None, **kwargs):
        if text_pair is not None:
            size = len(self.encode(text)) + len(self.encode(text_pair)) + self.num_special_tokens_to_add(pair=True)
            return {"input_ids": list(range(size))}
        return {"offset_mapping": [m.span() for m in re.finditer(r"\S+", text)]}


class FakeReranker:
    max_length = None
    tokenizer = FakeTokenizer()

    def predict(self, pairs, batch_size):
        self.pairs = pairs
        return [1.0 if "needle" in text else 0.1 for _, text in pairs]


class RegressionTests(unittest.TestCase):
    def test_reranker_cpu_quantization_is_configurable(self):
        model = unittest.mock.MagicMock()
        with patch.object(models, "_reranker", None), patch.object(models.config, "USE_CROSS_RERANKER", True), \
                patch.object(models.config, "RERANKER_QUANTIZE_CPU", True), \
                patch("rag.models.CrossEncoder", return_value=model), \
                patch("rag.models.torch.ao.quantization.quantize_dynamic") as quantize:
            self.assertIs(models.get_reranker(), model)
            quantize.assert_called_once()
        with patch.object(models, "_reranker", None), patch.object(models.config, "USE_CROSS_RERANKER", True), \
                patch.object(models.config, "RERANKER_QUANTIZE_CPU", False), \
                patch("rag.models.CrossEncoder", return_value=model), \
                patch("rag.models.torch.ao.quantization.quantize_dynamic") as quantize:
            self.assertIs(models.get_reranker(), model)
            quantize.assert_not_called()

    def test_quantization_failure_reloads_unmodified_reranker(self):
        damaged = unittest.mock.MagicMock()
        original = unittest.mock.MagicMock()
        with patch.object(models, "_reranker", None), patch.object(models.config, "USE_CROSS_RERANKER", True), \
                patch.object(models.config, "RERANKER_QUANTIZE_CPU", True), \
                patch("rag.models.CrossEncoder", side_effect=[damaged, original]), \
                patch("rag.models.torch.ao.quantization.quantize_dynamic", side_effect=RuntimeError("unavailable")):
            self.assertIs(models.get_reranker(), original)

    def test_headings_bound_sections_without_splitting_source(self):
        text = "\ufeffFIRST STORY\n\nfirst text\n\nSECOND STORY\n\nsecond text"
        chunks = chunk_text(text)
        self.assertEqual(len(chunks), 2)
        self.assertIn("first text", chunks[0])
        self.assertNotIn("SECOND STORY", chunks[0])
        self.assertIn("second text", chunks[1])

    def test_ingest_keeps_one_source_for_sections(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "stories.txt"
            path.write_text("FIRST STORY\n\nfirst text\n\nSECOND STORY\n\nsecond text", encoding="utf-8")
            with patch("rag.ingest.data_store.add_chunks", return_value=2) as add:
                self.assertEqual(ingest_folder(folder), 2)
        chunks = add.call_args.args[0]
        self.assertEqual(len(chunks), 2)
        self.assertEqual({chunk["source"] for chunk in chunks}, {"stories.txt"})
        self.assertEqual(len({chunk["meta"]["path"] for chunk in chunks}), 1)
        self.assertEqual(len({chunk["id"] for chunk in chunks}), 2)

    def test_markdown_headings_and_code_fence(self):
        text = "# First\ntext\n\n```\n# not a heading\n\nEXAMPLE TITLE\n\n```\n\n## Second\nmore"
        chunks = chunk_text(text)
        self.assertEqual(len(chunks), 2)
        self.assertIn("# not a heading", chunks[0])
        self.assertIn("EXAMPLE TITLE", chunks[0])
        self.assertIn("## Second", chunks[1])

    def test_html_headings_preserved_as_boundaries(self):
        text = html_to_text("<header><h1>Title</h1><nav>menu</nav></header><p>First.</p><h2>Next</h2><p>Second.</p>")
        chunks = chunk_text(text)
        self.assertEqual(len(chunks), 2)
        self.assertIn("First.", chunks[0])
        self.assertIn("Second.", chunks[1])
        self.assertNotIn("menu", text)

    def test_reranker_receives_full_chunk_and_splits_only_oversized(self):
        reranker = FakeReranker()
        full = "start " + "x" * 1500 + " needle"
        self.assertEqual(_rerank("q", [(0, 0.0)], [full], reranker), [(0, 1.0)])
        self.assertEqual(reranker.pairs, [["q", full]])
        long = "one two three four five six seven eight nine ten eleven needle"
        self.assertEqual(_rerank("q", [(0, 0.0)], [long], reranker), [(0, 1.0)])
        self.assertGreater(len(reranker.pairs), 1)
        self.assertTrue(any("needle" in window for _, window in reranker.pairs))
        self.assertTrue(all(len(reranker.tokenizer.encode(window)) <= 8 for _, window in reranker.pairs))

    def test_reranker_rejects_question_that_fills_window(self):
        with self.assertRaises(QueryTooLongError):
            _rerank(" ".join(["q"] * 9), [(0, 0.0)], ["text"], FakeReranker())

    def test_ask_uses_six_candidates_per_method_for_top_one(self):
        with patch("rag.retrieval.retrieve_async", new_callable=AsyncMock, return_value=[]) as retrieve:
            with self.assertRaises(HTTPException) as error:
                asyncio.run(ask(AskReq(messages=[Message(role="user", content="question")], top_k=1)))
        self.assertEqual(error.exception.status_code, 404)
        self.assertEqual(retrieve.call_args.args[1:4], (6, 6, 1))

    def test_ask_returns_413_for_overlong_reranker_question(self):
        with patch("rag.retrieval.retrieve_async", new_callable=AsyncMock,
                   side_effect=QueryTooLongError("too long")):
            with self.assertRaises(HTTPException) as error:
                asyncio.run(ask(AskReq(messages=[Message(role="user", content="question")], top_k=1)))
        self.assertEqual(error.exception.status_code, 413)

    def test_stream_final_without_timings_does_not_duplicate(self):
        pieces = iter([LLMResp(content="Привет"), LLMResp(content="!"), LLMResp(content="", final=True)])
        with patch("api.ask_service._log_csv"):
            events = [json.loads(line) for line in _stream_answer(pieces, "", [], "q", 0.0, "t")]
        self.assertEqual("".join(event["content"] for event in events[:-1]), "Привет!")
        self.assertEqual(events[-1]["answer"]["content"], "Привет!")

    def test_stream_without_final_is_an_error(self):
        events = [json.loads(line) for line in _stream_answer(iter([LLMResp(content="a")]), "", [], "q", 0, "t")]
        self.assertIn("error", events[-1])
        self.assertNotIn("answer", events[-1])

    def test_stream_elapsed_includes_wait_for_first_chunk(self):
        pieces = iter([LLMResp(content="ok"), LLMResp(content="", final=True)])
        with patch("api.ask_service.time.monotonic", return_value=112.0), patch("api.ask_service._log_csv") as log:
            list(_stream_answer(pieces, "", [], "q", 0.0, "t", llm_start=100.0))
        self.assertEqual(log.call_args.args[0]["llm time"], "12.000")

    def test_domain_checks_hostname_and_subdomains(self):
        rows = [{"kind": "site", "source": url} for url in [
            "https://example.com/path", "https://sub.example.com/", "https://example.com.evil/",
            "https://elsewhere.test/example.com",
        ]]
        self.assertEqual(_allowed_indices(rows, ["site"], "example.com"), {0, 1})

    def test_bm25_indices_include_empty_token_rows(self):
        bm25 = data_store._bm25(["!!!", "alpha beta"])
        self.assertEqual(len(bm25.get_scores(["alpha"])), 2)

    def test_failed_index_build_preserves_published_snapshot(self):
        first = [{"id": "one", "text": "first", "meta": {}}]
        with patch("rag.data_store.get_embed_model", return_value=FakeEmbedding()):
            data_store._commit(first)
        previous = data_store.snapshot()
        pointer = (Path(_data.name) / "current.json").read_text(encoding="utf-8")
        with patch("rag.data_store.get_embed_model", side_effect=RuntimeError("failed")):
            with self.assertRaises(RuntimeError):
                data_store._commit([{"id": "two", "text": "second", "meta": {}}])
        self.assertIs(data_store.snapshot(), previous)
        self.assertEqual((Path(_data.name) / "current.json").read_text(encoding="utf-8"), pointer)

    def test_medqa_requires_unambiguous_letter(self):
        self.assertEqual(extract_answer("(B).\n"), "B")
        self.assertIsNone(extract_answer("The answer is B"))
        self.assertIsNone(extract_answer("A or B"))

    def test_benchmark_transport_error_is_not_a_prediction(self):
        model = BitNetLM()
        request = type("Request", (), {"args": ("prompt", {})})()
        with patch.object(model, "_call_api", side_effect=RuntimeError("offline")):
            with self.assertRaisesRegex(RuntimeError, "offline"):
                model.generate_until([request])
        with self.assertRaises(NotImplementedError):
            model.loglikelihood([request])

    def test_benchmark_preserves_whitespace(self):
        model = BitNetLM(mode="completion")
        request = type("Request", (), {"args": ("prompt", {})})()
        with patch.object(model, "_call_api", return_value={"content": "one\n  two"}):
            self.assertEqual(model.generate_until([request]), ["one\n  two"])

    def test_benchmark_model_alias_does_not_verify_identity(self):
        model = BitNetLM()
        model.model_name = "gpt-3.5-turbo-0613"
        response = unittest.mock.MagicMock()
        response.json.return_value = {"data": [{"id": "bitnet.gguf"}]}
        with patch("benchmarks.lm_eval_adapter.requests.get", return_value=response):
            identity = model.model_identity()
        self.assertEqual(identity["server_model"], "bitnet.gguf")
        self.assertEqual(identity["response_model"], "gpt-3.5-turbo-0613")
        self.assertFalse(identity["server_model_verified"])

    def test_completion_stream_requires_final_event(self):
        response = unittest.mock.MagicMock()
        response.__enter__.return_value = response
        response.iter_lines.return_value = [b'data: {"content":"hello","stop":false}']
        with patch("clients.bitnet_client.requests.post", return_value=response):
            stream = BitNetAPI(mode="completion").generate_stream([])
            self.assertEqual(next(stream).content, "hello")
            with self.assertRaisesRegex(ValueError, "finish event"):
                next(stream)


if __name__ == "__main__":
    unittest.main()
