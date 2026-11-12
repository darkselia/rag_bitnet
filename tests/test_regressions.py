import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

_data = tempfile.TemporaryDirectory()
os.environ["DATA_DIR"] = _data.name
os.environ["LOG_REQUEST_CONTENT"] = "false"

from api.ask_service import _stream_answer
from benchmarks.lm_eval_adapter import BitNetLM
from benchmarks.run_medqa import extract_answer
from clients.bitnet_client import BitNetAPI
from rag import data_store
from rag.retrieval import _allowed_indices
from schemas.app_schema import LLMResp


class FakeEmbedding:
    def encode(self, texts, **kwargs):
        return np.array([[1.0, float(i)] for i, _ in enumerate(texts)], dtype="float32")


class RegressionTests(unittest.TestCase):
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
