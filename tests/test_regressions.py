import asyncio
import csv
import json
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np
import requests
from docx import Document
from fastapi import HTTPException
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
from starlette.responses import StreamingResponse

_data = tempfile.TemporaryDirectory()
os.environ["DATA_DIR"] = _data.name
os.environ["LOG_REQUEST_CONTENT"] = "false"

from api.ask_service import NO_CONTEXT_ANSWER, _stream_answer, ask
from api.server import app, startup_event
from benchmarks.lm_eval_adapter import BitNetLM
from benchmarks.run_medqa import extract_answer
from clients.bitnet_client import BitNetAPI
from common import config
from common.formatters import file_to_document, html_to_text
from rag import data_store, models
from rag.ingest import chunk_text, ingest_folder
from rag.retrieval import QueryTooLongError, _allowed_indices, _rerank, retrieve
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


def write_test_pdf(path: Path, text: str | None) -> None:
    writer = PdfWriter()
    page = writer.add_blank_page(width=200, height=200)
    if text is not None:
        font = DictionaryObject({
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        })
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): font}),
        })
        content = DecodedStreamObject()
        content.set_data(f"BT /F1 12 Tf 20 100 Td ({text}) Tj ET".encode("ascii"))
        page[NameObject("/Contents")] = writer._add_object(content)
    writer.write(str(path))


async def asgi_request(
        method: str, path: str, payload: dict | None = None, api_key: str | None = None,
) -> tuple[int, bytes]:
    headers = [(b"content-type", b"application/json")]
    if api_key is not None:
        headers.append((b"x-api-key", api_key.encode("latin-1")))
    body = json.dumps(payload).encode("utf-8") if payload is not None else b""
    scope = {
        "type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1", "method": method, "scheme": "http", "path": path,
        "raw_path": path.encode("ascii"), "query_string": b"", "headers": headers,
        "server": ("testserver", 80), "client": ("testclient", 123), "root_path": "",
    }
    events: list[dict] = []
    received = False

    async def receive() -> dict:
        nonlocal received
        if not received:
            received = True
            return {"type": "http.request", "body": body, "more_body": False}
        await asyncio.Event().wait()
        return {"type": "http.disconnect"}

    async def send(event: dict) -> None:
        events.append(event)

    await app(scope, receive, send)
    status = next(event["status"] for event in events if event["type"] == "http.response.start")
    content = b"".join(event.get("body", b"") for event in events if event["type"] == "http.response.body")
    return status, content


class RegressionTests(unittest.TestCase):
    def test_api_key_is_required_at_startup(self):
        with patch.object(config, "RAG_API_KEY", ""):
            with self.assertRaisesRegex(RuntimeError, "RAG_API_KEY"):
                asyncio.run(startup_event())
        with patch.object(config, "RAG_API_KEY", "test-secret"):
            asyncio.run(startup_event())

    def test_api_key_protects_all_routes(self):
        cases = [
            ("/ask", {"messages": [{"role": "user", "content": "question"}]}, "api.routes.ask_service.ask"),
            ("/ask", {"messages": [{"role": "user", "content": "question"}], "stream": True},
             "api.routes.ask_service.ask"),
            ("/ingest/files", {"folder": "data/rag_docs"}, "api.routes.ingest_service.ingest_files"),
            ("/ingest/site", {"base_url": "https://example.com"}, "api.routes.ingest_service.ingest_site"),
        ]
        with patch.object(config, "RAG_API_KEY", "test-secret"):
            for path, payload, target in cases:
                for key in (None, "", "wrong-secret", "\u00e9"):
                    with self.subTest(path=path, stream=payload.get("stream"), key=key), \
                            patch(target, new_callable=AsyncMock) as service:
                        status, body = asyncio.run(asgi_request("POST", path, payload, key))
                        self.assertEqual(status, 401)
                        self.assertNotIn(b"test-secret", body)
                        self.assertNotIn(b"wrong-secret", body)
                        service.assert_not_awaited()
                with self.subTest(path=path, stream=payload.get("stream"), key="valid"), \
                        patch(target, new_callable=AsyncMock, return_value={"ok": True}) as service:
                    status, body = asyncio.run(asgi_request("POST", path, payload, "test-secret"))
                    self.assertEqual(status, 200)
                    self.assertEqual(json.loads(body), {"ok": True})
                    service.assert_awaited_once()
        with patch.object(config, "RAG_API_KEY", ""):
            status, _ = asyncio.run(asgi_request("POST", "/ingest/files", {"folder": "x"}, "test-secret"))
            self.assertEqual(status, 401)

    def test_api_key_allows_streaming_and_is_documented(self):
        payload = {"messages": [{"role": "user", "content": "question"}], "stream": True}
        response = StreamingResponse(iter([b'{"content":"ok"}\n']), media_type="application/x-ndjson")
        with patch.object(config, "RAG_API_KEY", "test-secret"), \
                patch("api.routes.ask_service.ask", new_callable=AsyncMock, return_value=response):
            status, body = asyncio.run(asgi_request("POST", "/ask", payload, "test-secret"))
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), {"content": "ok"})

        status, body = asyncio.run(asgi_request("GET", "/openapi.json"))
        self.assertEqual(status, 200)
        schema = json.loads(body)
        self.assertEqual(schema["components"]["securitySchemes"]["APIKeyHeader"]["name"], "X-API-Key")
        for path in ("/ask", "/ingest/files", "/ingest/site"):
            self.assertIn({"APIKeyHeader": []}, schema["paths"][path]["post"]["security"])
        status, _ = asyncio.run(asgi_request("GET", "/docs"))
        self.assertEqual(status, 200)

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

    def test_docx_preserves_document_and_table_order(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sample.docx"
            document = Document()
            document.add_heading("Проверка", level=1)
            document.add_paragraph("Первый абзац")
            table = document.add_table(rows=1, cols=2)
            table.cell(0, 0).text = "Цвет"
            table.cell(0, 1).text = "Синий"
            nested = table.cell(0, 1).add_table(rows=1, cols=1)
            nested.cell(0, 0).text = "Вложенная таблица"
            document.add_heading("Раздел 7", level=7)
            document.add_paragraph("Последний абзац")
            document.save(path)
            formatted = file_to_document(path)
        self.assertIsNotNone(formatted)
        self.assertEqual(formatted.source, "sample.docx")
        self.assertIn("# Проверка", formatted.text)
        self.assertLess(formatted.text.index("Первый абзац"), formatted.text.index("Цвет"))
        self.assertLess(formatted.text.index("Вложенная таблица"), formatted.text.index("Последний абзац"))
        self.assertIn("Синий", formatted.text)
        self.assertIn("###### Раздел 7", formatted.text)

    def test_csv_formats_utf8_quoted_multiline_values(self):
        with tempfile.TemporaryDirectory() as folder:
            for delimiter in (",", ";"):
                with self.subTest(delimiter=delimiter):
                    path = Path(folder) / f"sample-{ord(delimiter)}.csv"
                    with path.open("w", encoding="utf-8-sig", newline="") as f:
                        writer = csv.writer(f, delimiter=delimiter)
                        writer.writerow(["Название, код" if delimiter == ";" else "Название; код", "Описание"])
                        writer.writerow(["Тест", 'Строка 1, "текст";\nСтрока 2'])
                    formatted = file_to_document(path)
                    self.assertIsNotNone(formatted)
                    self.assertEqual(formatted.source, path.name)
                    self.assertIn("Описание: Строка 1, \"текст\";\nСтрока 2", formatted.text)
                    self.assertIn("Тест", formatted.text)

    def test_csv_rejects_ambiguous_malformed_and_invalid_encoding(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "invalid.csv"
            for content, message in [
                (b"a,b;c\n1,2;3\n", "Ambiguous CSV delimiter"),
                (b'a,b\n1,"unfinished\n', "Invalid CSV file"),
                (b"a,b\n1\n", "column count mismatch"),
                (b"a,b\n1,\xff\n", "must be UTF-8"),
            ]:
                with self.subTest(message=message):
                    path.write_bytes(content)
                    with self.assertRaisesRegex(ValueError, message):
                        file_to_document(path)

    def test_csv_accepts_large_field_within_file_limit(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "large.csv"
            path.write_text("name;description\nitem;" + "x" * 150000 + "\n", encoding="utf-8")
            formatted = file_to_document(path)
        self.assertIsNotNone(formatted)
        self.assertIn("x" * 150000, formatted.text)

    def test_ingest_indexes_supported_formats_as_separate_sources(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            document = Document()
            document.add_paragraph("DOCX говорит: сапфир")
            document.save(root / "sample.docx")
            (root / "sample.csv").write_text("название;значение\nкамень;топаз\n", encoding="utf-8")
            (root / "sample.txt").write_text("Текст: янтарь", encoding="utf-8")
            (root / "sample.md").write_text("# Текст\nРубин", encoding="utf-8")
            (root / "sample.html").write_text("<h1>Текст</h1><p>Алмаз</p>", encoding="utf-8")
            write_test_pdf(root / "sample.pdf", "PDF text")
            with patch("rag.ingest.data_store.add_chunks", return_value=6) as add:
                self.assertEqual(ingest_folder(folder), 6)
        chunks = add.call_args.args[0]
        self.assertEqual({chunk["source"] for chunk in chunks}, {
            "sample.docx", "sample.csv", "sample.txt", "sample.md", "sample.html", "sample.pdf",
        })
        self.assertEqual(len({chunk["meta"]["path"] for chunk in chunks}), 6)
        self.assertTrue(any("сапфир" in chunk["text"] for chunk in chunks))
        self.assertTrue(any("значение: топаз" in chunk["text"] for chunk in chunks))
        self.assertTrue(any("PDF text" in chunk["text"] for chunk in chunks))

    def test_pdf_without_text_reports_error(self):
        with tempfile.TemporaryDirectory() as folder:
            write_test_pdf(Path(folder) / "scan.pdf", None)
            with self.assertRaisesRegex(ValueError, "No extractable text.*scan.pdf"):
                ingest_folder(folder)

    def test_bad_or_empty_files_preserve_published_index(self):
        with tempfile.TemporaryDirectory() as docs, tempfile.TemporaryDirectory() as store:
            root = Path(docs)
            csv_path = root / "source.csv"
            docx_path = root / "source.docx"
            csv_path.write_text("предмет;цвет\nцветок;синий\n", encoding="utf-8")
            document = Document()
            document.add_paragraph("Документ до обновления")
            document.save(docx_path)
            with patch.object(config, "DATA_DIR", Path(store)), patch.object(data_store, "rows", []), \
                    patch.object(data_store, "_snapshot", data_store.Snapshot([], [], None, None)), \
                    patch("rag.data_store.get_embed_model", return_value=FakeEmbedding()):
                self.assertEqual(ingest_folder(docs, replace_existing=True), 2)
                previous = data_store.snapshot()
                pointer = (Path(store) / "current.json").read_text(encoding="utf-8")
                for name, content in [("source.docx", b"not a docx"), ("source.csv", b"")]:
                    path = root / name
                    original = path.read_bytes()
                    path.write_bytes(content)
                    with self.subTest(name=name), self.assertRaises(ValueError):
                        ingest_folder(docs, replace_existing=True)
                    self.assertIs(data_store.snapshot(), previous)
                    self.assertEqual((Path(store) / "current.json").read_text(encoding="utf-8"), pointer)
                    path.write_bytes(original)
                csv_path.write_text("предмет;цвет\nцветок;красный\n", encoding="utf-8")
                updated = Document()
                updated.add_paragraph("Документ после обновления")
                updated.save(docx_path)
                self.assertEqual(ingest_folder(docs, replace_existing=True), 2)
                self.assertTrue(any("красный" in row["text"] for row in data_store.rows))
                self.assertFalse(any("синий" in row["text"] for row in data_store.rows))
                self.assertTrue(any("после обновления" in row["text"] for row in data_store.rows))

    def test_ingest_limits_file_batch_and_expanded_docx_size(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "a.txt").write_text("1234", encoding="utf-8")
            (root / "b.txt").write_text("5678", encoding="utf-8")
            with patch.object(config, "INGEST_MAX_FILE_BYTES", 3):
                with self.assertRaisesRegex(ValueError, "File size exceeds limit"):
                    ingest_folder(folder)
            with patch.object(config, "INGEST_MAX_FILE_BYTES", 10), \
                    patch.object(config, "INGEST_MAX_BATCH_BYTES", 7):
                with self.assertRaisesRegex(ValueError, "Total file size exceeds"):
                    ingest_folder(folder)
            path = root / "sample.docx"
            document = Document()
            document.add_paragraph("Текст")
            document.save(path)
            with patch.object(config, "INGEST_MAX_DOCX_UNCOMPRESSED_BYTES", 1):
                with self.assertRaisesRegex(ValueError, "DOCX uncompressed size exceeds limit"):
                    file_to_document(path)

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

    def test_reranker_min_score_filters_fused_candidates_and_enables_fallback(self):
        rows = [
            {"id": "first", "text": "first text", "source": "one", "kind": "docs"},
            {"id": "second", "text": "second text", "source": "two", "kind": "docs"},
        ]
        index = unittest.mock.MagicMock()
        index.search.return_value = (np.array([[0.9, 0.8]], dtype="float32"), np.array([[0, 1]]))
        bm25 = unittest.mock.MagicMock()
        bm25.get_scores.return_value = np.array([1.0, 0.5])
        state = data_store.Snapshot(rows, [row["text"] for row in rows], index, bm25)
        with patch("rag.data_store.snapshot", return_value=state), \
                patch("rag.models.get_embed_model", return_value=FakeEmbedding()), \
                patch("rag.models.get_reranker", return_value=object()), \
                patch("rag.retrieval._rerank", return_value=[(0, 0.09), (1, 0.04)]):
            with patch.object(config, "RERANKER_MIN_SCORE", None):
                self.assertEqual([row["id"] for row in retrieve("question", k_final=2)], ["first", "second"])
            with patch.object(config, "RERANKER_MIN_SCORE", 0.05):
                self.assertEqual([row["id"] for row in retrieve("question", k_final=2)], ["first"])
            with patch.object(config, "RERANKER_MIN_SCORE", 0.1), \
                    patch.object(config, "RAG_API_KEY", "test-secret"), \
                    patch("api.ask_service.get_llm") as get_llm, patch("api.ask_service._log_csv"):
                self.assertEqual(retrieve("question", k_final=2), [])
                payload = {"messages": [{"role": "user", "content": "question"}]}
                status, body = asyncio.run(asgi_request("POST", "/ask", payload, "test-secret"))
                self.assertEqual(status, 200)
                self.assertEqual(json.loads(body)["answer"]["content"], NO_CONTEXT_ANSWER)
                get_llm.assert_not_called()

    def test_ask_uses_six_candidates_per_method_for_top_one(self):
        with patch("rag.retrieval.retrieve_async", new_callable=AsyncMock, return_value=[]) as retrieve, \
                patch("api.ask_service.get_llm") as get_llm, patch("api.ask_service._log_csv"):
            response = asyncio.run(ask(AskReq(messages=[Message(role="user", content="question")], top_k=1)))
        self.assertEqual(response.answer.content, NO_CONTEXT_ANSWER)
        self.assertEqual(response.used_context, [])
        get_llm.assert_not_called()
        self.assertEqual(retrieve.call_args.args[1:4], (6, 6, 1))

    def test_empty_context_returns_matching_json_and_ndjson_without_llm(self):
        with patch.object(config, "RAG_API_KEY", "test-secret"), \
                patch("rag.retrieval.retrieve_async", new_callable=AsyncMock, return_value=[]), \
                patch("api.ask_service.get_llm") as get_llm, patch("api.ask_service._log_csv") as log:
            for stream in (False, True):
                payload = {"messages": [{"role": "user", "content": "question"}], "stream": stream}
                status, body = asyncio.run(asgi_request("POST", "/ask", payload, "test-secret"))
                self.assertEqual(status, 200)
                events = [json.loads(line) for line in body.splitlines()]
                self.assertEqual(events[-1]["answer"]["content"], NO_CONTEXT_ANSWER)
                self.assertEqual(events[-1]["used_context"], [])
                self.assertIsNone(events[-1]["timings"])
                if stream:
                    self.assertEqual(len(events), 2)
                    self.assertEqual(events[0]["content"], NO_CONTEXT_ANSWER)
                else:
                    self.assertEqual(len(events), 1)
            self.assertEqual(log.call_count, 2)
            get_llm.assert_not_called()

    def test_context_window_removes_chunks_and_never_generates_without_context(self):
        contexts = [{"id": "first", "text": "first"}, {"id": "second", "text": "second"}]
        for stream in (False, True):
            with self.subTest(stream=stream):
                llm = unittest.mock.MagicMock()
                llm.count_prompt_tokens.side_effect = [300, 250, 100]
                payload = {"messages": [{"role": "user", "content": "question"}], "stream": stream}
                with patch.object(config, "RAG_API_KEY", "test-secret"), \
                        patch.object(config, "BITNET_CONTEXT_SIZE", 200), \
                        patch.object(config, "LLM_MAX_TOKENS", 20), \
                        patch("rag.retrieval.retrieve_async", new_callable=AsyncMock, return_value=contexts.copy()), \
                        patch("api.ask_service.get_llm", return_value=llm), patch("api.ask_service._log_csv"):
                    status, body = asyncio.run(asgi_request("POST", "/ask", payload, "test-secret"))
                self.assertEqual(status, 200)
                events = [json.loads(line) for line in body.splitlines()]
                self.assertEqual(events[-1]["answer"]["content"], NO_CONTEXT_ANSWER)
                self.assertEqual(events[-1]["used_context"], [])
                self.assertEqual(llm.count_prompt_tokens.call_args_list[-1].args[1], "")
                llm.generate.assert_not_called()
                llm.generate_stream.assert_not_called()

    def test_context_window_preserves_fitting_chunks_or_returns_413(self):
        contexts = [{"id": "first", "text": "first"}, {"id": "second", "text": "second"}]
        llm = unittest.mock.MagicMock()
        llm.count_prompt_tokens.side_effect = [300, 100]
        llm.generate.return_value = LLMResp(content="answer")
        req = AskReq(messages=[Message(role="user", content="question")])
        with patch.object(config, "BITNET_CONTEXT_SIZE", 200), patch.object(config, "LLM_MAX_TOKENS", 20), \
                patch("rag.retrieval.retrieve_async", new_callable=AsyncMock, return_value=contexts.copy()), \
                patch("api.ask_service.get_llm", return_value=llm), patch("api.ask_service._log_csv"):
            response = asyncio.run(ask(req))
        self.assertEqual(response.used_context, contexts[:1])
        self.assertEqual(llm.generate.call_args.kwargs["context"], "[first]\nfirst")

        llm.reset_mock()
        llm.count_prompt_tokens.side_effect = [300, 300, 300]
        with patch.object(config, "BITNET_CONTEXT_SIZE", 200), patch.object(config, "LLM_MAX_TOKENS", 20), \
                patch("rag.retrieval.retrieve_async", new_callable=AsyncMock, return_value=contexts.copy()), \
                patch("api.ask_service.get_llm", return_value=llm), patch("api.ask_service._log_csv"):
            with self.assertRaises(HTTPException) as error:
                asyncio.run(ask(req))
        self.assertEqual(error.exception.status_code, 413)
        llm.generate.assert_not_called()
        llm.generate_stream.assert_not_called()

    def test_context_window_reports_tokenization_errors(self):
        contexts = [{"id": "first", "text": "first"}]
        req = AskReq(messages=[Message(role="user", content="question")])
        for error, status in ((requests.Timeout("timeout"), 504), (requests.HTTPError("not found"), 502)):
            with self.subTest(status=status):
                llm = unittest.mock.MagicMock()
                llm.count_prompt_tokens.side_effect = error
                with patch.object(config, "BITNET_CONTEXT_SIZE", 200), \
                        patch("rag.retrieval.retrieve_async", new_callable=AsyncMock, return_value=contexts.copy()), \
                        patch("api.ask_service.get_llm", return_value=llm), patch("api.ask_service._log_csv"):
                    with self.assertRaises(HTTPException) as exc:
                        asyncio.run(ask(req))
                self.assertEqual(exc.exception.status_code, status)
                llm.generate.assert_not_called()
                llm.generate_stream.assert_not_called()

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
