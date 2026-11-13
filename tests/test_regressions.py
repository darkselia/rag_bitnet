import asyncio
import csv
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
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
from api.server import OperationMiddleware, app, startup_event
from benchmarks.lm_eval_adapter import BitNetLM
from benchmarks.run_medqa import extract_answer
from clients.bitnet_client import BitNetAPI
from common import config
from common import logger as logging_utils
from common.formatters import FormattedDocument, file_to_document, html_to_text
from rag import data_store, models
from rag.ingest import CrawlConfig, chunk_text, crawl_site, ingest_folder
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
        response_headers: list[tuple[bytes, bytes]] | None = None,
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
    response = next(event for event in events if event["type"] == "http.response.start")
    status = response["status"]
    if response_headers is not None:
        response_headers.extend(response.get("headers", []))
    content = b"".join(event.get("body", b"") for event in events if event["type"] == "http.response.body")
    return status, content


class RegressionTests(unittest.TestCase):
    def test_operation_log_connects_ask_events_without_content(self):
        events, csv_rows, headers = [], [], []
        secret = "private-question-and-key"
        llm = unittest.mock.MagicMock()
        llm.generate.return_value = LLMResp(id="model-id", content="private-answer")
        contexts = [{"id": "doc", "text": "private-context", "source": "file", "kind": "docs"}]
        payload = {"messages": [{"role": "user", "content": secret}], "top_k": 1}
        with patch.object(config, "RAG_API_KEY", secret), patch.object(config, "BITNET_CONTEXT_SIZE", 0), \
                patch.object(config, "LOG_REQUEST_CONTENT", False), \
                patch("api.server.log_operation", side_effect=lambda event: events.append(event.copy())), \
                patch("api.ask_service.csv_logger.log", side_effect=lambda row: csv_rows.append(row.copy())), \
                patch("api.ask_service.get_llm", return_value=llm), \
                patch("rag.retrieval.retrieve_async", new_callable=AsyncMock, return_value=contexts):
            status, _ = asyncio.run(asgi_request("POST", "/ask", payload, secret, headers))
        self.assertEqual(status, 200)
        self.assertEqual([event["event"] for event in events], ["started", "completed"])
        operation_id = events[0]["operation_id"]
        self.assertEqual(events[1]["operation_id"], operation_id)
        self.assertEqual(dict(headers)[b"x-request-id"].decode(), operation_id)
        self.assertEqual(events[1]["status"], "success")
        self.assertEqual(events[1]["response_mode"], "json")
        self.assertGreaterEqual(events[1]["retrieval_ms"], 0)
        self.assertGreaterEqual(events[1]["generation_ms"], 0)
        self.assertIsNotNone(datetime.fromisoformat(events[1]["timestamp"]).utcoffset())
        self.assertTrue(all(row["operation_id"] == operation_id for row in csv_rows))
        logged = json.dumps([events, csv_rows])
        for value in (secret, "private-answer", "private-context"):
            self.assertNotIn(value, logged)

    def test_operation_log_records_failures_and_stream_error(self):
        events = []
        payload = {"messages": [{"role": "user", "content": "private question"}], "stream": True}
        contexts = [{"id": "doc", "text": "private context", "source": "file", "kind": "docs"}]

        def broken_stream():
            yield LLMResp(content="partial answer")
            raise requests.ConnectionError("sensitive backend error")

        llm = unittest.mock.MagicMock()
        llm.generate_stream.return_value = broken_stream()
        with patch.object(config, "RAG_API_KEY", "test-secret"), patch.object(config, "BITNET_CONTEXT_SIZE", 0), \
                patch("api.server.log_operation", side_effect=lambda event: events.append(event.copy())), \
                patch("api.ask_service._log_csv"), patch("api.ask_service.get_llm", return_value=llm), \
                patch("rag.retrieval.retrieve_async", new_callable=AsyncMock, return_value=contexts):
            status, body = asyncio.run(asgi_request("POST", "/ask", payload, "test-secret"))
            self.assertEqual(status, 200)
            self.assertIn("error", json.loads(body.splitlines()[-1]))
            completed = events[-1]
            self.assertEqual((completed["status"], completed["error_category"]), ("error", "llm_stream_error"))
            self.assertEqual(completed["http_status"], 200)
            self.assertEqual(completed["response_mode"], "stream")
            self.assertGreaterEqual(completed["generation_ms"], 0)
            status, _ = asyncio.run(asgi_request("POST", "/ask", payload, "wrong-secret"))
            self.assertEqual(status, 401)
            self.assertEqual(events[-1]["error_category"], "authentication")
            status, _ = asyncio.run(asgi_request("POST", "/ask", {"messages": []}, "test-secret"))
            self.assertEqual(status, 422)
            self.assertEqual(events[-1]["error_category"], "invalid_request")
        self.assertNotEqual(events[0]["operation_id"], events[-1]["operation_id"])
        for value in ("private question", "private context", "partial answer", "sensitive backend error",
                      "wrong-secret"):
            self.assertNotIn(value, json.dumps(events))

    def test_csv_content_requires_explicit_setting(self):
        from api.ask_service import _log_csv

        row = {"timestamp": "time", "user question": "question", "contest": "context", "llm answer": "answer"}
        with patch("api.ask_service.csv_logger.log") as log:
            with patch.object(config, "LOG_REQUEST_CONTENT", False):
                _log_csv(row)
            self.assertEqual(log.call_args.args[0], {"operation_id": "time", "timestamp": "time"})
            with patch.object(config, "LOG_REQUEST_CONTENT", True):
                _log_csv(row)
            self.assertEqual(log.call_args.args[0]["user question"], "question")
            self.assertEqual(log.call_args.args[0]["contest"], "context")
            self.assertEqual(log.call_args.args[0]["llm answer"], "answer")

    def test_parallel_http_requests_have_distinct_operation_ids(self):
        events = []
        payload = {"messages": [{"role": "user", "content": "question"}]}

        async def requests_together():
            return await asyncio.gather(*(asgi_request("POST", "/ask", payload, "test-secret") for _ in range(20)))

        with patch.object(config, "RAG_API_KEY", "test-secret"), \
                patch("api.server.log_operation", side_effect=lambda event: events.append(event.copy())), \
                patch("api.ask_service._log_csv"), \
                patch("rag.retrieval.retrieve_async", new_callable=AsyncMock, return_value=[]):
            responses = asyncio.run(requests_together())
        self.assertTrue(all(status == 200 for status, _ in responses))
        operation_ids = {event["operation_id"] for event in events}
        self.assertEqual(len(operation_ids), 20)
        for operation_id in operation_ids:
            related = [event for event in events if event["operation_id"] == operation_id]
            self.assertEqual([event["event"] for event in related], ["started", "completed"])
            self.assertEqual(related[-1]["status"], "success")

    def test_operation_log_records_successful_stream_and_initial_timeout(self):
        events = []
        payload = {"messages": [{"role": "user", "content": "question"}], "stream": True}
        contexts = [{"id": "doc", "text": "context", "source": "file", "kind": "docs"}]
        llm = unittest.mock.MagicMock()
        llm.generate_stream.return_value = iter([LLMResp(content="answer"), LLMResp(content="", final=True)])
        with patch.object(config, "RAG_API_KEY", "test-secret"), patch.object(config, "BITNET_CONTEXT_SIZE", 0), \
                patch("api.server.log_operation", side_effect=lambda event: events.append(event.copy())), \
                patch("api.ask_service._log_csv"), patch("api.ask_service.get_llm", return_value=llm), \
                patch("rag.retrieval.retrieve_async", new_callable=AsyncMock, return_value=contexts):
            status, body = asyncio.run(asgi_request("POST", "/ask", payload, "test-secret"))
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body.splitlines()[-1])["answer"]["content"], "answer")
            self.assertEqual((events[-1]["status"], events[-1]["response_mode"]), ("success", "stream"))

            def timeout_stream():
                raise requests.Timeout("private timeout")
                yield

            llm.generate_stream.return_value = timeout_stream()
            status, body = asyncio.run(asgi_request("POST", "/ask", payload, "test-secret"))
            self.assertEqual(status, 504)
            self.assertEqual((events[-1]["status"], events[-1]["error_category"]), ("error", "llm_timeout"))
            self.assertGreaterEqual(events[-1]["generation_ms"], 0)
            self.assertNotIn(b"private timeout", body)

    def test_operation_log_records_interrupted_response(self):
        events = []

        async def interrupted_app(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"part", "more_body": True})

        async def receive():
            return {"type": "http.disconnect"}

        async def send(message):
            pass

        middleware = OperationMiddleware(interrupted_app)
        with patch("api.server.log_operation", side_effect=lambda event: events.append(event.copy())):
            asyncio.run(middleware({"type": "http", "method": "POST", "path": "/ask"}, receive, send))
        self.assertEqual((events[-1]["status"], events[-1]["error_category"]), ("error", "interrupted"))

    def test_ingest_operation_records_version_and_sanitizes_errors(self):
        events = []
        with tempfile.TemporaryDirectory() as tmp, patch.object(config, "DATA_DIR", Path(tmp) / "index"), \
                patch.object(config, "RAG_API_KEY", "test-secret"), \
                patch.object(data_store, "rows", []), patch.object(data_store, "_snapshot",
                                                                   data_store.Snapshot([], [], None, None)), \
                patch("rag.data_store.get_embed_model", return_value=FakeEmbedding()), \
                patch("api.server.log_operation", side_effect=lambda event: events.append(event.copy())):
            folder = Path(tmp) / "private-folder"
            folder.mkdir()
            (folder / "private-file.txt").write_text("private document contents", encoding="utf-8")
            payload = {"folder": str(folder), "replace_existing": True}
            status, _ = asyncio.run(asgi_request("POST", "/ingest/files", payload, "test-secret"))
            self.assertEqual(status, 200)
            self.assertEqual(events[-1]["index_version"], data_store.snapshot().version)
            self.assertEqual((events[-1]["added_chunks"], events[-1]["total_chunks"]), (1, 1))
            (folder / "private-file.txt").write_text("", encoding="utf-8")
            status, body = asyncio.run(asgi_request("POST", "/ingest/files", payload, "test-secret"))
            self.assertEqual(status, 400)
            self.assertEqual(events[-1]["error_category"], "invalid_input")
            self.assertNotIn(b"private", body)
            (folder / "private-file.txt").unlink()
            (folder / "private-file.pdf").write_bytes(b"secret-document-header that is not a PDF")
            status, body = asyncio.run(asgi_request("POST", "/ingest/files", payload, "test-secret"))
            self.assertEqual(status, 400)
            self.assertEqual(events[-1]["error_category"], "invalid_input")
            self.assertNotIn(b"secret-document-header", body)
            self.assertGreater(logging.getLogger("pypdf._reader").getEffectiveLevel(), logging.WARNING)
            with patch("api.ingest_service.ingest_folder", side_effect=OSError("sensitive internal path")):
                status, body = asyncio.run(asgi_request("POST", "/ingest/files", payload, "test-secret"))
            self.assertEqual(status, 500)
            self.assertEqual(events[-1]["error_category"], "ingest_internal_error")
            self.assertNotIn(b"sensitive", body)
        for value in ("private-folder", "private-file", "private document contents", "test-secret",
                      "sensitive internal path", "secret-document-header"):
            self.assertNotIn(value, json.dumps(events))

    def test_partial_site_operation_records_counts_without_urls(self):
        events = []

        def partial(cfg, failed_urls):
            failed_urls.append("https://example.com/private?token=secret")
            return 0

        with patch.object(config, "RAG_API_KEY", "test-secret"), \
                patch("api.server.log_operation", side_effect=lambda event: events.append(event.copy())), \
                patch("api.ingest_service.crawl_site", side_effect=partial):
            status, body = asyncio.run(
                asgi_request("POST", "/ingest/site", {"base_url": "https://example.com"}, "test-secret"))
        self.assertEqual(status, 200)
        self.assertEqual(len(json.loads(body)["failed_urls"]), 1)
        self.assertEqual((events[-1]["status"], events[-1]["failed_sources"]), ("partial", 1))
        self.assertNotIn("secret", json.dumps(events))

    def test_rotated_logs_are_safe_for_parallel_threads(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(logging_utils, "LOG_DIR", Path(tmp)), \
                patch.object(logging_utils, "LOG_MAX_BYTES", 512), patch.object(logging_utils, "LOG_BACKUP_COUNT", 20):
            audit = logging.Logger("temporary-operations")
            csv_logger = logging_utils.CsvLogger("parallel", fieldnames=["operation_id", "value"], key="operation_id")
            with patch("common.logger.logging.getLogger", return_value=audit):
                def write(number):
                    logging_utils.log_operation({"operation_id": str(number), "event": "completed"})
                    csv_logger.log({"operation_id": str(number), "value": str(number)})

                with ThreadPoolExecutor(max_workers=8) as pool:
                    list(pool.map(write, range(80)))
            for handler in audit.handlers:
                handler.close()
            operation_events = []
            for path in Path(tmp).glob("operations.jsonl*"):
                operation_events.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
            self.assertEqual({event["operation_id"] for event in operation_events}, {str(i) for i in range(80)})
            self.assertGreater(len(list(Path(tmp).glob("operations.jsonl*"))), 1)
            csv_rows = []
            for path in Path(tmp).glob("parallel.csv*"):
                with path.open(encoding="utf-8", newline="") as file:
                    csv_rows.extend(csv.DictReader(file, delimiter=";"))
            self.assertEqual({row["operation_id"] for row in csv_rows}, {str(i) for i in range(80)})

    def test_legacy_csv_is_archived_when_schema_changes(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(logging_utils, "LOG_DIR", Path(tmp)):
            path = Path(tmp) / "parallel.csv"
            path.write_text("timestamp;value\nold;kept\n", encoding="utf-8")
            csv_logger = logging_utils.CsvLogger("parallel", fieldnames=["operation_id", "value"], key="operation_id")
            csv_logger.log({"operation_id": "new", "value": "first"})
            csv_logger.log({"operation_id": "new", "value": "final"})
            self.assertEqual((Path(tmp) / "parallel.csv.1").read_text(encoding="utf-8"), "timestamp;value\nold;kept\n")
            with path.open(encoding="utf-8", newline="") as file:
                self.assertEqual(list(csv.DictReader(file, delimiter=";")), [{"operation_id": "new", "value": "final"}])

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

    def test_publication_failures_preserve_index_and_restart(self):
        original_write = Path.write_text
        original_dumps = json.dumps

        def fail_write(path, text, *args, **kwargs):
            if path.name == "meta.json":
                raise OSError("metadata failure")
            return original_write(path, text, *args, **kwargs)

        def fail_pointer_write(path, text, *args, **kwargs):
            if path.name.startswith("current.") and path.suffix == ".tmp":
                raise OSError("temporary pointer failure")
            return original_write(path, text, *args, **kwargs)

        def fail_row(value, *args, **kwargs):
            if isinstance(value, dict) and value.get("id") == "new":
                raise OSError("row failure")
            return original_dumps(value, *args, **kwargs)

        failures = (
            ("snapshot", lambda: patch("rag.data_store.Snapshot", side_effect=RuntimeError("snapshot"))),
            ("rows", lambda: patch("rag.data_store.json.dumps", side_effect=fail_row)),
            ("vectors", lambda: patch("rag.data_store.faiss.write_index", side_effect=OSError("vectors"))),
            ("checksums", lambda: patch("rag.data_store._digest", side_effect=OSError("checksum"))),
            ("metadata", lambda: patch.object(Path, "write_text", new=fail_write)),
            ("temporary pointer", lambda: patch.object(Path, "write_text", new=fail_pointer_write)),
            ("pointer", lambda: patch("rag.data_store.os.replace", side_effect=OSError("pointer"))),
        )
        old = [{"id": "old", "text": "old text", "source": "old", "kind": "docs", "meta": {}}]
        new = [{"id": "new", "text": "new text", "source": "new", "kind": "docs", "meta": {}}]
        with tempfile.TemporaryDirectory() as tmp, patch.object(config, "DATA_DIR", Path(tmp)), \
                patch.object(data_store, "rows", []), patch.object(data_store, "_snapshot",
                                                                   data_store.Snapshot([], [], None, None)), \
                patch("rag.data_store.get_embed_model", return_value=FakeEmbedding()):
            data_store._commit(old)
            pointer_path = Path(tmp) / "current.json"
            pointer = pointer_path.read_bytes()
            for name, failure in failures:
                with self.subTest(step=name):
                    before = data_store.snapshot()
                    with failure(), self.assertRaises((RuntimeError, OSError)):
                        data_store._commit(new)
                    self.assertEqual(pointer_path.read_bytes(), pointer)
                    self.assertIs(data_store.snapshot(), before)
                    self.assertEqual(data_store.rows, old)
                    data_store.load_index()
                    self.assertEqual(data_store.snapshot().rows, old)
                    self.assertEqual(data_store.snapshot().index.ntotal, 1)

    def test_backup_restore_and_corrupt_index_rejection(self):
        row = {"id": "old", "text": "backup text", "source": "file", "kind": "docs", "meta": {}}
        with tempfile.TemporaryDirectory() as tmp, patch.object(data_store, "rows", []), \
                patch.object(data_store, "_snapshot", data_store.Snapshot([], [], None, None)), \
                patch("rag.data_store.get_embed_model", return_value=FakeEmbedding()):
            original, backup = Path(tmp) / "original", Path(tmp) / "backup"
            with patch.object(config, "DATA_DIR", original):
                data_store._commit([row])
            shutil.copytree(original, backup)
            with patch.object(config, "DATA_DIR", backup):
                data_store.load_index()
                self.assertEqual(data_store.snapshot().rows, [row])
                before = data_store.snapshot()
                version = json.loads((backup / "current.json").read_text())["version"]
                metadata_path = backup / "versions" / version / "meta.json"
                metadata = json.loads(metadata_path.read_text())
                invalid_metadata = {**metadata, "count": 2}
                metadata_path.write_text(json.dumps(invalid_metadata), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "sizes do not match"):
                    data_store.load_index()
                self.assertIs(data_store.snapshot(), before)
                metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
                wrong_index = unittest.mock.MagicMock(ntotal=2)
                with patch("rag.data_store.faiss.read_index", return_value=wrong_index):
                    with self.assertRaisesRegex(ValueError, "sizes do not match"):
                        data_store.load_index()
                self.assertIs(data_store.snapshot(), before)
                with patch("rag.data_store._bm25", side_effect=RuntimeError("BM25 failure")):
                    with self.assertRaisesRegex(RuntimeError, "BM25"):
                        data_store.load_index()
                self.assertIs(data_store.snapshot(), before)
                self.assertIs(data_store.rows, before.rows)
                (backup / "versions" / version / "rows.jsonl").write_text("broken", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "checksum"):
                    data_store.load_index()
                self.assertIs(data_store.snapshot(), before)
                self.assertEqual(data_store.rows, [row])

    def test_query_while_index_is_rebuilt(self):
        started, release = threading.Event(), threading.Event()
        errors = []
        old = {"id": "old", "text": "old lantern blue", "source": "lantern", "kind": "docs",
               "meta": {"path": "/lantern"}}
        new = {"id": "new", "text": "new lantern green", "source": "lantern", "kind": "docs",
               "meta": {"path": "/lantern"}}

        class BlockingEmbedding(FakeEmbedding):
            def encode(self, texts, **kwargs):
                started.set()
                if not release.wait(30):
                    raise TimeoutError("test did not release index build")
                return super().encode(texts, **kwargs)

        def rebuild():
            try:
                data_store.replace_chunks_by_paths({"/lantern"}, [new])
            except Exception as exc:
                errors.append(exc)

        llm = unittest.mock.MagicMock()
        llm.generate.return_value = LLMResp(content="answer")
        payload = {"messages": [{"role": "user", "content": "lantern"}], "top_k": 1}
        with tempfile.TemporaryDirectory() as tmp, patch.object(config, "DATA_DIR", Path(tmp)), \
                patch.object(config, "RAG_API_KEY", "test-secret"), patch.object(config, "BITNET_CONTEXT_SIZE", 0), \
                patch.object(config, "RETRIEVAL_SIM_THRESHOLD", 0), patch.object(config, "RERANKER_MIN_SCORE", None), \
                patch.object(data_store, "rows", []), patch.object(data_store, "_snapshot",
                                                                   data_store.Snapshot([], [], None, None)), \
                patch("rag.models.get_embed_model", return_value=FakeEmbedding()), \
                patch("rag.models.get_reranker", return_value=None), \
                patch("api.ask_service.get_llm", return_value=llm), patch("api.ask_service._log_csv"):
            with patch("rag.data_store.get_embed_model", return_value=FakeEmbedding()):
                data_store._commit([old])
            pointer = (Path(tmp) / "current.json").read_bytes()
            with patch("rag.data_store.get_embed_model", return_value=BlockingEmbedding()):
                writer = threading.Thread(target=rebuild)
                writer.start()
                try:
                    self.assertTrue(started.wait(10))
                    status, body = asyncio.run(asgi_request("POST", "/ask", payload, "test-secret"))
                    self.assertEqual(status, 200)
                    self.assertEqual(json.loads(body)["used_context"][0]["text"], old["text"])
                    self.assertTrue(writer.is_alive())
                    self.assertEqual((Path(tmp) / "current.json").read_bytes(), pointer)
                finally:
                    release.set()
                    writer.join(10)
                self.assertFalse(writer.is_alive())
            self.assertEqual(errors, [])
            self.assertNotEqual((Path(tmp) / "current.json").read_bytes(), pointer)
            status, body = asyncio.run(asgi_request("POST", "/ask", payload, "test-secret"))
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body)["used_context"][0]["text"], new["text"])
            data_store.load_index()
            self.assertEqual(data_store.snapshot().rows, [new])

    def test_site_partial_failure_keeps_previously_indexed_sources(self):
        base = "https://example.com/"
        pages = {
            base: '<p>' + 'root text ' * 30 + '</p><a href="/ok">ok</a><a href="/failed">bad</a><script src="/bad.json"></script>',
            base + "ok": '<p>' + 'new content ' * 30 + '</p>',
            base + "failed": None,
            base + "bad.json": "{",
        }
        old = [
            {"id": "old-ok", "text": "old ok", "source": base + "ok", "kind": "site",
             "meta": {"document_id": "site-html:" + base + "ok"}},
            {"id": "old-failed", "text": "old failed", "source": base + "failed", "kind": "site",
             "meta": {"document_id": "site-html:" + base + "failed"}},
            {"id": "old-json", "text": "old json", "source": base + "bad.json", "kind": "site",
             "meta": {"document_id": "site-json:" + base + "bad.json#root"}},
        ]
        with tempfile.TemporaryDirectory() as tmp, patch.object(config, "DATA_DIR", Path(tmp)), \
                patch.object(config, "RAG_API_KEY", "test-secret"), \
                patch.object(data_store, "rows", []), patch.object(data_store, "_snapshot",
                                                                   data_store.Snapshot([], [], None, None)), \
                patch("rag.data_store.get_embed_model", return_value=FakeEmbedding()), \
                patch("common.formatters.same_domain", return_value=True), \
                patch("common.formatters.fetch_text", side_effect=lambda url, **kwargs: pages.get(url)):
            data_store._commit(old)
            status, body = asyncio.run(asgi_request("POST", "/ingest/site", {"base_url": base}, "test-secret"))
            self.assertEqual(status, 200)
            self.assertCountEqual(json.loads(body)["failed_urls"], [base + "failed", base + "bad.json"])
            texts = [row["text"] for row in data_store.snapshot().rows]
            self.assertIn("old failed", texts)
            self.assertIn("old json", texts)
            self.assertNotIn("old ok", texts)
            self.assertTrue(any("new content" in text for text in texts))

    def test_site_source_without_chunks_is_not_replaced(self):
        old = {"id": "old", "text": "old text", "source": "https://example.com/empty", "kind": "site",
               "meta": {"document_id": "empty"}}
        documents = [
            FormattedDocument(text="  ", source=old["source"], kind="site", stable_id="empty"),
            FormattedDocument(text="new text", source="https://example.com/new", kind="site", stable_id="new"),
        ]
        with tempfile.TemporaryDirectory() as tmp, patch.object(config, "DATA_DIR", Path(tmp)), \
                patch.object(data_store, "rows", []), patch.object(data_store, "_snapshot",
                                                                   data_store.Snapshot([], [], None, None)), \
                patch("rag.data_store.get_embed_model", return_value=FakeEmbedding()), \
                patch("rag.ingest.site_to_documents", return_value=documents):
            data_store._commit([old])
            failed_urls: list[str] = []
            crawl_site(CrawlConfig(base_url="https://example.com"), failed_urls)
            self.assertEqual(failed_urls, [old["source"]])
            self.assertCountEqual([row["text"] for row in data_store.snapshot().rows], ["old text", "new text"])

    def test_missing_local_file_is_not_deleted_from_index(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(config, "DATA_DIR", Path(tmp) / "index"), \
                patch.object(data_store, "rows", []), patch.object(data_store, "_snapshot",
                                                                   data_store.Snapshot([], [], None, None)), \
                patch("rag.data_store.get_embed_model", return_value=FakeEmbedding()):
            folder = Path(tmp) / "docs"
            folder.mkdir()
            (folder / "one.txt").write_text("old one", encoding="utf-8")
            (folder / "two.txt").write_text("old two", encoding="utf-8")
            ingest_folder(str(folder), True)
            (folder / "one.txt").write_text("new one", encoding="utf-8")
            (folder / "two.txt").unlink()
            ingest_folder(str(folder), True)
            self.assertCountEqual([row["text"] for row in data_store.snapshot().rows], ["new one", "old two"])

    def test_file_ingest_writes_safe_technical_events(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(config, "DATA_DIR", Path(tmp) / "index"), \
                patch.object(data_store, "rows", []), patch.object(data_store, "_snapshot",
                                                                   data_store.Snapshot([], [], None, None)), \
                patch("rag.data_store.get_embed_model", return_value=FakeEmbedding()), \
                patch("rag.ingest.logger.info") as log:
            folder = Path(tmp) / "private-folder"
            folder.mkdir()
            (folder / "private-name.txt").write_text("private content", encoding="utf-8")
            self.assertEqual(ingest_folder(str(folder)), 1)
            messages = str(log.call_args_list)
            self.assertIn("files start", messages)
            self.assertIn("file processed", messages)
            self.assertIn("files done", messages)
            self.assertNotIn("private", messages)

    def test_site_minimum_length_uses_config(self):
        from common.formatters import site_to_documents
        with patch.object(config, "SITE_MIN_TEXT_CHARS", 5), \
                patch("common.formatters.fetch_text", return_value="<p>12345</p>"):
            self.assertEqual(len(site_to_documents("https://example.com", max_pages=1, timeout=1, user_agent="test")),
                             1)

    def test_logging_and_chunk_configuration_is_validated_at_startup(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {**os.environ, "DATA_DIR": tmp, "LOG_DIR": str(Path(tmp) / "custom-logs"),
                   "LOG_MAX_BYTES": "1024", "LOG_BACKUP_COUNT": "2", "LOG_LEVEL": "INFO",
                   "INGEST_CHUNK_CHARS": "500", "INGEST_CHUNK_OVERLAP": "50"}
            command = [sys.executable, "-B", "-c",
                       "from common import config; print(config.LOG_DIR, config.LOG_MAX_BYTES, config.LOG_LEVEL, config.INGEST_CHUNK_CHARS)"]
            result = subprocess.run(command, env=env, text=True, capture_output=True, check=True)
            self.assertIn("custom-logs 1024 INFO 500", result.stdout)
            for name, value, expected in (("LOG_LEVEL", "wrong", "LOG_LEVEL"),
                                          ("LOG_BACKUP_COUNT", "0", "LOG_BACKUP_COUNT"),
                                          ("INGEST_CHUNK_OVERLAP", "500", "INGEST_CHUNK_OVERLAP")):
                with self.subTest(name=name):
                    result = subprocess.run(command, env={**env, name: value}, text=True, capture_output=True)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(expected, result.stderr)

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
