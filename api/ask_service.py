import json
import threading
import time
from collections.abc import Iterator
from datetime import datetime, timezone
from itertools import chain
from typing import Dict, List

import requests
from fastapi import HTTPException
from starlette.concurrency import run_in_threadpool
from starlette.responses import StreamingResponse

from clients.llm import get_llm
from common import config
from common.logger import CsvLogger, current_operation, get_logger, operation_update
from schemas.app_schema import AskReq, AskResp, LLMResp, Message

logger = get_logger("ask_service")
NO_CONTEXT_ANSWER = "Couldn't find the answer in the database."


class _TrackedStream:
    def __init__(self, pieces: Iterator[LLMResp]) -> None:
        self.pieces = pieces
        self.lock = threading.Lock()

    def __iter__(self) -> Iterator[LLMResp]:
        return self

    def __next__(self) -> LLMResp:
        with self.lock:
            return next(self.pieces)

    def close(self) -> None:
        with self.lock:
            close = getattr(self.pieces, "close", None)
            if close is not None:
                close()

CSV_FIELDS = [
    "operation_id", "timestamp", "retrieve time", "user question", "contest",
    "llm time", "llm answer", "id",
    "prompt_n", "prompt_ms", "prompt_per_token_ms", "prompt_per_second",
    "predicted_n", "predicted_ms", "predicted_per_token_ms", "predicted_per_second",
    "model",
]
csv_logger = CsvLogger("ask_service", fieldnames=CSV_FIELDS, key="operation_id")


def _log_csv(row: dict) -> None:
    operation = current_operation.get()
    row = {"operation_id": operation["operation_id"] if operation else row["timestamp"], **row}
    if not config.LOG_REQUEST_CONTENT:
        row = {k: v for k, v in row.items() if k not in {"user question", "contest", "llm answer"}}
    csv_logger.log(row)


def _fmt(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else ""


def _build_context_text(contexts: List[Dict]) -> str:
    return "\n\n---\n\n".join(f"[{c['id']}]\n{c['text']}" for c in contexts)


def _extract_query(messages: List[Message]) -> str:
    for msg in reversed(messages):
        if msg.role == "user":
            return msg.content
    raise HTTPException(status_code=400, detail="No user message found in messages")


def _no_context_response(
        stream: bool, query: str, retrieve_elapsed: float, timestamp: str,
) -> AskResp | StreamingResponse:
    response = AskResp(answer=Message(role="assistant", content=NO_CONTEXT_ANSWER), used_context=[])
    logger.info("answer: no context")
    _log_csv({
        "timestamp": timestamp,
        "retrieve time": f"{retrieve_elapsed:.3f}",
        "user question": query,
        "contest": "",
    })
    if not stream:
        return response
    chunk = LLMResp(content=NO_CONTEXT_ANSWER).model_dump(exclude={"final"})
    events = (json.dumps(event, ensure_ascii=False) + "\n" for event in (chunk, response.model_dump()))
    return StreamingResponse(events, media_type="application/x-ndjson; charset=utf-8")


def _stream_answer(
        pieces: Iterator, ctx_text: str, contexts: List[Dict],
        query: str, retrieve_elapsed: float,
        timestamp: str, llm_start: float | None = None,
) -> Iterator[str]:
    chunks: list[str] = []
    final_id = ""
    final_timings = None
    final_model = ""
    if llm_start is None:
        llm_start = time.monotonic()
    finished = False
    try:
        for piece in pieces:
            if piece.final:
                final_id = piece.id
                final_timings = piece.timings
                final_model = piece.model
                finished = True
                continue
            chunks.append(piece.content)
            yield json.dumps(piece.model_dump(exclude={"final"}), ensure_ascii=False) + "\n"
        if not finished:
            raise ValueError("LLM stream ended without a final event")
    except Exception as e:
        logger.warning("LLM stream failed error_type=%s", type(e).__name__)
        operation_update(status="error", error_category="llm_stream_error", error_type=type(e).__name__)
        yield json.dumps({"error": "LLM stream failed"}) + "\n"
        return

    llm_elapsed = time.monotonic() - llm_start
    operation_update(generation_ms=round(llm_elapsed * 1000, 2))
    answer = "".join(chunks)
    logger.info(
        "llm: mode=%s model=%s prompt_n=%s predicted_n=%s elapsed=%.3fs",
        config.BITNET_MODE, final_model,
        final_timings.prompt_n if final_timings else "?",
        final_timings.predicted_n if final_timings else "?",
        llm_elapsed,
    )
    logger.info("answer: id=%s", final_id)
    yield json.dumps(AskResp(
        id=final_id,
        answer=Message(role="assistant", content=answer),
        timings=final_timings,
        used_context=contexts,
    ).model_dump(), ensure_ascii=False) + "\n"

    _log_csv({
        "timestamp": timestamp,
        "retrieve time": f"{retrieve_elapsed:.3f}",
        "user question": query,
        "contest": ctx_text,
        "llm time": f"{llm_elapsed:.3f}",
        "llm answer": answer,
        "id": final_id,
        "prompt_n": final_timings.prompt_n if final_timings else "",
        "prompt_ms": _fmt(final_timings.prompt_ms) if final_timings else "",
        "prompt_per_token_ms": _fmt(final_timings.prompt_per_token_ms) if final_timings else "",
        "prompt_per_second": _fmt(final_timings.prompt_per_second) if final_timings else "",
        "predicted_n": final_timings.predicted_n if final_timings else "",
        "predicted_ms": _fmt(final_timings.predicted_ms) if final_timings else "",
        "predicted_per_token_ms": _fmt(final_timings.predicted_per_token_ms) if final_timings else "",
        "predicted_per_second": _fmt(final_timings.predicted_per_second) if final_timings else "",
        "model": final_model,
    })


async def ask(req: AskReq) -> AskResp | StreamingResponse:
    from rag.retrieval import MIN_SEARCH_CANDIDATES, QueryTooLongError, retrieve_async

    operation_update(response_mode="stream" if req.stream else "json")
    query = _extract_query(req.messages)
    timestamp = datetime.now(timezone.utc).isoformat()
    logger.info(
        "request top_k=%s temperature=%s stream=%s",
        req.top_k, req.temperature, req.stream,
    )

    retrieve_start = time.monotonic()
    candidate_count = max(MIN_SEARCH_CANDIDATES, req.top_k)
    try:
        contexts = await retrieve_async(
            query, candidate_count, candidate_count, req.top_k,
            config.RETRIEVAL_SIM_THRESHOLD, req.sources, req.site_domain,
        )
    except QueryTooLongError as e:
        operation_update(error_category="invalid_input", error_type=type(e).__name__)
        raise HTTPException(status_code=413, detail="Question exceeds the reranker token limit") from e
    finally:
        retrieve_elapsed = time.monotonic() - retrieve_start
        operation_update(retrieval_ms=round(retrieve_elapsed * 1000, 2))
    logger.info("retrieve: contexts=%d elapsed=%.3fs", len(contexts), retrieve_elapsed)

    if not contexts:
        return _no_context_response(req.stream, query, retrieve_elapsed, timestamp)

    messages = [Message(role="system", content=config.SYSTEM_PROMPT)] + req.messages
    llm = get_llm()
    if llm is None:
        raise HTTPException(status_code=500, detail="No LLM configured")

    if config.BITNET_CONTEXT_SIZE:
        while True:
            ctx_text = _build_context_text(contexts)
            try:
                count = await run_in_threadpool(llm.count_prompt_tokens, messages, ctx_text)
            except requests.Timeout as e:
                operation_update(error_category="llm_timeout", error_type=type(e).__name__)
                raise HTTPException(status_code=504, detail="LLM tokenizer timed out") from e
            except (requests.RequestException, ValueError, KeyError) as e:
                operation_update(error_category="llm_error", error_type=type(e).__name__)
                raise HTTPException(status_code=502, detail="Exact prompt tokenization unavailable") from e
            if count + config.LLM_MAX_TOKENS <= config.BITNET_CONTEXT_SIZE:
                if not contexts:
                    return _no_context_response(req.stream, query, retrieve_elapsed, timestamp)
                break
            if not contexts:
                raise HTTPException(status_code=413, detail="Messages exceed the model context window")
            contexts.pop()
    ctx_text = _build_context_text(contexts)
    _log_csv({
        "timestamp": timestamp,
        "retrieve time": f"{retrieve_elapsed:.3f}",
        "user question": query,
        "contest": ctx_text,
    })

    if req.stream:
        logger.info("stream: started")
        llm_start = time.monotonic()
        operation_update(_generation_started=llm_start)
        pieces = _TrackedStream(llm.generate_stream(messages, config.LLM_MAX_TOKENS, req.temperature, context=ctx_text))
        operation_update(_stream=pieces)
        try:
            first = await run_in_threadpool(next, pieces, None)
        except requests.Timeout as e:
            operation_update(error_category="llm_timeout", error_type=type(e).__name__)
            raise HTTPException(status_code=504, detail="LLM request timed out") from e
        except (requests.RequestException, ValueError) as e:
            operation_update(error_category="llm_error", error_type=type(e).__name__)
            raise HTTPException(status_code=502, detail="LLM request failed") from e
        if first is None:
            operation_update(error_category="llm_stream_error")
            raise HTTPException(status_code=502, detail="LLM stream ended unexpectedly")
        return StreamingResponse(
            _stream_answer(
                chain([first], pieces), ctx_text, contexts,
                query, retrieve_elapsed, timestamp, llm_start,
            ),
            media_type="application/x-ndjson; charset=utf-8",
        )

    llm_start = time.monotonic()
    operation_update(_generation_started=llm_start)
    try:
        resp = await run_in_threadpool(
            llm.generate, messages, config.LLM_MAX_TOKENS, req.temperature, context=ctx_text,
        )
    except requests.Timeout as e:
        operation_update(error_category="llm_timeout", error_type=type(e).__name__)
        raise HTTPException(status_code=504, detail="LLM request timed out") from e
    except (requests.RequestException, ValueError) as e:
        operation_update(error_category="llm_error", error_type=type(e).__name__)
        raise HTTPException(status_code=502, detail="LLM request failed") from e
    llm_elapsed = time.monotonic() - llm_start
    operation_update(generation_ms=round(llm_elapsed * 1000, 2))
    logger.info(
        "llm: mode=%s model=%s prompt_n=%s predicted_n=%s elapsed=%.3fs",
        config.BITNET_MODE, resp.model,
        resp.timings.prompt_n if resp.timings else "?",
        resp.timings.predicted_n if resp.timings else "?",
        llm_elapsed,
    )
    logger.info("answer: id=%s", resp.id)

    _log_csv({
        "timestamp": timestamp,
        "retrieve time": f"{retrieve_elapsed:.3f}",
        "user question": query,
        "contest": ctx_text,
        "llm time": f"{llm_elapsed:.3f}",
        "llm answer": resp.content,
        "id": resp.id,
        "prompt_n": resp.timings.prompt_n if resp.timings else "",
        "prompt_ms": _fmt(resp.timings.prompt_ms) if resp.timings else "",
        "prompt_per_token_ms": _fmt(resp.timings.prompt_per_token_ms) if resp.timings else "",
        "prompt_per_second": _fmt(resp.timings.prompt_per_second) if resp.timings else "",
        "predicted_n": resp.timings.predicted_n if resp.timings else "",
        "predicted_ms": _fmt(resp.timings.predicted_ms) if resp.timings else "",
        "predicted_per_token_ms": _fmt(resp.timings.predicted_per_token_ms) if resp.timings else "",
        "predicted_per_second": _fmt(resp.timings.predicted_per_second) if resp.timings else "",
        "model": resp.model,
    })

    return AskResp(
        id=resp.id,
        answer=Message(role="assistant", content=resp.content),
        timings=resp.timings,
        used_context=contexts,
    )
