import json
import time
from collections.abc import Iterator
from datetime import datetime, timezone
from typing import Dict, List

from fastapi import HTTPException
from starlette.concurrency import run_in_threadpool
from starlette.responses import StreamingResponse

from clients.llm import get_llm
from common import config
from common.logger import CsvLogger, get_logger
from schemas.app_schema import AskReq, AskResp, Message

logger = get_logger("ask_service")

CSV_FIELDS = [
    "timestamp", "retrieve time", "user question", "contest",
    "llm time", "llm answer", "id",
    "prompt_n", "prompt_ms", "prompt_per_token_ms", "prompt_per_second",
    "predicted_n", "predicted_ms", "predicted_per_token_ms", "predicted_per_second",
    "model",
]
csv_logger = CsvLogger("ask_service", fieldnames=CSV_FIELDS, key="timestamp")


def _build_context_text(contexts: List[Dict]) -> str:
    return "\n\n---\n\n".join(f"[{c['id']}]\n{c['text']}" for c in contexts)


def _extract_query(messages: List[Message]) -> str:
    for msg in reversed(messages):
        if msg.role == "user":
            return msg.content
    raise HTTPException(status_code=400, detail="No user message found in messages")


def _stream_answer(
        llm, messages: List[Message], ctx_text: str, contexts: List[Dict],
        temperature: float, max_tokens: int,
        query: str, retrieve_elapsed: float,
        timestamp: str,
) -> Iterator[str]:
    chunks: list[str] = []
    final_id = ""
    final_timings = None
    final_model = ""
    llm_start = time.time()
    try:
        for piece in llm.generate_stream(
                messages, max_tokens, temperature, context=ctx_text,
        ):
            if piece.timings is not None:
                final_id = piece.id
                final_timings = piece.timings
                final_model = piece.model
                continue
            chunks.append(piece.content)
            yield json.dumps(piece.model_dump(), ensure_ascii=False) + "\n"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {e}") from e

    llm_elapsed = time.time() - llm_start
    answer = "".join(chunks)
    preview = answer[:80] + ("..." if len(answer) > 80 else "")
    logger.info(
        "llm: mode=%s model=%s prompt_n=%s predicted_n=%s elapsed=%.3fs",
        config.BITNET_MODE, final_model,
        final_timings.prompt_n if final_timings else "?",
        final_timings.predicted_n if final_timings else "?",
        llm_elapsed,
    )
    logger.info("answer: id=%s preview=%s", final_id, preview)
    yield json.dumps(AskResp(
        id=final_id,
        answer=Message(role="assistant", content=answer),
        timings=final_timings,
        used_context=contexts,
    ).model_dump(exclude_none=True), ensure_ascii=False) + "\n"

    csv_logger.log({
        "timestamp": timestamp,
        "retrieve time": f"{retrieve_elapsed:.3f}",
        "user question": query,
        "contest": ctx_text,
        "llm time": f"{llm_elapsed:.3f}",
        "llm answer": answer,
        "id": final_id,
        "prompt_n": final_timings.prompt_n if final_timings else "",
        "prompt_ms": f"{final_timings.prompt_ms:.3f}" if final_timings else "",
        "prompt_per_token_ms": f"{final_timings.prompt_per_token_ms:.3f}" if final_timings else "",
        "prompt_per_second": f"{final_timings.prompt_per_second:.3f}" if final_timings else "",
        "predicted_n": final_timings.predicted_n if final_timings else "",
        "predicted_ms": f"{final_timings.predicted_ms:.3f}" if final_timings else "",
        "predicted_per_token_ms": f"{final_timings.predicted_per_token_ms:.3f}" if final_timings else "",
        "predicted_per_second": f"{final_timings.predicted_per_second:.3f}" if final_timings else "",
        "model": final_model,
    })


async def ask(req: AskReq) -> AskResp | StreamingResponse:
    from rag.retrieval import retrieve_async

    query = _extract_query(req.messages)
    timestamp = datetime.now(timezone.utc).isoformat()
    logger.info(
        "request query=%r sources=%s top_k=%s temperature=%s stream=%s",
        query, req.sources, req.top_k, req.temperature, req.stream,
    )

    retrieve_start = time.time()
    contexts = await retrieve_async(
        query, 8, 8, req.top_k, 0.0, req.sources, req.site_domain,
    )
    retrieve_elapsed = time.time() - retrieve_start
    logger.info("retrieve: contexts=%d elapsed=%.3fs", len(contexts), retrieve_elapsed)

    if not contexts:
        raise HTTPException(status_code=404, detail="No relevant context found")

    ctx_text = _build_context_text(contexts)
    csv_logger.log({
        "timestamp": timestamp,
        "retrieve time": f"{retrieve_elapsed:.3f}",
        "user question": query,
        "contest": ctx_text,
    })

    messages = [Message(role="system", content=config.SYSTEM_PROMPT)] + req.messages
    llm = get_llm()
    if llm is None:
        raise HTTPException(status_code=500, detail="No LLM configured")

    if req.stream:
        logger.info("stream: started")
        return StreamingResponse(
            _stream_answer(
                llm, messages, ctx_text, contexts,
                req.temperature, config.LLM_MAX_TOKENS,
                query, retrieve_elapsed, timestamp,
            ),
            media_type="application/x-ndjson; charset=utf-8",
        )

    llm_start = time.time()
    resp = await run_in_threadpool(
        llm.generate, messages, config.LLM_MAX_TOKENS, req.temperature, context=ctx_text,
    )
    llm_elapsed = time.time() - llm_start
    preview = resp.content[:80] + ("..." if len(resp.content) > 80 else "")
    logger.info(
        "llm: mode=%s model=%s prompt_n=%s predicted_n=%s elapsed=%.3fs",
        config.BITNET_MODE, resp.model,
        resp.timings.prompt_n if resp.timings else "?",
        resp.timings.predicted_n if resp.timings else "?",
        llm_elapsed,
    )
    logger.info("answer: id=%s preview=%s", resp.id, preview)

    csv_logger.log({
        "timestamp": timestamp,
        "retrieve time": f"{retrieve_elapsed:.3f}",
        "user question": query,
        "contest": ctx_text,
        "llm time": f"{llm_elapsed:.3f}",
        "llm answer": resp.content,
        "id": resp.id,
        "prompt_n": resp.timings.prompt_n if resp.timings else "",
        "prompt_ms": f"{resp.timings.prompt_ms:.3f}" if resp.timings else "",
        "prompt_per_token_ms": f"{resp.timings.prompt_per_token_ms:.3f}" if resp.timings else "",
        "prompt_per_second": f"{resp.timings.prompt_per_second:.3f}" if resp.timings else "",
        "predicted_n": resp.timings.predicted_n if resp.timings else "",
        "predicted_ms": f"{resp.timings.predicted_ms:.3f}" if resp.timings else "",
        "predicted_per_token_ms": f"{resp.timings.predicted_per_token_ms:.3f}" if resp.timings else "",
        "predicted_per_second": f"{resp.timings.predicted_per_second:.3f}" if resp.timings else "",
        "model": resp.model,
    })

    return AskResp(
        id=resp.id,
        answer=Message(role="assistant", content=resp.content),
        timings=resp.timings,
        used_context=contexts,
    )
