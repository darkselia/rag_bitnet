import json
import time
from collections.abc import Iterator
from typing import Dict, List

from fastapi import HTTPException
from starlette.concurrency import run_in_threadpool
from starlette.responses import StreamingResponse

from clients.llm import get_llm
from common import config
from common.logger import get_logger
from schemas.app_schema import AskReq, AskResp, Message

logger = get_logger("api")


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
) -> Iterator[str]:
    chunks: list[str] = []
    final_id = ""
    final_timings = None
    try:
        for piece in llm.generate_stream(
                messages, max_tokens, temperature, context=ctx_text,
        ):
            if piece.timings is not None:
                final_id = piece.id
                final_timings = piece.timings
                continue
            chunks.append(piece.content)
            yield json.dumps(piece.model_dump(), ensure_ascii=False) + "\n"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {e}") from e

    answer = "".join(chunks)
    yield json.dumps(AskResp(
        id=final_id,
        answer=Message(role="assistant", content=answer),
        timings=final_timings,
        used_context=contexts,
    ).model_dump(exclude_none=True), ensure_ascii=False) + "\n"


async def ask(req: AskReq) -> AskResp | StreamingResponse:
    from rag.retrieval import retrieve_async

    query = _extract_query(req.messages)

    started_at = time.time()
    contexts = await retrieve_async(
        query, 8, 8, req.top_k, 0.0, req.sources, req.site_domain,
    )
    print(contexts)
    logger.info("RETRIEVE TIME: %.3fs", time.time() - started_at)
    logger.info("RETRIEVED %d contexts", len(contexts))

    if not contexts:
        raise HTTPException(status_code=404, detail="No relevant context found")

    ctx_text = _build_context_text(contexts)
    messages = [Message(role="system", content=config.SYSTEM_PROMPT)] + req.messages
    llm = get_llm()
    if llm is None:
        raise HTTPException(status_code=500, detail="No LLM configured")

    if req.stream:
        return StreamingResponse(
            _stream_answer(llm, messages, ctx_text, contexts, req.temperature, config.LLM_MAX_TOKENS),
            media_type="application/x-ndjson; charset=utf-8",
        )

    started_at = time.time()
    resp = await run_in_threadpool(
        llm.generate, messages, config.LLM_MAX_TOKENS, req.temperature, context=ctx_text,
    )
    print(resp)
    logger.info("LLM TIME: %.3fs", time.time() - started_at)
    logger.info("LLM ANSWER: %s", resp.content)
    return AskResp(
        id=resp.id,
        answer=Message(role="assistant", content=resp.content),
        timings=resp.timings,
        used_context=contexts,
    )
