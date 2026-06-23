import re
import time
from typing import Dict, Iterator, List

from fastapi import HTTPException
from starlette.concurrency import run_in_threadpool
from starlette.responses import StreamingResponse

from clients.llm import LLM, get_llm
from common import config
from common.logger import get_logger
from rag.prompt import build_prompt
from schemas.app_schema import AskReq, AskResp

logger = get_logger("api")

_SOURCES_LINE_RE = re.compile(r"(?:^|\n)\s*(?:Sources|Источники)\s*:\s*(.*)$", re.IGNORECASE)
_ID_RE = re.compile(r"\[([^\[\]]+)\]")


def _extract_source_ids(answer: str) -> List[str]:
    match = _SOURCES_LINE_RE.search(answer)
    if not match:
        return []
    tail = match.group(1)
    return [item.strip() for item in _ID_RE.findall(tail) if item.strip()]


def _normalize_answer(answer: str, contexts: List[Dict]) -> tuple[str, List[str]]:
    clean_answer = _SOURCES_LINE_RE.sub("", answer).strip()
    context_ids = {str(c.get("id", "")) for c in contexts}
    source_ids = [sid for sid in _extract_source_ids(answer) if sid in context_ids]
    if not source_ids:
        source_ids = [str(c.get("id", "")) for c in contexts if c.get("id")]

    seen = set()
    deduped_source_ids = []
    for sid in source_ids:
        if sid in seen:
            continue
        seen.add(sid)
        deduped_source_ids.append(sid)

    return clean_answer, deduped_source_ids


def _sources_line(contexts: List[Dict]) -> str:
    ids = []
    seen = set()
    for context in contexts:
        context_id = str(context.get("id", ""))
        if not context_id or context_id in seen:
            continue
        seen.add(context_id)
        ids.append(context_id)
    if not ids:
        return ""
    return "\n\nSources: " + ", ".join(f"[{context_id}]" for context_id in ids)


async def _build_rag_prompt(req: AskReq) -> tuple[str | None, List[Dict], float]:
    from rag.retrieval import retrieve_async

    started_at = time.time()
    contexts = await retrieve_async(req.query, 8, 8, req.top_k, 0.0, req.sources, req.site_domain)
    logger.info("RETRIEVE TIME: %s", time.time() - started_at)
    if not contexts:
        return None, [], started_at

    prompt = build_prompt(req.query, contexts)
    logger.debug("PROMPT: %s", prompt)
    return prompt, contexts, time.time()


def _temperature(req: AskReq) -> float:
    return req.temperature if req.temperature is not None else config.LLM_TEMPERATURE


def _stream_answer(llm: LLM, prompt: str, req: AskReq, contexts: List[Dict], started_at: float) -> Iterator[str]:
    chunks: list[str] = []
    try:
        for piece in llm.generate_stream(prompt, config.LLM_MAX_TOKENS, _temperature(req)):
            chunks.append(piece)
            yield piece
        answer = "".join(chunks)
        if not _SOURCES_LINE_RE.search(answer):
            line = _sources_line(contexts)
            if line:
                yield line
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {e}") from e
    finally:
        answer = "".join(chunks)
        logger.info("LLM TIME: elapsed=%s)", time.time() - started_at)
        logger.info("LLM ANSWER: %s", answer)


async def ask_json(req: AskReq, llm: LLM | None = None) -> AskResp | HTTPException:
    prompt, contexts, started_at = await _build_rag_prompt(req)
    if not prompt:
        return AskResp(answer="Couldn't find the answer in the database.", sources=[], used_context=[])

    llm = llm or get_llm()
    if llm is None:
        return HTTPException(status_code=500, detail=f"LLM error: no LLM configured")

    try:
        raw_answer = await run_in_threadpool(
            llm.generate,
            prompt,
            config.LLM_MAX_TOKENS,
            _temperature(req),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {e}") from e

    answer, sources = _normalize_answer(raw_answer, contexts)
    logger.info("LLM TIME: elapsed=%s)", time.time() - started_at)
    logger.info("LLM ANSWER: %s", answer)
    return AskResp(answer=answer, sources=sources, used_context=contexts)


async def ask_stream(req: AskReq, llm: LLM | None = None) -> StreamingResponse:
    prompt, contexts, started_at = await _build_rag_prompt(req)
    if not prompt:
        return StreamingResponse(iter(["Couldn't find the answer in the database."]),
                                 media_type="text/plain; charset=utf-8")

    llm = llm or get_llm()
    if llm is None:
        return StreamingResponse(iter(["Do not have llm"]),)
    return StreamingResponse(
        _stream_answer(llm, prompt, req, contexts, started_at),
        media_type="text/plain; charset=utf-8",
    )


async def ask(req: AskReq, response_type: str) -> StreamingResponse | AskResp | HTTPException:
    if response_type == "stream":
        return await ask_stream(req)
    if response_type == "json":
        return await ask_json(req)
    raise HTTPException(status_code=400, detail="type must be 'json' or 'stream'")
