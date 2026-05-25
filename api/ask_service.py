import re
import time
from typing import Dict, Iterator, List

from fastapi import HTTPException
from starlette.concurrency import run_in_threadpool
from starlette.responses import StreamingResponse

from api.schemas import AskReq, AskResp
from clients.llm import LLM, get_llm
from core import config
from core.prompt import build_prompt

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

    seen = set()
    deduped_source_ids = []
    for sid in source_ids:
        if sid in seen:
            continue
        seen.add(sid)
        deduped_source_ids.append(sid)

    if deduped_source_ids:
        clean_answer = f"{clean_answer}\n\nSources: " + ", ".join(f"[{sid}]" for sid in deduped_source_ids)

    return clean_answer, deduped_source_ids


async def _build_rag_prompt(req: AskReq) -> tuple[str | None, List[Dict], float]:
    from core.retrieval import retrieve_async

    started_at = time.time()
    contexts = await retrieve_async(req.query, 8, 8, req.top_k, 0.0, req.sources, req.site_domain)
    print("RETRIEVE TIME:", time.time() - started_at)
    if not contexts:
        return None, [], started_at

    prompt = build_prompt(req.query, contexts)
    print("PROMPT:", prompt)
    return prompt, contexts, time.time()


def _temperature(req: AskReq) -> float:
    return req.temperature if req.temperature is not None else config.LLM_TEMPERATURE


def _stream_answer(llm: LLM, prompt: str, req: AskReq, started_at: float) -> Iterator[str]:
    chunks: list[str] = []
    try:
        for piece in llm.generate_stream(prompt, config.LLM_MAX_TOKENS, _temperature(req)):
            chunks.append(piece)
            yield piece
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {e}") from e
    finally:
        answer = "".join(chunks)
        print("LLM ANSWER:", answer, time.time() - started_at)


async def ask_json(req: AskReq, llm: LLM | None = None) -> AskResp:
    prompt, contexts, started_at = await _build_rag_prompt(req)
    if not prompt:
        return AskResp(answer="Не нашёл ответа в базе.", sources=[], used_context=[])

    llm = llm or get_llm()
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
    print("LLM ANSWER:", answer, time.time() - started_at)
    return AskResp(answer=answer, sources=sources, used_context=contexts)


async def ask_stream(req: AskReq, llm: LLM | None = None) -> StreamingResponse:
    prompt, _, started_at = await _build_rag_prompt(req)
    if not prompt:
        return StreamingResponse(iter(["Не нашёл ответа в базе."]), media_type="text/plain; charset=utf-8")

    llm = llm or get_llm()
    return StreamingResponse(
        _stream_answer(llm, prompt, req, started_at),
        media_type="text/plain; charset=utf-8",
    )


async def ask(req: AskReq, response_type: str) -> AskResp | StreamingResponse:
    if response_type == "stream":
        return await ask_stream(req)
    if response_type == "json":
        return await ask_json(req)
    raise HTTPException(status_code=400, detail="type must be 'json' or 'stream'")
