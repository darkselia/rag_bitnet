import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from fastapi.responses import StreamingResponse

from clients.bitnet_client import bitnet_client
from core import config
from core.prompt import build_prompt


class AskReq(BaseModel):
    query: str
    top_k: int = 6
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)


router = APIRouter()


@router.post("/ask")
async def ask(req: AskReq):
    t = time.time()
    prompt = build_prompt(req.query)
    print("PROMPT:", prompt, time.time() - t)
    started_at = time.time()

    def stream_answer():
        chunks: list[str] = []
        try:
            temperature = req.temperature if req.temperature is not None else config.BITNET_TEMPERATURE
            for piece in bitnet_client.generate_stream(prompt, config.BITNET_MAX_TOKENS, temperature):
                chunks.append(piece)
                yield piece
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"BitNet error: {e}")
        finally:
            answer = "".join(chunks)
            print("BITNET ANSWER:", answer, time.time() - started_at)

    return StreamingResponse(stream_answer(), media_type="text/plain; charset=utf-8")
