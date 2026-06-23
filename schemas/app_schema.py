from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from common import config


class Message(BaseModel):
    role: str
    content: str


class Timings(BaseModel):
    prompt_n: int
    prompt_ms: float
    prompt_per_token_ms: float
    prompt_per_second: float
    predicted_n: int
    predicted_ms: float
    predicted_per_token_ms: float
    predicted_per_second: float


class IngestFilesReq(BaseModel):
    folder: str
    replace_existing: bool = False


class IngestSiteReq(BaseModel):
    base_url: str
    max_pages: int = Field(100, ge=1, le=2000)


class AskReq(BaseModel):
    messages: List[Message] = Field(min_length=1)
    stream: bool = False
    top_k: int = Field(default=6, ge=1, le=50)
    sources: Optional[List[str]] = Field(default_factory=lambda: ["docs", "site"])
    site_domain: Optional[str] = None
    temperature: float = Field(default=config.LLM_TEMPERATURE, ge=0.0, le=2.0)


class AskResp(BaseModel):
    id: str = ""
    answer: Message
    timings: Optional[Timings] = None
    used_context: List[Dict]


class LLMReq(BaseModel):
    messages: List[Message]
    context: str = ""
    temperature: float | None = None
    max_tokens: int | None = None


class LLMResp(BaseModel):
    id: str = ""
    content: str
    timings: Optional[Timings] = None
    model: str = ""
