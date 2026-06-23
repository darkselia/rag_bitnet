from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class IngestFilesReq(BaseModel):
    folder: str
    replace_existing: bool = False


class IngestSiteReq(BaseModel):
    base_url: str
    max_pages: int = Field(100, ge=1, le=2000)


class AskReq(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=6, ge=1, le=50)
    sources: Optional[List[str]] = Field(default_factory=lambda: ["docs", "site"])
    site_domain: Optional[str] = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)


class Message(BaseModel):
    role: str
    content: str


class LLMReq(BaseModel):
    messages: List[Message]
    context: str = ""
    temperature: float | None = None
    max_tokens: int | None = None


class AskResp(BaseModel):
    answer: str
    sources: List[str] = Field(default_factory=list)
    used_context: List[Dict]


class Timings(BaseModel):
    prompt_n: int
    prompt_ms: float
    prompt_per_token_ms: float
    prompt_per_second: float
    predicted_n: int
    predicted_ms: float
    predicted_per_token_ms: float
    predicted_per_second: float


class LLMResp(BaseModel):
    id: str = ""
    content: str
    timings: Optional[Timings] = None
