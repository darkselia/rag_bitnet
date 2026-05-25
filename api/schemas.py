from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class IngestFilesReq(BaseModel):
    folder: str


class IngestSiteReq(BaseModel):
    base_url: str
    max_pages: int = Field(100, ge=1, le=2000)


class AskReq(BaseModel):
    query: str
    top_k: int = 6
    sources: Optional[List[str]] = Field(default_factory=lambda: ["docs", "site"])
    site_domain: Optional[str] = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)


class AskResp(BaseModel):
    answer: str
    sources: List[str] = Field(default_factory=list)
    used_context: List[Dict]
