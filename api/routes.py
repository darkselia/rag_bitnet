import secrets

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import APIKeyHeader

from api import ask_service, ingest_service
from common import config
from schemas.app_schema import AskReq, IngestFilesReq, IngestSiteReq

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(api_key: str | None = Depends(api_key_header)) -> None:
    if not config.RAG_API_KEY or not api_key or not secrets.compare_digest(
            api_key.encode("utf-8"), config.RAG_API_KEY.encode("utf-8")
    ):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


router = APIRouter(dependencies=[Depends(require_api_key)])


@router.post("/ingest/files")
async def ingest_files(req: IngestFilesReq):
    return await ingest_service.ingest_files(req)


@router.post("/ingest/site")
async def ingest_site(req: IngestSiteReq):
    return await ingest_service.ingest_site(req)


@router.post("/ask")
async def ask(req: AskReq):
    return await ask_service.ask(req)
