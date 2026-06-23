from fastapi import APIRouter

from api import ask_service, ingest_service
from schemas.app_schema import AskReq, IngestFilesReq, IngestSiteReq

router = APIRouter()


@router.post("/ingest/files")
async def ingest_files(req: IngestFilesReq):
    return await ingest_service.ingest_files(req)


@router.post("/ingest/site")
async def ingest_site(req: IngestSiteReq):
    return await ingest_service.ingest_site(req)


@router.post("/ask")
async def ask(req: AskReq):
    return await ask_service.ask(req)
