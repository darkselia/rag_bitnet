from fastapi import APIRouter, Query

from api import ask_service, ingest_service
from api.schemas import AskReq, IngestFilesReq, IngestSiteReq

router = APIRouter()


@router.post("/ingest/files")
async def ingest_files(req: IngestFilesReq):
    return await ingest_service.ingest_files(req)


@router.post("/ingest/site")
async def ingest_site(req: IngestSiteReq):
    return await ingest_service.ingest_site(req)


@router.post("/ask")
async def ask(req: AskReq, response_type: str = Query(default="json", alias="type")):
    return await ask_service.ask(req, response_type)
