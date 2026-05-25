import time

from fastapi import HTTPException
from starlette.concurrency import run_in_threadpool

from api.schemas import IngestFilesReq, IngestSiteReq
from core import data_store
from ingestion.ingest import ingest_folder, CrawlConfig, crawl_site


async def ingest_files(req: IngestFilesReq) -> dict:
    started_at = time.time()
    try:
        added = await run_in_threadpool(ingest_folder, req.folder)
        return {"added_chunks": added, "total_chunks": len(data_store.rows)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    finally:
        print("INGEST FILES TIME:", time.time() - started_at)


async def ingest_site(req: IngestSiteReq) -> dict:

    if not req.base_url.startswith("http"):
        raise HTTPException(status_code=400, detail="base_url must start with http/https")

    cfg = CrawlConfig(base_url=req.base_url, max_pages=req.max_pages)
    try:
        added = await run_in_threadpool(crawl_site, cfg)
        return {"added_chunks": added, "total_chunks": len(data_store.rows)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
