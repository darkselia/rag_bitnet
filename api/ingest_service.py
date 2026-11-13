import time

from fastapi import HTTPException
from starlette.concurrency import run_in_threadpool

from common.logger import get_logger
from rag import data_store
from rag.ingest import ingest_folder, CrawlConfig, crawl_site
from schemas.app_schema import IngestFilesReq, IngestSiteReq

logger = get_logger("api")


async def ingest_files(req: IngestFilesReq) -> dict:
    started_at = time.time()
    try:
        added = await run_in_threadpool(ingest_folder, req.folder, req.replace_existing)
        return {"added_chunks": added, "total_chunks": len(data_store.rows)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    finally:
        logger.info("INGEST FILES TIME: %s", time.time() - started_at)


async def ingest_site(req: IngestSiteReq) -> dict:

    if not req.base_url.startswith("http"):
        raise HTTPException(status_code=400, detail="base_url must start with http/https")

    cfg = CrawlConfig(base_url=req.base_url, max_pages=req.max_pages)
    try:
        failed_urls: list[str] = []
        added = await run_in_threadpool(crawl_site, cfg, failed_urls)
        return {"added_chunks": added, "total_chunks": len(data_store.rows), "failed_urls": failed_urls}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
