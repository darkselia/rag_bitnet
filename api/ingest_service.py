import time

from fastapi import HTTPException
from starlette.concurrency import run_in_threadpool

from common.logger import current_operation, get_logger, operation_update
from rag import data_store
from rag.ingest import CrawlConfig, IngestInputError, crawl_site, ingest_folder
from schemas.app_schema import IngestFilesReq, IngestSiteReq

logger = get_logger("api")
ingest_logger = get_logger("ingest")


async def ingest_files(req: IngestFilesReq) -> dict:
    started_at = time.time()
    try:
        added = await run_in_threadpool(ingest_folder, req.folder, req.replace_existing)
        operation_update(source_type="files", added_chunks=added)
        return {"added_chunks": added, "total_chunks": len(data_store.rows)}
    except IngestInputError as e:
        ingest_logger.warning("files failed category=invalid_input error_type=%s", type(e.__cause__ or e).__name__)
        operation_update(source_type="files", error_category="invalid_input",
                         error_type=type(e.__cause__ or e).__name__)
        raise HTTPException(status_code=400, detail="Invalid or unreadable input files") from e
    except Exception as e:
        ingest_logger.warning("files failed category=internal_error error_type=%s", type(e).__name__)
        operation_update(source_type="files", error_category="ingest_internal_error", error_type=type(e).__name__)
        raise HTTPException(status_code=500, detail="File ingestion failed") from e
    finally:
        logger.info("INGEST FILES TIME: %s", time.time() - started_at)


async def ingest_site(req: IngestSiteReq) -> dict:

    if not req.base_url.startswith("http"):
        operation_update(source_type="site", error_category="invalid_input")
        raise HTTPException(status_code=400, detail="base_url must start with http/https")

    cfg = CrawlConfig(base_url=req.base_url, max_pages=req.max_pages)
    try:
        failed_urls: list[str] = []
        added = await run_in_threadpool(crawl_site, cfg, failed_urls)
        operation = current_operation.get()
        if operation is not None and "index_version" not in operation:
            state = data_store.snapshot()
            operation_update(index_version=state.version, total_chunks=len(state.rows))
        operation_update(source_type="site", added_chunks=added, failed_sources=len(failed_urls))
        if failed_urls:
            operation_update(status="partial", error_category="source_processing_failed")
        return {"added_chunks": added, "total_chunks": len(data_store.rows), "failed_urls": failed_urls}
    except Exception as e:
        operation_update(source_type="site", error_category="ingest_internal_error", error_type=type(e).__name__)
        raise HTTPException(status_code=500, detail="Site ingestion failed") from e
