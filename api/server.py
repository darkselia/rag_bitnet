from fastapi import FastAPI

from api.routes import router
from common import config
from common.logger import setup_logging, get_console_logger

setup_logging()
app = FastAPI(title="RAG + BitNet", version="0.1.0")
app.include_router(router)

logger = get_console_logger("app")


@app.on_event("startup")
async def startup_event():
    if not config.RAG_API_KEY:
        raise RuntimeError("RAG_API_KEY must be configured before starting the API")
    logger.info("RAG BitNet server started. Models load lazily on first use.")
