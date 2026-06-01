from fastapi import FastAPI

from api.routes import router
from common.logger import setup_logging, get_console_logger

setup_logging()
app = FastAPI(title="RAG + BitNet", version="0.1.0")
app.include_router(router)

logger = get_console_logger("app")


@app.on_event("startup")
async def startup_event():
    logger.info("RAG BitNet server started. Models load lazily on first use.")
