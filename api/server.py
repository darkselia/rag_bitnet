import secrets
import threading
import time
import uuid
from datetime import datetime, timezone

from fastapi import FastAPI
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from api.routes import router
from common import config
from common.logger import current_operation, get_console_logger, log_operation, setup_logging

_ask_limit = threading.BoundedSemaphore(config.MAX_CONCURRENT_ASK)
_ingest_limit = threading.BoundedSemaphore(1)


class OperationMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        routes = {"/ask": "ask", "/ingest/files": "ingest_files", "/ingest/site": "ingest_site"}
        if scope["type"] != "http" or scope["method"] != "POST" or scope["path"] not in routes:
            await self.app(scope, receive, send)
            return
        started = time.monotonic()
        operation = {"operation_id": uuid.uuid4().hex, "operation": routes[scope["path"]]}
        token = current_operation.set(operation)
        log_operation({**operation, "event": "started", "timestamp": datetime.now(timezone.utc).isoformat()})
        response_status = None
        complete = False
        permit = None

        async def send_event(message: Message) -> None:
            nonlocal response_status, complete
            if message["type"] == "http.response.start":
                response_status = message["status"]
                message = {**message, "headers": [*message.get("headers", []),
                                                  (b"x-request-id", operation["operation_id"].encode("ascii"))]}
            await send(message)
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                complete = True

        try:
            key = Headers(raw=scope.get("headers", [])).get("x-api-key")
            if config.RAG_API_KEY and key and secrets.compare_digest(key.encode("utf-8"),
                                                                     config.RAG_API_KEY.encode("utf-8")):
                limit = _ask_limit if scope["path"] == "/ask" else _ingest_limit
                if not limit.acquire(blocking=False):
                    operation["error_category"] = "capacity_exceeded"
                    response = JSONResponse({"detail": "Service busy; retry later"}, status_code=503,
                                            headers={"Retry-After": "1"})
                    await response(scope, receive, send_event)
                    return
                permit = limit
            await self.app(scope, receive, send_event)
        except BaseException as exc:
            operation.setdefault("error_category",
                                 "interrupted" if not isinstance(exc, Exception) else "internal_error")
            operation["status"] = "error"
            raise
        finally:
            stream = operation.pop("_stream", None)
            if stream is not None and hasattr(stream, "close"):
                try:
                    await run_in_threadpool(stream.close)
                except Exception:
                    pass
            if permit is not None:
                permit.release()
            generation_started = operation.pop("_generation_started", None)
            if generation_started is not None and "generation_ms" not in operation:
                operation["generation_ms"] = round((time.monotonic() - generation_started) * 1000, 2)
            if "status" not in operation:
                if not complete:
                    operation.update(status="error", error_category="interrupted")
                elif response_status is None or response_status >= 400:
                    if response_status == 401:
                        category = "authentication"
                    elif response_status is not None and response_status < 500:
                        category = "invalid_request"
                    else:
                        category = "internal_error"
                    operation.update(status="error", error_category=operation.get("error_category", category))
                else:
                    operation["status"] = "success"
            log_operation({**operation, "event": "completed", "timestamp": datetime.now(timezone.utc).isoformat(),
                           "http_status": response_status,
                           "duration_ms": round((time.monotonic() - started) * 1000, 2)})
            current_operation.reset(token)


setup_logging()
app = FastAPI(title="RAG + BitNet", version="0.1.0")
app.add_middleware(OperationMiddleware)
app.include_router(router)

logger = get_console_logger("app")


@app.on_event("startup")
async def startup_event():
    if not config.RAG_API_KEY:
        raise RuntimeError("RAG_API_KEY must be configured before starting the API")
    logger.info("RAG BitNet server started. Models load lazily on first use.")
