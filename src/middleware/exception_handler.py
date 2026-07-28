"""Global Exception Handler returning standardized JSON response envelopes."""

import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.api.v1.schemas.envelope import APIResponse

logger = logging.getLogger("api.exceptions")


def register_exception_handlers(app: FastAPI) -> None:
    """Register unhandled exception handlers on FastAPI app."""

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        req_id = getattr(request.state, "request_id", "unknown")
        logger.error("Unhandled exception [req_id=%s] on %s: %s", req_id, request.url.path, exc, exc_info=True)
        envelope = APIResponse.fail(errors=str(exc), request_id=req_id)
        return JSONResponse(status_code=500, content=envelope.model_dump())
