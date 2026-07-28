"""Structured JSON HTTP request/response logging middleware."""

import json
import logging
from datetime import datetime, timezone
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("api.access")


class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    """Logs every HTTP request with structured fields (request ID, method, path, status, latency)."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)
        req_id = getattr(request.state, "request_id", "unknown")
        proc_time = getattr(request.state, "process_time_ms", 0.0)

        log_payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": req_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "latency_ms": round(proc_time, 2),
        }
        logger.info(json.dumps(log_payload))
        return response
