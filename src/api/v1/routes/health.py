"""Three-Tier Health Check Routes (/health/live, /health/ready, /health/startup)."""

from fastapi import APIRouter, Request
from src.api.v1.schemas.envelope import APIResponse
from src.data.db import check_readiness, check_liveness

router = APIRouter(prefix="/health", tags=["Health Checks"])


@router.get("/live", response_model=APIResponse[dict])
async def liveness_probe(request: Request):
    """Liveness probe: verifies process health."""
    req_id = getattr(request.state, "request_id", None)
    return APIResponse.ok(data=check_liveness(), request_id=req_id)


@router.get("/ready", response_model=APIResponse[dict])
async def readiness_probe(request: Request):
    """Readiness probe: executes SELECT 1 against Neon PostgreSQL."""
    req_id = getattr(request.state, "request_id", None)
    readiness = check_readiness()
    if readiness.get("status") == "ready":
        return APIResponse.ok(data=readiness, request_id=req_id)
    return APIResponse.fail(errors=readiness, request_id=req_id)


@router.get("/startup", response_model=APIResponse[dict])
async def startup_probe(request: Request):
    """Startup probe: verifies database readiness and process state."""
    req_id = getattr(request.state, "request_id", None)
    readiness = check_readiness()
    is_ready = readiness.get("status") == "ready"
    startup_data = {
        "status": "started" if is_ready else "starting",
        "database": readiness.get("database"),
        "timestamp": readiness.get("timestamp"),
    }
    if is_ready:
        return APIResponse.ok(data=startup_data, request_id=req_id)
    return APIResponse.fail(errors=startup_data, request_id=req_id)
