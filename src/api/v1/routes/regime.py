"""Market Regime API Routes."""

from fastapi import APIRouter, Depends, Request
from src.api.v1.schemas.envelope import APIResponse
from src.api.v1.dependencies import get_regime_app_service
from src.application.services import RegimeApplicationService

router = APIRouter(prefix="/api/v1/regime", tags=["Market Regime"])


@router.get("/current", response_model=APIResponse[dict])
async def get_current_regime(
    request: Request,
    app_service: RegimeApplicationService = Depends(get_regime_app_service),
):
    """Retrieve current HMM market regime state, label, and soft probability distribution."""
    req_id = getattr(request.state, "request_id", None)
    data = app_service.get_current_regime()
    return APIResponse.ok(data=data, request_id=req_id)
