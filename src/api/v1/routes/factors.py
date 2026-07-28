"""Alpha Factors API Routes."""

from fastapi import APIRouter, Depends, Request
from src.api.v1.schemas.envelope import APIResponse
from src.api.v1.dependencies import get_factor_app_service
from src.application.services import FactorApplicationService

router = APIRouter(prefix="/api/v1/factors", tags=["Alpha Factors"])


@router.get("/latest", response_model=APIResponse[dict])
async def get_latest_factors(
    request: Request,
    app_service: FactorApplicationService = Depends(get_factor_app_service),
):
    """Retrieve latest cross-sectional alpha factor scores for active universe."""
    req_id = getattr(request.state, "request_id", None)
    data = app_service.get_latest_factor_scores()
    return APIResponse.ok(data=data, request_id=req_id)
