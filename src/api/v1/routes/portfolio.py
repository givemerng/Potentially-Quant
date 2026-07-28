"""Target Portfolio & Risk Optimization API Routes."""

from fastapi import APIRouter, Depends, Request
from src.api.v1.schemas.envelope import APIResponse
from src.api.v1.dependencies import get_portfolio_app_service
from src.application.services import PortfolioApplicationService

router = APIRouter(prefix="/api/v1/portfolio", tags=["Portfolio Construction"])


@router.get("/current", response_model=APIResponse[dict])
async def get_current_portfolio(
    request: Request,
    app_service: PortfolioApplicationService = Depends(get_portfolio_app_service),
):
    """Retrieve recommended target stock weights and sector breakdown."""
    req_id = getattr(request.state, "request_id", None)
    data = app_service.get_current_portfolio_weights()
    return APIResponse.ok(data=data, request_id=req_id)
