"""Backtest Metrics API Routes."""

from fastapi import APIRouter, Depends, Request
from src.api.v1.schemas.envelope import APIResponse
from src.api.v1.dependencies import get_portfolio_app_service
from src.application.services import PortfolioApplicationService

router = APIRouter(prefix="/api/v1/backtest", tags=["Backtest Analytics"])


@router.get("/metrics", response_model=APIResponse[dict])
async def get_backtest_metrics(
    request: Request,
    app_service: PortfolioApplicationService = Depends(get_portfolio_app_service),
):
    """Retrieve comparative backtest performance metrics across strategies."""
    req_id = getattr(request.state, "request_id", None)
    data = app_service.get_backtest_metrics_summary()
    return APIResponse.ok(data=data, request_id=req_id)
