"""API v1 Router exports."""

from src.api.v1.routes.health import router as health_router
from src.api.v1.routes.regime import router as regime_router
from src.api.v1.routes.factors import router as factors_router
from src.api.v1.routes.portfolio import router as portfolio_router
from src.api.v1.routes.backtest import router as backtest_router

__all__ = [
    "health_router",
    "regime_router",
    "factors_router",
    "portfolio_router",
    "backtest_router",
]
