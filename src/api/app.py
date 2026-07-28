"""FastAPI Application Factory."""

from __future__ import annotations
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import AppConfig
from src.middleware import (
    RequestIDMiddleware,
    TimingMiddleware,
    StructuredLoggingMiddleware,
    register_exception_handlers,
)
from src.api.v1.routes import (
    health_router,
    regime_router,
    factors_router,
    portfolio_router,
    backtest_router,
)


@asynccontextmanager
async def app_lifespan(app: FastAPI):
    """FastAPI application lifespan context manager for startup and shutdown events."""
    logger = logging.getLogger("api.lifecycle")
    logger.info("FastAPI Application initialized and running.")
    yield
    logger.info("FastAPI Application shutting down.")


def create_app(config: AppConfig | None = None) -> FastAPI:
    """Create and configure FastAPI application instance."""
    cfg = config or AppConfig.from_yaml("config/config.yaml")

    app = FastAPI(
        title=cfg.week10.api_title,
        version=cfg.week10.api_version,
        description="Production REST API for Regime-Adaptive Multi-Factor Quant Research Engine",
        lifespan=app_lifespan,
    )

    # Add Middleware Stack
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.week10.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(StructuredLoggingMiddleware)
    app.add_middleware(TimingMiddleware)
    app.add_middleware(RequestIDMiddleware)

    # Register Exception Handlers
    register_exception_handlers(app)

    # Include API Routers
    app.include_router(health_router)
    app.include_router(regime_router)
    app.include_router(factors_router)
    app.include_router(portfolio_router)
    app.include_router(backtest_router)

    return app


app = create_app()
