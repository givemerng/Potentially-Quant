"""FastAPI Dependency Injections for Application Services and Caching."""

from __future__ import annotations
from typing import Generator
from fastapi import Depends

from src.config import AppConfig
from src.cache import get_cache_provider, BaseCache
from src.application.services import (
    RegimeApplicationService,
    FactorApplicationService,
    PortfolioApplicationService,
)

_global_cache: BaseCache | None = None


def get_cache() -> BaseCache:
    global _global_cache
    if _global_cache is None:
        cfg = AppConfig.from_yaml("config/config.yaml")
        _global_cache = get_cache_provider(
            provider_type=cfg.week10.cache_provider,
            redis_url=cfg.week10.redis_url,
        )
    return _global_cache


def get_regime_app_service(cache: BaseCache = Depends(get_cache)) -> RegimeApplicationService:
    return RegimeApplicationService(cache=cache)


def get_factor_app_service(cache: BaseCache = Depends(get_cache)) -> FactorApplicationService:
    return FactorApplicationService(cache=cache)


def get_portfolio_app_service(cache: BaseCache = Depends(get_cache)) -> PortfolioApplicationService:
    return PortfolioApplicationService(cache=cache)
