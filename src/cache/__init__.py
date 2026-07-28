"""Cache Provider Factory and Exports."""

from __future__ import annotations
import logging
from typing import Optional

from src.cache.base import BaseCache
from src.cache.memory_cache import MemoryCache
from src.cache.redis_cache import RedisCache


def get_cache_provider(
    provider_type: str = "memory",
    redis_url: str = "redis://localhost:6379/0",
    logger: Optional[logging.Logger] = None,
) -> BaseCache:
    """Factory helper for instantiating configured cache provider."""
    key = provider_type.lower().strip()
    if key == "redis":
        return RedisCache(redis_url=redis_url, logger=logger)
    return MemoryCache(logger=logger)


__all__ = ["BaseCache", "MemoryCache", "RedisCache", "get_cache_provider"]
