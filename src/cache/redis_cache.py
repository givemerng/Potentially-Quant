"""Redis Cache Provider Implementation with Fallback Protection."""

from __future__ import annotations
import json
import logging
from typing import Any, Optional

import redis

from src.cache.base import BaseCache
from src.cache.memory_cache import MemoryCache


class RedisCache(BaseCache):
    """Redis cache provider with automatic fallback to MemoryCache if Redis is offline."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        logger: Optional[logging.Logger] = None,
    ) -> None:
        super().__init__(logger=logger)
        self.redis_url = redis_url
        self.fallback = MemoryCache(logger=self.logger)
        self._client: Optional[redis.Redis] = None

    @property
    def client(self) -> Optional[redis.Redis]:
        if self._client is None:
            try:
                client = redis.Redis.from_url(self.redis_url, decode_responses=True, socket_timeout=2.0)
                client.ping()
                self._client = client
                self.logger.info("Connected to Redis cache at %s", self.redis_url)
            except Exception as exc:
                self.logger.warning("Redis connection failed (%s). Falling back to MemoryCache.", exc)
                self._client = None
        return self._client

    def get(self, key: str) -> Optional[Any]:
        cli = self.client
        if cli is None:
            return self.fallback.get(key)

        try:
            val = cli.get(key)
            if val is None:
                return None
            return json.loads(val)
        except Exception as exc:
            self.logger.warning("Redis get error (%s). Falling back to MemoryCache.", exc)
            return self.fallback.get(key)

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        cli = self.client
        if cli is None:
            return self.fallback.set(key, value, ttl_seconds=ttl_seconds)

        try:
            serialized = json.dumps(value, default=str)
            if ttl_seconds and ttl_seconds > 0:
                cli.setex(key, ttl_seconds, serialized)
            else:
                cli.set(key, serialized)
            return True
        except Exception as exc:
            self.logger.warning("Redis set error (%s). Falling back to MemoryCache.", exc)
            return self.fallback.set(key, value, ttl_seconds=ttl_seconds)

    def delete(self, key: str) -> bool:
        cli = self.client
        if cli is None:
            return self.fallback.delete(key)

        try:
            res = cli.delete(key)
            return bool(res > 0)
        except Exception as exc:
            self.logger.warning("Redis delete error (%s).", exc)
            return self.fallback.delete(key)

    def clear(self) -> bool:
        cli = self.client
        if cli is None:
            return self.fallback.clear()

        try:
            cli.flushdb()
            return True
        except Exception as exc:
            self.logger.warning("Redis clear error (%s).", exc)
            return self.fallback.clear()
