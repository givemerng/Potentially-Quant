"""Thread-safe In-Memory Cache Implementation with TTL Expiration."""

from __future__ import annotations
import logging
import time
from threading import Lock
from typing import Any, Dict, Optional, Tuple

from src.cache.base import BaseCache


class MemoryCache(BaseCache):
    """In-Memory cache provider with thread safety and automatic TTL expiration."""

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        super().__init__(logger=logger)
        self._store: Dict[str, Tuple[Any, float]] = {}
        self._lock = Lock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self._store:
                return None
            val, expire_at = self._store[key]
            if expire_at > 0 and time.time() > expire_at:
                del self._store[key]
                return None
            return val

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        expire_at = (time.time() + ttl_seconds) if ttl_seconds and ttl_seconds > 0 else 0.0
        with self._lock:
            self._store[key] = (value, expire_at)
        return True

    def delete(self, key: str) -> bool:
        with self._lock:
            if key in self._store:
                del self._store[key]
                return True
        return False

    def clear(self) -> bool:
        with self._lock:
            self._store.clear()
        return True
