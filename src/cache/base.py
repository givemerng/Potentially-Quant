"""Base Abstract Interface for Cache Providers."""

from __future__ import annotations
from abc import ABC, abstractmethod
import logging
from typing import Any, Optional


class BaseCache(ABC):
    """Abstract Base Class for application caching providers (Redis / Memory)."""

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Retrieve item from cache by key. Returns None if cache miss."""
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """Store item in cache with key and optional TTL."""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        pass

    @abstractmethod
    def clear(self) -> bool:
        """Clear all cache keys."""
        pass
