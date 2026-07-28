"""Standardized API Response Envelope Schema."""

from __future__ import annotations
from datetime import datetime, timezone
from typing import Any, Generic, Optional, TypeVar
from pydantic import BaseModel, Field

T = TypeVar("T")


class APIResponse(BaseModel, Generic[T]):
    """Standardized API Response Envelope wrapper for all REST endpoints."""

    success: bool = True
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    request_id: Optional[str] = None
    data: Optional[T] = None
    errors: Optional[Any] = None

    @classmethod
    def ok(cls, data: T, request_id: Optional[str] = None) -> APIResponse[T]:
        return cls(success=True, data=data, request_id=request_id, errors=None)

    @classmethod
    def fail(cls, errors: Any, request_id: Optional[str] = None) -> APIResponse[T]:
        return cls(success=False, data=None, request_id=request_id, errors=errors)
