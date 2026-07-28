"""Security Authentication Module Placeholder for Future API Key / OAuth Integration."""

from __future__ import annotations
from typing import Optional
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: Optional[str] = Security(api_key_header)) -> bool:
    """Placeholder security dependency. Currently passes all requests for development."""
    return True
