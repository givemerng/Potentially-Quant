"""Application Service for Market Regime Operations with Caching Integration."""

from __future__ import annotations
import logging
from typing import Dict, Any, Optional
import pandas as pd

from src.services.regime_service import RegimeService
from src.cache.base import BaseCache


class RegimeApplicationService:
    """Application Service wrapping RegimeService with caching logic and DTO conversions."""

    def __init__(self, regime_service: Optional[RegimeService] = None, cache: Optional[BaseCache] = None) -> None:
        self.regime_service = regime_service or RegimeService()
        self.cache = cache

    def get_current_regime(self) -> Dict[str, Any]:
        cache_key = "api:v1:regime:current"
        if self.cache:
            cached_val = self.cache.get(cache_key)
            if cached_val:
                return cached_val

        df = self.regime_service.get_market_regimes()
        if df.empty:
            result = {
                "regime_id": 0,
                "regime_label": "Unknown / Uninitialized",
                "probabilities": {},
                "last_updated": None,
            }
        else:
            df["date"] = pd.to_datetime(df["date"])
            latest_row = df.sort_values("date").iloc[-1]
            prob_cols = [c for c in df.columns if c.startswith("prob_state_")]
            probs = {col: float(latest_row[col]) for col in prob_cols}

            result = {
                "regime_id": int(latest_row.get("regime_id", 0)),
                "regime_label": str(latest_row.get("regime_label", "Unknown")),
                "probabilities": probs,
                "last_updated": str(latest_row["date"].date()),
            }

        if self.cache:
            self.cache.set(cache_key, result, ttl_seconds=86400)

        return result
