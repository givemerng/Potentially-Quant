"""Application Service for Alpha Factor Operations."""

from __future__ import annotations
import logging
from typing import Dict, Any, List, Optional
import pandas as pd

from src.services.factor_service import FactorService
from src.cache.base import BaseCache


class FactorApplicationService:
    """Application Service wrapping FactorService with caching logic."""

    def __init__(self, factor_service: Optional[FactorService] = None, cache: Optional[BaseCache] = None) -> None:
        self.factor_service = factor_service or FactorService()
        self.cache = cache

    def get_latest_factor_scores(self) -> Dict[str, Any]:
        cache_key = "api:v1:factors:latest"
        if self.cache:
            cached_val = self.cache.get(cache_key)
            if cached_val:
                return cached_val

        # Query latest factor scores from database
        with self.factor_service.factors_repo.engine.connect() as conn:
            from sqlalchemy import text
            df = pd.read_sql(text("SELECT date, ticker, factor_name, final_score FROM factors"), con=conn)

        if df.empty:
            result = {"evaluation_date": None, "factors_count": 0, "scores": []}
        else:
            df["date"] = pd.to_datetime(df["date"])
            latest_date = df["date"].max()
            latest_df = df[df["date"] == latest_date]

            scores_list = latest_df[["ticker", "factor_name", "final_score"]].to_dict(orient="records")
            result = {
                "evaluation_date": str(latest_date.date()),
                "factors_count": len(latest_df["factor_name"].unique()),
                "scores": scores_list,
            }

        if self.cache:
            self.cache.set(cache_key, result, ttl_seconds=86400)

        return result
