"""Application Service for Target Portfolio & Backtest Metrics."""

from __future__ import annotations
import logging
from typing import Dict, Any, List, Optional
import pandas as pd

from src.services.portfolio_service import PortfolioService
from src.services.market_data_service import MarketDataService
from src.cache.base import BaseCache


class PortfolioApplicationService:
    """Application Service wrapping PortfolioService for target weight and backtest queries."""

    def __init__(
        self,
        portfolio_service: Optional[PortfolioService] = None,
        market_service: Optional[MarketDataService] = None,
        cache: Optional[BaseCache] = None,
    ) -> None:
        self.portfolio_service = portfolio_service or PortfolioService()
        self.market_service = market_service or MarketDataService()
        self.cache = cache

    def get_current_portfolio_weights(self) -> Dict[str, Any]:
        cache_key = "api:v1:portfolio:current"
        if self.cache:
            cached_val = self.cache.get(cache_key)
            if cached_val:
                return cached_val

        weights_df = self.portfolio_service.get_backtest_results()
        metadata_df = self.market_service.get_ticker_metadata()
        sector_map = metadata_df.set_index("ticker")["sector"].to_dict() if not metadata_df.empty else {}

        if weights_df.empty or "weight" not in weights_df.columns:
            result = {"rebalance_date": None, "holdings_count": 0, "weights": [], "sector_breakdown": {}}
        else:
            weights_df["date"] = pd.to_datetime(weights_df["date"])
            latest_date = weights_df["date"].max()
            latest_weights = weights_df[weights_df["date"] == latest_date]

            weights_list = []
            sector_breakdown: Dict[str, float] = {}

            for _, row in latest_weights.iterrows():
                ticker = str(row["ticker"])
                w_val = float(row["weight"])
                sec = sector_map.get(ticker, "Unknown")

                weights_list.append({"ticker": ticker, "weight": w_val, "sector": sec})
                sector_breakdown[sec] = sector_breakdown.get(sec, 0.0) + w_val

            result = {
                "rebalance_date": str(latest_date.date()),
                "holdings_count": len(weights_list),
                "weights": weights_list,
                "sector_breakdown": sector_breakdown,
            }

        if self.cache:
            self.cache.set(cache_key, result, ttl_seconds=86400)

        return result

    def get_backtest_metrics_summary(self) -> Dict[str, Any]:
        cache_key = "api:v1:backtest:metrics"
        if self.cache:
            cached_val = self.cache.get(cache_key)
            if cached_val:
                return cached_val

        with self.portfolio_service.portfolio_repo.engine.connect() as conn:
            from sqlalchemy import text
            df = pd.read_sql(text("SELECT backtest_name, metric_name, value FROM backtest_metrics"), con=conn)

        if df.empty:
            result = {"strategies": {}}
        else:
            strategies = {}
            for backtest_name, group in df.groupby("backtest_name"):
                metrics_dict = {row["metric_name"]: float(row["value"]) for _, row in group.iterrows()}
                strategies[str(backtest_name)] = metrics_dict
            result = {"strategies": strategies}

        if self.cache:
            self.cache.set(cache_key, result, ttl_seconds=86400)

        return result
