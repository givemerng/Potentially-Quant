from __future__ import annotations

from typing import Any, Dict, List, Optional
import pandas as pd
from sqlalchemy import select

from src.data.db import prices_table, returns_table, universe_membership_table
from src.data.repositories.base import BaseRepository, parse_date_arg
from src.data.repositories.interfaces import IPricesRepository


class PricesRepository(BaseRepository, IPricesRepository):
    """Repository for managing daily prices, calculated returns, and universe membership."""

    def get_daily_prices(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        tickers: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        stmt = select(
            prices_table.c.date,
            prices_table.c.ticker,
            prices_table.c.open,
            prices_table.c.high,
            prices_table.c.low,
            prices_table.c.close,
            prices_table.c.adj_close,
            prices_table.c.volume,
        )

        conditions = []
        dt_start = parse_date_arg(start_date)
        dt_end = parse_date_arg(end_date)

        if dt_start:
            conditions.append(prices_table.c.date >= dt_start)
        if dt_end:
            conditions.append(prices_table.c.date <= dt_end)
        if tickers:
            conditions.append(prices_table.c.ticker.in_(tickers))

        if conditions:
            stmt = stmt.where(*conditions)

        stmt = stmt.order_by(prices_table.c.date, prices_table.c.ticker)
        return self.fetch_dataframe(stmt, index_col=["date", "ticker"])

    def get_returns(
        self,
        freq: str = "monthly",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        stmt = select(
            returns_table.c.date,
            returns_table.c.ticker,
            returns_table.c.log_return,
            returns_table.c.simple_return,
        ).where(returns_table.c.freq == freq)

        dt_start = parse_date_arg(start_date)
        dt_end = parse_date_arg(end_date)

        if dt_start:
            stmt = stmt.where(returns_table.c.date >= dt_start)
        if dt_end:
            stmt = stmt.where(returns_table.c.date <= dt_end)

        stmt = stmt.order_by(returns_table.c.date, returns_table.c.ticker)
        return self.fetch_dataframe(stmt, index_col=["date", "ticker"])

    def get_universe_membership(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        stmt = select(
            universe_membership_table.c.date,
            universe_membership_table.c.ticker,
            universe_membership_table.c.in_universe,
            universe_membership_table.c.market_cap_proxy,
            universe_membership_table.c.avg_dollar_vol,
            universe_membership_table.c.days_since_first_price,
        )

        dt_start = parse_date_arg(start_date)
        dt_end = parse_date_arg(end_date)

        if dt_start:
            stmt = stmt.where(universe_membership_table.c.date >= dt_start)
        if dt_end:
            stmt = stmt.where(universe_membership_table.c.date <= dt_end)

        stmt = stmt.order_by(universe_membership_table.c.date, universe_membership_table.c.ticker)
        return self.fetch_dataframe(stmt)

    def save_prices(self, rows: List[Dict[str, Any]]) -> int:
        return self.bulk_upsert(prices_table, rows)

    def save_returns(self, rows: List[Dict[str, Any]]) -> int:
        return self.bulk_upsert(returns_table, rows)

    def save_universe_membership(self, rows: List[Dict[str, Any]]) -> int:
        return self.bulk_upsert(universe_membership_table, rows)
