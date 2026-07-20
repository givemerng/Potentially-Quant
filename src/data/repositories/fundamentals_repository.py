from __future__ import annotations

from typing import Any, Dict, List, Optional
import pandas as pd
from sqlalchemy import select

from src.data.db import (
    fundamentals_table,
    earnings_calendar_table,
    insider_transactions_table,
)
from src.data.repositories.base import BaseRepository, parse_date_arg
from src.data.repositories.interfaces import IFundamentalsRepository


class FundamentalsRepository(BaseRepository, IFundamentalsRepository):
    """Repository for fundamental balance sheet data, earnings calendar, and insider trading."""

    def get_fundamentals(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        tickers: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        stmt = select(
            fundamentals_table.c.date,
            fundamentals_table.c.ticker,
            fundamentals_table.c.book_value,
            fundamentals_table.c.gross_profit,
            fundamentals_table.c.total_assets,
            fundamentals_table.c.eps,
            fundamentals_table.c.ebitda,
            fundamentals_table.c.total_debt,
            fundamentals_table.c.operating_cash_flow,
            fundamentals_table.c.capital_expenditures,
            fundamentals_table.c.net_income,
        )

        conditions = []
        dt_start = parse_date_arg(start_date)
        dt_end = parse_date_arg(end_date)

        if dt_start:
            conditions.append(fundamentals_table.c.date >= dt_start)
        if dt_end:
            conditions.append(fundamentals_table.c.date <= dt_end)
        if tickers:
            conditions.append(fundamentals_table.c.ticker.in_(tickers))

        if conditions:
            stmt = stmt.where(*conditions)

        stmt = stmt.order_by(fundamentals_table.c.date, fundamentals_table.c.ticker)
        return self.fetch_dataframe(stmt)

    def get_earnings_calendar(self, tickers: Optional[List[str]] = None) -> pd.DataFrame:
        stmt = select(
            earnings_calendar_table.c.ticker,
            earnings_calendar_table.c.date,
            earnings_calendar_table.c.eps_estimate,
            earnings_calendar_table.c.reported_eps,
            earnings_calendar_table.c.surprise_pct,
        )
        if tickers:
            stmt = stmt.where(earnings_calendar_table.c.ticker.in_(tickers))
        return self.fetch_dataframe(stmt)

    def get_insider_transactions(self, tickers: Optional[List[str]] = None) -> pd.DataFrame:
        stmt = select(
            insider_transactions_table.c.id,
            insider_transactions_table.c.ticker,
            insider_transactions_table.c.date,
            insider_transactions_table.c.insider,
            insider_transactions_table.c.position,
            insider_transactions_table.c.shares,
            insider_transactions_table.c.value,
            insider_transactions_table.c.text,
        )
        if tickers:
            stmt = stmt.where(insider_transactions_table.c.ticker.in_(tickers))
        return self.fetch_dataframe(stmt)

    def save_fundamentals(self, rows: List[Dict[str, Any]]) -> int:
        return self.bulk_upsert(fundamentals_table, rows)

    def save_earnings_calendar(self, rows: List[Dict[str, Any]]) -> int:
        return self.bulk_upsert(earnings_calendar_table, rows)

    def save_insider_transactions(self, rows: List[Dict[str, Any]]) -> int:
        return self.bulk_upsert(insider_transactions_table, rows)
