from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, Optional


@dataclass
class TickerMetadata:
    ticker: str
    sector: Optional[str] = None
    industry: Optional[str] = None
    shares_outstanding: Optional[float] = None


@dataclass
class CombinationRun:
    run_id: str
    method: str
    config_hash: Optional[str] = None
    train_start: Optional[date] = None
    train_end: Optional[date] = None
    test_start: Optional[date] = None
    test_end: Optional[date] = None
    created_at: Optional[date] = None


@dataclass
class BacktestResult:
    backtest_name: str
    cumulative_return_net: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    annualized_turnover: Optional[float] = None


@dataclass
class RegimeWeight:
    date: date
    factor_name: str
    run_id: str
    regime_probability: Optional[float] = None
    dynamic_weight: Optional[float] = None
    posterior_ic: Optional[float] = None


@dataclass
class PipelineSummary:
    ticker_count: int
    start_date: str
    end_date: str
    price_rows_upserted: int
    macro_rows_upserted: int
    metadata_rows_upserted: int = 0
    fundamentals_rows_upserted: int = 0
    ff_rows_upserted: int = 0
    earnings_calendar_rows_upserted: int = 0
    insider_transactions_rows_upserted: int = 0
    failed: bool = False

    @property
    def total_rows_upserted(self) -> int:
        return (
            self.price_rows_upserted
            + self.macro_rows_upserted
            + self.metadata_rows_upserted
            + self.fundamentals_rows_upserted
            + self.ff_rows_upserted
            + self.earnings_calendar_rows_upserted
            + self.insider_transactions_rows_upserted
        )
