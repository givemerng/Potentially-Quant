from __future__ import annotations

from typing import Any, Dict, List, Optional
import pandas as pd
from sqlalchemy import select

from src.data.db import (
    market_regimes_table,
    regime_factor_ic_table,
    regime_factor_weights_table,
    adaptive_runs_table,
)
from src.data.repositories.base import BaseRepository, parse_date_arg
from src.data.repositories.interfaces import IRegimesRepository


class RegimesRepository(BaseRepository, IRegimesRepository):
    """Repository for HMM market regimes, Bayesian posterior ICs, and dynamic factor weights."""

    def get_market_regimes(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        stmt = select(
            market_regimes_table.c.date,
            market_regimes_table.c.regime_id,
            market_regimes_table.c.regime_label,
            market_regimes_table.c.prob_0,
            market_regimes_table.c.prob_1,
            market_regimes_table.c.prob_2,
            market_regimes_table.c.prob_3,
        )

        conditions = []
        dt_start = parse_date_arg(start_date)
        dt_end = parse_date_arg(end_date)

        if dt_start:
            conditions.append(market_regimes_table.c.date >= dt_start)
        if dt_end:
            conditions.append(market_regimes_table.c.date <= dt_end)

        if conditions:
            stmt = stmt.where(*conditions)

        stmt = stmt.order_by(market_regimes_table.c.date)
        return self.fetch_dataframe(stmt)

    def get_regime_factor_ic(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        stmt = select(
            regime_factor_ic_table.c.date,
            regime_factor_ic_table.c.factor_name,
            regime_factor_ic_table.c.regime_id,
            regime_factor_ic_table.c.regime_label,
            regime_factor_ic_table.c.sample_ic,
            regime_factor_ic_table.c.rolling_icir,
            regime_factor_ic_table.c.posterior_ic,
            regime_factor_ic_table.c.posterior_variance,
            regime_factor_ic_table.c.effective_sample_size,
        )

        conditions = []
        dt_start = parse_date_arg(start_date)
        dt_end = parse_date_arg(end_date)

        if dt_start:
            conditions.append(regime_factor_ic_table.c.date >= dt_start)
        if dt_end:
            conditions.append(regime_factor_ic_table.c.date <= dt_end)

        if conditions:
            stmt = stmt.where(*conditions)

        stmt = stmt.order_by(regime_factor_ic_table.c.date, regime_factor_ic_table.c.factor_name)
        return self.fetch_dataframe(stmt)

    def get_regime_factor_weights(self, run_id: Optional[str] = None) -> pd.DataFrame:
        stmt = select(
            regime_factor_weights_table.c.date,
            regime_factor_weights_table.c.factor_name,
            regime_factor_weights_table.c.run_id,
            regime_factor_weights_table.c.regime_probability,
            regime_factor_weights_table.c.dynamic_weight,
            regime_factor_weights_table.c.posterior_ic,
        )
        if run_id:
            stmt = stmt.where(regime_factor_weights_table.c.run_id == run_id)

        stmt = stmt.order_by(regime_factor_weights_table.c.date, regime_factor_weights_table.c.factor_name)
        return self.fetch_dataframe(stmt)

    def save_market_regimes(self, rows: List[Dict[str, Any]]) -> int:
        return self.bulk_upsert(market_regimes_table, rows)

    def save_regime_factor_ic(self, rows: List[Dict[str, Any]]) -> int:
        return self.bulk_upsert(regime_factor_ic_table, rows)

    def save_regime_factor_weights(self, rows: List[Dict[str, Any]]) -> int:
        return self.bulk_upsert(regime_factor_weights_table, rows)

    def save_adaptive_run(self, rows: List[Dict[str, Any]]) -> int:
        return self.bulk_upsert(adaptive_runs_table, rows)
