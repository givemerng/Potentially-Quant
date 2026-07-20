from __future__ import annotations

from typing import Any, Dict, List, Optional
import pandas as pd
from sqlalchemy import select

from src.data.db import (
    factors_table,
    factor_metrics_table,
    fama_french_table,
    macro_table,
)
from src.data.repositories.base import BaseRepository, parse_date_arg
from src.data.repositories.interfaces import IFactorsRepository


class FactorsRepository(BaseRepository, IFactorsRepository):
    """Repository for factor scores, evaluation metrics, Fama-French benchmark factors, and macro series."""

    def get_factor_scores(
        self,
        factor_name: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        stmt = select(
            factors_table.c.date,
            factors_table.c.ticker,
            factors_table.c.factor_name,
            factors_table.c.raw_score,
            factors_table.c.winsorized_score,
            factors_table.c.z_score,
            factors_table.c.final_score,
        )

        conditions = []
        dt_start = parse_date_arg(start_date)
        dt_end = parse_date_arg(end_date)

        if factor_name:
            conditions.append(factors_table.c.factor_name == factor_name)
        if dt_start:
            conditions.append(factors_table.c.date >= dt_start)
        if dt_end:
            conditions.append(factors_table.c.date <= dt_end)

        if conditions:
            stmt = stmt.where(*conditions)

        stmt = stmt.order_by(factors_table.c.date, factors_table.c.ticker)
        return self.fetch_dataframe(stmt)

    def get_factor_metrics(
        self,
        factor_name: Optional[str] = None,
        stage: Optional[str] = None,
    ) -> pd.DataFrame:
        stmt = select(
            factor_metrics_table.c.date,
            factor_metrics_table.c.factor_name,
            factor_metrics_table.c.stage,
            factor_metrics_table.c.metric_name,
            factor_metrics_table.c.value,
        )

        conditions = []
        if factor_name:
            conditions.append(factor_metrics_table.c.factor_name == factor_name)
        if stage:
            conditions.append(factor_metrics_table.c.stage == stage)

        if conditions:
            stmt = stmt.where(*conditions)

        return self.fetch_dataframe(stmt)

    def get_fama_french(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        stmt = select(
            fama_french_table.c.date,
            fama_french_table.c.mkt_rf,
            fama_french_table.c.smb,
            fama_french_table.c.hml,
            fama_french_table.c.rmw,
            fama_french_table.c.cma,
            fama_french_table.c.rf,
        )

        dt_start = parse_date_arg(start_date)
        dt_end = parse_date_arg(end_date)

        if dt_start:
            stmt = stmt.where(fama_french_table.c.date >= dt_start)
        if dt_end:
            stmt = stmt.where(fama_french_table.c.date <= dt_end)

        stmt = stmt.order_by(fama_french_table.c.date)
        return self.fetch_dataframe(stmt)

    def get_macro_series(
        self,
        series_names: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        stmt = select(
            macro_table.c.date,
            macro_table.c.series_name,
            macro_table.c.value,
        )

        conditions = []
        dt_start = parse_date_arg(start_date)
        dt_end = parse_date_arg(end_date)

        if series_names:
            conditions.append(macro_table.c.series_name.in_(series_names))
        if dt_start:
            conditions.append(macro_table.c.date >= dt_start)
        if dt_end:
            conditions.append(macro_table.c.date <= dt_end)

        if conditions:
            stmt = stmt.where(*conditions)

        stmt = stmt.order_by(macro_table.c.date)
        return self.fetch_dataframe(stmt)

    def save_factor_scores(self, rows: List[Dict[str, Any]]) -> int:
        return self.bulk_upsert(factors_table, rows)

    def save_factor_metrics(self, rows: List[Dict[str, Any]]) -> int:
        return self.bulk_upsert(factor_metrics_table, rows)

    def save_fama_french(self, rows: List[Dict[str, Any]]) -> int:
        return self.bulk_upsert(fama_french_table, rows)

    def save_macro(self, rows: List[Dict[str, Any]]) -> int:
        return self.bulk_upsert(macro_table, rows)
