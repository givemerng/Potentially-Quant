from __future__ import annotations

from typing import Any, Dict, List, Optional
import pandas as pd
from sqlalchemy import select

from src.data.db import (
    portfolio_weights_table,
    trades_table,
    backtest_results_table,
    backtest_metrics_table,
    combination_runs_table,
    composite_scores_table,
    combination_weights_table,
    combination_metrics_table,
)
from src.data.repositories.base import BaseRepository
from src.data.repositories.interfaces import IPortfolioRepository


class PortfolioRepository(BaseRepository, IPortfolioRepository):
    """Repository for managing portfolio backtest results, trades, weights, and ML combination runs."""

    def get_backtest_results(self, backtest_name: Optional[str] = None) -> pd.DataFrame:
        stmt = select(
            backtest_results_table.c.date,
            backtest_results_table.c.backtest_name,
            backtest_results_table.c.gross_return,
            backtest_results_table.c.net_return,
            backtest_results_table.c.portfolio_value,
            backtest_results_table.c.drawdown,
            backtest_results_table.c.turnover,
            backtest_results_table.c.transaction_costs,
        )
        if backtest_name:
            stmt = stmt.where(backtest_results_table.c.backtest_name == backtest_name)

        stmt = stmt.order_by(backtest_results_table.c.date)
        return self.fetch_dataframe(stmt)

    def get_composite_scores(
        self,
        method: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> pd.DataFrame:
        stmt = select(
            composite_scores_table.c.date,
            composite_scores_table.c.ticker,
            composite_scores_table.c.method,
            composite_scores_table.c.run_id,
            composite_scores_table.c.composite_score,
        )
        conditions = []
        if method:
            conditions.append(composite_scores_table.c.method == method)
        if run_id:
            conditions.append(composite_scores_table.c.run_id == run_id)

        if conditions:
            stmt = stmt.where(*conditions)

        stmt = stmt.order_by(composite_scores_table.c.date, composite_scores_table.c.ticker)
        return self.fetch_dataframe(stmt)

    def save_backtest_results(
        self,
        results_df: pd.DataFrame,
        weights_df: pd.DataFrame,
        trades_df: pd.DataFrame,
        metrics_df: pd.DataFrame,
        backtest_name: str,
    ) -> Dict[str, int]:
        counts = {}
        if not results_df.empty:
            res_rows = results_df.to_dict(orient="records")
            counts["results"] = self.bulk_upsert(backtest_results_table, res_rows)

        if not weights_df.empty:
            w_rows = weights_df.to_dict(orient="records")
            counts["weights"] = self.bulk_upsert(portfolio_weights_table, w_rows)

        if not trades_df.empty:
            t_rows = trades_df.to_dict(orient="records")
            counts["trades"] = self.bulk_upsert(trades_table, t_rows)

        if not metrics_df.empty:
            m_rows = metrics_df.to_dict(orient="records")
            counts["metrics"] = self.bulk_upsert(backtest_metrics_table, m_rows)

        return counts

    def save_combination_run(
        self,
        run_id: str,
        method: str,
        config_hash: str,
        train_start: str,
        train_end: str,
        test_start: str,
        test_end: str,
        created_at: str,
    ) -> int:
        row = [{
            "run_id": run_id,
            "method": method,
            "config_hash": config_hash,
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "created_at": created_at,
        }]
        return self.bulk_upsert(combination_runs_table, row)

    def save_composite_scores(self, rows: List[Dict[str, Any]]) -> int:
        return self.bulk_upsert(composite_scores_table, rows)

    def save_combination_weights(self, rows: List[Dict[str, Any]]) -> int:
        return self.bulk_upsert(combination_weights_table, rows)

    def save_combination_metrics(self, rows: List[Dict[str, Any]]) -> int:
        return self.bulk_upsert(combination_metrics_table, rows)
