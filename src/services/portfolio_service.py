from __future__ import annotations

from typing import Dict, Any, Optional
import pandas as pd

from src.data.repositories.portfolio_repository import PortfolioRepository


class PortfolioService:
    """Service orchestrating backtest strategy execution persistence, combination runs, and metrics."""

    def __init__(self, portfolio_repo: Optional[PortfolioRepository] = None) -> None:
        self.portfolio_repo = portfolio_repo or PortfolioRepository()

    def get_backtest_results(self, backtest_name: Optional[str] = None) -> pd.DataFrame:
        return self.portfolio_repo.get_backtest_results(backtest_name=backtest_name)

    def get_composite_scores(
        self,
        method: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> pd.DataFrame:
        return self.portfolio_repo.get_composite_scores(method=method, run_id=run_id)

    def save_backtest_results(
        self,
        results_df: pd.DataFrame,
        weights_df: pd.DataFrame,
        trades_df: pd.DataFrame,
        metrics_df: pd.DataFrame,
        backtest_name: str,
    ) -> Dict[str, int]:
        return self.portfolio_repo.save_backtest_results(
            results_df=results_df,
            weights_df=weights_df,
            trades_df=trades_df,
            metrics_df=metrics_df,
            backtest_name=backtest_name,
        )
