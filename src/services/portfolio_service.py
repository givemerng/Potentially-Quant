from __future__ import annotations

from typing import Dict, Any, Optional, Tuple
import pandas as pd

from src.data.repositories.portfolio_repository import PortfolioRepository
from src.data.repositories.optimization_repository import OptimizationRepository
from src.data.repositories.risk_repository import RiskRepository
from src.portfolio.execution.walk_forward import WalkForwardOptimizer
from src.experiments.tracking import ExperimentTracker
from src.experiments.registry import ExperimentRegistry


class PortfolioService:
    """Service orchestrating backtest strategy execution, walk-forward portfolio optimization, and risk analytics."""

    def __init__(
        self,
        portfolio_repo: Optional[PortfolioRepository] = None,
        opt_repo: Optional[OptimizationRepository] = None,
        risk_repo: Optional[RiskRepository] = None,
    ) -> None:
        self.portfolio_repo = portfolio_repo or PortfolioRepository()
        self.opt_repo = opt_repo or OptimizationRepository()
        self.risk_repo = risk_repo or RiskRepository()
        self.walk_forward_optimizer = WalkForwardOptimizer()
        self.tracker = ExperimentTracker()
        self.registry = ExperimentRegistry()

    def get_backtest_results(self, backtest_name: Optional[str] = None) -> pd.DataFrame:
        return self.portfolio_repo.get_backtest_results(backtest_name=backtest_name)

    def get_composite_scores(
        self,
        method: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> pd.DataFrame:
        return self.portfolio_repo.get_composite_scores(method=method, run_id=run_id)

    def run_walk_forward_optimization(
        self,
        composite_scores_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        metadata_df: pd.DataFrame,
        config: Dict[str, Any],
        experiment_name: str = "week9_cvar_opt",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """Orchestrate walk-forward optimization, log experiment tracking, and save to DB."""
        # 1. Experiment tracking
        exp_record = self.tracker.create_experiment_record(
            experiment_name=experiment_name,
            config_dict=config,
            dataset_summary=f"scores_{len(composite_scores_df)}_returns_{len(returns_df)}",
        )

        # 2. Run walk-forward optimization
        weights_df, logs_df = self.walk_forward_optimizer.run(
            composite_scores_df=composite_scores_df,
            returns_df=returns_df,
            metadata_df=metadata_df,
            config=config,
        )

        # 3. Save logs to optimization repository
        if not logs_df.empty:
            self.opt_repo.save_optimization_logs(logs_df, run_name=experiment_name)

        return weights_df, logs_df, exp_record

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
