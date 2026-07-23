"""Walk-Forward Portfolio Rebalancing Loop."""

from __future__ import annotations
import logging
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np

from src.portfolio.risk_models.covariance import get_covariance_estimator, BaseCovarianceEstimator
from src.portfolio.optimizers.base import BasePortfolioOptimizer, OptimizationResult


class WalkForwardOptimizer:
    """Executes walk-forward rolling portfolio optimization without lookahead bias."""

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def _get_optimizer(self, optimizer_type: str) -> BasePortfolioOptimizer:
        from src.portfolio.optimizers.cvar_optimizer import CVaROptimizer
        from src.portfolio.optimizers.mean_variance import MeanVarianceOptimizer
        from src.portfolio.optimizers.risk_parity import RiskParityOptimizer

        key = optimizer_type.lower().strip()
        if key == "cvar":
            return CVaROptimizer(logger=self.logger)
        elif key == "mean_variance":
            return MeanVarianceOptimizer(logger=self.logger)
        elif key == "risk_parity":
            return RiskParityOptimizer(logger=self.logger)
        else:
            return CVaROptimizer(logger=self.logger)

    def run(
        self,
        composite_scores_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        metadata_df: pd.DataFrame,
        config: Dict[str, Any],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run walk-forward optimization.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            (Rebalance Weights DataFrame, Optimization Run Metadata DataFrame)
        """
        cov_name = config.get("covariance_model", "ledoit_wolf")
        opt_type = config.get("optimizer_type", "cvar")
        cov_lookback = config.get("cov_lookback_days", 252)

        cov_estimator = get_covariance_estimator(cov_name, logger=self.logger)
        optimizer = self._get_optimizer(opt_type)

        sector_map = pd.Series(dtype=str)
        if "ticker" in metadata_df.columns and "sector" in metadata_df.columns:
            sector_map = metadata_df.set_index("ticker")["sector"]

        # Sort dates
        composite_scores_df["date"] = pd.to_datetime(composite_scores_df["date"])
        eval_dates = sorted(composite_scores_df["date"].unique())

        all_weights: List[Dict[str, Any]] = []
        run_logs: List[Dict[str, Any]] = []

        current_weights: Optional[pd.Series] = None

        for eval_date in eval_dates:
            cross_section = composite_scores_df[composite_scores_df["date"] == eval_date]
            if cross_section.empty:
                continue

            active_alpha = cross_section.set_index("ticker")["final_score"].dropna()
            active_tickers = list(active_alpha.index)

            if not active_tickers:
                continue

            # Build historical return panel for covariance estimation
            hist_returns = returns_df[returns_df["date"] <= eval_date]
            pivoted_returns = hist_returns.pivot(index="date", columns="ticker", values="simple_return")
            pivoted_returns = pivoted_returns.reindex(columns=active_tickers).tail(cov_lookback)

            cov_df = cov_estimator.estimate(pivoted_returns)

            res: OptimizationResult = optimizer.optimize(
                expected_returns=active_alpha,
                covariance_df=cov_df,
                current_weights=current_weights,
                sector_map=sector_map,
                returns_panel=pivoted_returns,
                config=config,
            )

            current_weights = res.weights

            for ticker, w_val in res.weights.items():
                if w_val > 0:
                    all_weights.append({
                        "date": eval_date,
                        "ticker": ticker,
                        "weight": float(w_val),
                    })

            run_logs.append({
                "date": eval_date,
                "covariance_model": cov_name,
                "optimizer_type": opt_type,
                "status": res.status,
                "solver": res.solver_name,
                "objective_value": res.objective_value,
                "turnover": res.turnover,
            })

        weights_df = pd.DataFrame(all_weights)
        logs_df = pd.DataFrame(run_logs)
        return weights_df, logs_df
