"""Equal Risk Contribution (Risk Parity) Portfolio Optimizer."""

from __future__ import annotations
import logging
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

from src.portfolio.optimizers.base import BasePortfolioOptimizer, OptimizationResult


class RiskParityOptimizer(BasePortfolioOptimizer):
    """Equal Risk Contribution (ERC) Portfolio Optimizer using inverse-volatility heuristic."""

    def optimize(
        self,
        expected_returns: pd.Series,
        covariance_df: pd.DataFrame,
        current_weights: Optional[pd.Series] = None,
        sector_map: Optional[pd.Series] = None,
        returns_panel: Optional[pd.DataFrame] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> OptimizationResult:
        cfg = config or {}
        tickers = list(expected_returns.index)
        N = len(tickers)

        if N == 0:
            return OptimizationResult(weights=pd.Series(dtype=float), status="empty_universe", solver_name="NONE")

        cov_matrix = covariance_df.reindex(index=tickers, columns=tickers).fillna(0.0).values
        variances = np.diag(cov_matrix).copy()
        variances[variances <= 0] = 1e-6

        inv_vol = 1.0 / np.sqrt(variances)
        weights_arr = inv_vol / np.sum(inv_vol)

        max_pos = cfg.get("max_position_size", 0.05)
        weights_arr = np.minimum(weights_arr, max_pos)
        if np.sum(weights_arr) > 0:
            weights_arr = weights_arr / np.sum(weights_arr)

        weights = pd.Series(weights_arr, index=tickers, name="weight")

        w_prev = np.zeros(N)
        if current_weights is not None:
            w_prev = current_weights.reindex(tickers).fillna(0.0).values
        turnover = float(np.sum(np.abs(weights.values - w_prev)))

        return OptimizationResult(
            weights=weights,
            status="optimal",
            solver_name="INVERSE_VOLATILITY",
            turnover=turnover,
        )
