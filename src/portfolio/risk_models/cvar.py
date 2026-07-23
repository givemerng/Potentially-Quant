"""Conditional Value-at-Risk (CVaR) Calculator & Scenario Generator."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd


class CVaRCalculator:
    """Calculates Parametric and Historical CVaR (Expected Shortfall) and prepares scenario matrices."""

    def __init__(self, alpha: float = 0.95, logger: Optional[logging.Logger] = None) -> None:
        if not (0.5 <= alpha < 1.0):
            raise ValueError(f"CVaR alpha confidence must be in [0.5, 1.0), got {alpha}")
        self.alpha = alpha
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def compute_historical_cvar(self, returns_series: pd.Series) -> float:
        """Compute historical CVaR at alpha confidence level for a portfolio return series."""
        clean_rets = returns_series.dropna()
        if clean_rets.empty:
            return 0.0

        cutoff_idx = int((1.0 - self.alpha) * len(clean_rets))
        if cutoff_idx == 0:
            cutoff_idx = 1

        sorted_rets = np.sort(clean_rets.values)
        var_alpha = sorted_rets[cutoff_idx - 1]
        cvar_alpha = -np.mean(sorted_rets[:cutoff_idx])
        return float(max(0.0, cvar_alpha))

    def build_scenario_matrix(self, returns_df: pd.DataFrame) -> Tuple[np.ndarray, list[str]]:
        """Extract scenario return matrix (T x N) for CVXPY CVaR optimization.

        Parameters
        ----------
        returns_df : pd.DataFrame
            Historical return panel indexed by date with ticker columns.

        Returns
        -------
        Tuple[np.ndarray, list[str]]
            (T x N scenario array, list of tickers)
        """
        clean_returns = returns_df.dropna(how="all").fillna(0.0)
        tickers = list(clean_returns.columns)
        scenario_matrix = clean_returns.values  # T x N
        return scenario_matrix, tickers
