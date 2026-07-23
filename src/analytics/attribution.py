"""Performance & Return Attribution Engine.

Decomposes portfolio returns into factor contributions vs idiosyncratic selection returns.
Accepts generic (weights, returns, factor_exposures) inputs.
"""

from __future__ import annotations
import logging
from typing import Dict, Optional
import numpy as np
import pandas as pd


class PerformanceAttributor:
    """Decomposes portfolio return into systematic factor contribution vs. stock selection return."""

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def attribute_returns(
        self,
        weights: pd.Series,
        asset_returns: pd.Series,
        factor_exposures: pd.DataFrame,
    ) -> Dict[str, float]:
        """Compute factor return attribution for a single period.

        Parameters
        ----------
        weights : pd.Series
            Portfolio weights indexed by ticker.
        asset_returns : pd.Series
            Realized period returns indexed by ticker.
        factor_exposures : pd.DataFrame
            Ticker x Factor matrix of factor loadings.

        Returns
        -------
        Dict[str, float]
            Dictionary of factor return contributions and idiosyncratic return.
        """
        common_tickers = list(weights.index.intersection(asset_returns.index).intersection(factor_exposures.index))
        if not common_tickers:
            return {"total_return": 0.0, "systematic_return": 0.0, "idiosyncratic_return": 0.0}

        w = weights.loc[common_tickers].values
        r = asset_returns.loc[common_tickers].values
        B = factor_exposures.loc[common_tickers].fillna(0.0).values  # N x K

        total_return = float(w @ r)

        try:
            # Estimate factor returns F_t via cross-sectional WLS/OLS: r_t = B * F_t + eps_t
            F = np.linalg.pinv(B) @ r  # K-dim
            port_beta = w @ B  # K-dim portfolio factor exposures
            factor_returns_contrib = port_beta * F  # K-dim

            systematic_return = float(np.sum(factor_returns_contrib))
            idiosyncratic_return = total_return - systematic_return

            result = {
                "total_return": total_return,
                "systematic_return": systematic_return,
                "idiosyncratic_return": idiosyncratic_return,
            }
            for idx, col in enumerate(factor_exposures.columns):
                result[f"factor_{col}"] = float(factor_returns_contrib[idx])

            return result
        except Exception as exc:
            self.logger.warning("Attribution failed (%s); returning un-decomposed total return", exc)
            return {"total_return": total_return, "systematic_return": 0.0, "idiosyncratic_return": total_return}
