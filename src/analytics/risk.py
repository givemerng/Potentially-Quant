"""Portfolio Risk & Risk Attribution Analytics."""

from __future__ import annotations
import logging
from typing import Dict, Optional
import numpy as np
import pandas as pd


class PortfolioRiskAnalytics:
    """Computes Marginal Contribution to Risk (MCR) and Component Risk Contribution."""

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def compute_mcr(
        self,
        weights: pd.Series,
        covariance_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute Marginal Contribution to Risk (MCR) and Percentage Contribution to Risk (PCR).

        Parameters
        ----------
        weights : pd.Series
            Portfolio weights indexed by ticker.
        covariance_df : pd.DataFrame
            Ticker x Ticker covariance matrix.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ['weight', 'mcr', 'pcr', 'component_risk'].
        """
        tickers = list(weights.index)
        cov = covariance_df.reindex(index=tickers, columns=tickers).fillna(0.0).values
        w = weights.values

        port_var = float(w @ cov @ w)
        port_vol = float(np.sqrt(max(1e-8, port_var)))

        mcr = (cov @ w) / port_vol  # Marginal Contribution to Risk (N-dim)
        component_risk = w * mcr    # Component Risk (N-dim)
        pcr = component_risk / port_vol  # Percentage Contribution to Risk

        df = pd.DataFrame(
            {
                "weight": w,
                "mcr": mcr,
                "pcr": pcr,
                "component_risk": component_risk,
            },
            index=tickers,
        )
        return df
