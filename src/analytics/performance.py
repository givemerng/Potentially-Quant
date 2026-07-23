"""Portfolio Performance Metrics Calculator."""

from __future__ import annotations
import logging
from typing import Dict, Optional
import numpy as np
import pandas as pd


class PortfolioPerformanceMetrics:
    """Calculates standardized annualized performance & risk metrics."""

    def __init__(self, ann_factor: int = 252, logger: Optional[logging.Logger] = None) -> None:
        self.ann_factor = ann_factor
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def compute(self, returns_series: pd.Series) -> Dict[str, float]:
        """Compute performance metrics for a daily or monthly return series."""
        rets = returns_series.dropna()
        if rets.empty or len(rets) < 2:
            return {}

        ann_return = float(np.mean(rets) * self.ann_factor)
        ann_vol = float(np.std(rets, ddof=1) * np.sqrt(self.ann_factor))

        sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

        downside_rets = rets[rets < 0]
        downside_vol = float(np.std(downside_rets, ddof=1) * np.sqrt(self.ann_factor)) if len(downside_rets) > 1 else ann_vol
        sortino = ann_return / downside_vol if downside_vol > 0 else 0.0

        cum_rets = (1.0 + rets).cumprod()
        running_max = cum_rets.cummax()
        drawdown = (cum_rets - running_max) / running_max
        max_drawdown = float(np.abs(drawdown.min()))

        calmar = ann_return / max_drawdown if max_drawdown > 0 else 0.0
        win_rate = float(np.mean(rets > 0))

        return {
            "annualized_return": ann_return,
            "annualized_volatility": ann_vol,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_drawdown,
            "calmar_ratio": calmar,
            "win_rate": win_rate,
            "total_observations": len(rets),
        }
