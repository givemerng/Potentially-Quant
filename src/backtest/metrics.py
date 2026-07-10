from __future__ import annotations

import numpy as np
import pandas as pd


class PortfolioMetrics:
    """Computes various risk/return performance statistics for a backtest return series."""

    @staticmethod
    def compute_all(
        net_returns: pd.Series,
        gross_returns: pd.Series,
        turnover_series: pd.Series,
        frequency: str = "monthly",
        risk_free_rate: float = 0.0,
    ) -> dict[str, float]:
        """Compute comprehensive performance metrics.

        Args:
            net_returns: pd.Series of net returns over time.
            gross_returns: pd.Series of gross returns over time.
            turnover_series: pd.Series of rebalance turnovers.
            frequency: 'daily' or 'monthly'.
            risk_free_rate: Annualized risk-free rate.

        Returns:
            dict containing performance statistics.
        """
        if net_returns.empty:
            return {}

        ann_factor = 12 if frequency == "monthly" else 252
        rf_per_period = risk_free_rate / ann_factor

        # Cumulative Return
        cum_ret = float((1.0 + net_returns).prod() - 1.0)
        cum_ret_gross = float((1.0 + gross_returns).prod() - 1.0)

        # Years elapsed
        n_periods = len(net_returns)
        years = n_periods / ann_factor
        
        # CAGR
        if cum_ret > -1.0 and years > 0:
            cagr = float((1.0 + cum_ret) ** (1.0 / years) - 1.0)
        else:
            cagr = -1.0

        # Volatility
        vol = float(net_returns.std(ddof=1) * np.sqrt(ann_factor)) if len(net_returns) > 1 else 0.0

        # Sharpe Ratio
        excess_returns = net_returns - rf_per_period
        mean_excess = excess_returns.mean()
        std_returns = net_returns.std(ddof=1)
        if std_returns > 1e-12:
            sharpe = float(mean_excess / std_returns * np.sqrt(ann_factor))
        else:
            sharpe = 0.0

        # Sortino Ratio
        neg_rets = net_returns[net_returns < 0.0]
        # Calculate standard deviation of negative returns relative to 0
        if len(neg_rets) > 0:
            std_neg = float(np.sqrt(np.mean(neg_rets ** 2) * ann_factor))
        else:
            std_neg = 0.0
            
        if std_neg > 1e-12:
            sortino = float(net_returns.mean() / (std_neg / np.sqrt(ann_factor)) * np.sqrt(ann_factor))
        else:
            sortino = 0.0

        # Maximum Drawdown
        equity = (1.0 + net_returns).cumprod()
        running_max = equity.cummax()
        drawdowns = (equity - running_max) / running_max
        max_dd = float(drawdowns.min()) if not drawdowns.empty else 0.0

        # Calmar Ratio
        if abs(max_dd) > 1e-12:
            calmar = float(cagr / abs(max_dd))
        else:
            calmar = np.nan

        # Win Rate
        win_rate = float((net_returns > 0.0).sum() / n_periods) if n_periods > 0 else 0.0

        # Average and Annualized Turnover
        avg_turnover = float(turnover_series.mean()) if not turnover_series.empty else 0.0
        ann_turnover = avg_turnover * (12 if frequency == "monthly" else 252)

        return {
            "cumulative_return_net": cum_ret,
            "cumulative_return_gross": cum_ret_gross,
            "cagr_net": cagr,
            "annualized_volatility": vol,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_dd,
            "calmar_ratio": calmar,
            "win_rate": win_rate,
            "average_turnover": avg_turnover,
            "annualized_turnover": ann_turnover,
        }
