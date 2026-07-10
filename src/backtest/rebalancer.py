from __future__ import annotations

import pandas as pd


class Rebalancer:
    """Manages the rebalancing schedule, compares portfolio weights, and calculates turnover."""

    def __init__(self, frequency: str = "monthly", logger=None) -> None:
        self.frequency = frequency
        self.logger = logger

    def get_rebalance_schedule(self, dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """Filter the index of dates to return those corresponding to the rebalance frequency."""
        if dates.empty:
            return dates

        if self.frequency == "daily":
            return dates.sort_values()
        elif self.frequency == "monthly":
            # Group by year and month and choose the last date of each month
            df = pd.DataFrame(index=dates)
            df["year"] = df.index.year
            df["month"] = df.index.month
            
            # Use a robust way to get month-end dates present in the index
            rebalance_dates = []
            for _, group in df.groupby(["year", "month"]):
                rebalance_dates.append(group.index[-1])
                
            return pd.DatetimeIndex(rebalance_dates).sort_values()
        else:
            raise ValueError(f"Unknown rebalance frequency: {self.frequency}")

    def calculate_turnover(
        self,
        prev_weights: pd.Series,
        target_weights: pd.Series,
        asset_returns: pd.Series,
    ) -> tuple[float, pd.Series]:
        """Compute the turnover and weight changes during rebalancing.

        Args:
            prev_weights: Weights at the start of the previous period (from t-1).
            target_weights: Target weights for the new period (at t).
            asset_returns: Returns of the assets during the period (from t-1 to t).

        Returns:
            turnover: Total absolute change in weights.
            weight_change: pd.Series of weight changes (trades).
        """
        # If no previous weights exist, initial turnover is just target weights absolute sum
        if prev_weights.empty or (prev_weights == 0.0).all():
            weight_change = target_weights.copy()
            turnover = float(target_weights.abs().sum())
            return turnover, weight_change

        # Align series
        all_tickers = prev_weights.index.union(target_weights.index).union(asset_returns.index)
        w_prev = prev_weights.reindex(all_tickers, fill_value=0.0)
        w_target = target_weights.reindex(all_tickers, fill_value=0.0)
        ret = asset_returns.reindex(all_tickers, fill_value=0.0)

        # Drift weight calculation: w_t- = w_prev * (1 + ret) / (1 + port_ret)
        port_ret = float((w_prev * ret).sum())
        
        if port_ret > -0.999:
            w_drift = w_prev * (1.0 + ret) / (1.0 + port_ret)
        else:
            w_drift = w_prev

        weight_change = w_target - w_drift
        turnover = float(weight_change.abs().sum())

        return turnover, weight_change
