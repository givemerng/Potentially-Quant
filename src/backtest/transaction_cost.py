from __future__ import annotations

import abc
import pandas as pd


class BaseTransactionCostModel(abc.ABC):
    """Abstract base class for transaction cost models."""

    @abc.abstractmethod
    def calculate_cost(self, trades: pd.Series, turnover: float) -> float:
        """Calculate the total transaction costs for a set of trades.

        Args:
            trades: pd.Series of weight changes (target_weight - drift_weight) by ticker.
            turnover: Total absolute sum of weight changes.

        Returns:
            cost: Total transaction cost as a fraction of portfolio value.
        """
        pass


class LinearTransactionCostModel(BaseTransactionCostModel):
    """Linear transaction cost model utilizing fixed commissions and bid-ask spreads.

    Cost = Turnover * (commission_bps + 0.5 * bid_ask_spread_bps)
    """

    def __init__(self, commission_bps: float = 5.0, bid_ask_spread_bps: float = 10.0) -> None:
        self.commission_bps = commission_bps
        self.bid_ask_spread_bps = bid_ask_spread_bps

    def calculate_cost(self, trades: pd.Series, turnover: float) -> float:
        total_bps = self.commission_bps + 0.5 * self.bid_ask_spread_bps
        return turnover * (total_bps / 10000.0)
