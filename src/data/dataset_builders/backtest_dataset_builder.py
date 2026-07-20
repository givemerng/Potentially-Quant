from __future__ import annotations

from typing import Optional, Tuple
import pandas as pd

from src.services.portfolio_service import PortfolioService
from src.services.market_data_service import MarketDataService


class BacktestDatasetBuilder:
    """Specialized dataset builder for Vectorized Backtesting."""

    def __init__(
        self,
        portfolio_service: Optional[PortfolioService] = None,
        market_data_service: Optional[MarketDataService] = None,
    ) -> None:
        self.portfolio_service = portfolio_service or PortfolioService()
        self.market_data_service = market_data_service or MarketDataService()

    def build_backtest_input(
        self,
        composite_method: str = "ic_weighted",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        scores = self.portfolio_service.get_composite_scores(method=composite_method)
        prices = self.market_data_service.get_price_history(start_date=start_date, end_date=end_date)
        return scores, prices
