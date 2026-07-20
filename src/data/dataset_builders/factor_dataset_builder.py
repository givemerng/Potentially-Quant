from __future__ import annotations

from typing import Optional, Tuple
import pandas as pd

from src.services.factor_service import FactorService
from src.services.market_data_service import MarketDataService


class FactorDatasetBuilder:
    """Specialized dataset builder for Factor Evaluation, IC analysis, and neutralizations."""

    def __init__(
        self,
        factor_service: Optional[FactorService] = None,
        market_data_service: Optional[MarketDataService] = None,
    ) -> None:
        self.factor_service = factor_service or FactorService()
        self.market_data_service = market_data_service or MarketDataService()

    def build_eval_dataset(
        self,
        factor_name: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        scores = self.factor_service.get_factor_scores(factor_name=factor_name, start_date=start_date, end_date=end_date)
        returns = self.market_data_service.get_returns_matrix(freq="monthly", start_date=start_date, end_date=end_date)
        return scores, returns
