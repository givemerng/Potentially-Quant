from __future__ import annotations

from typing import Optional, Tuple
import pandas as pd

from src.config import DatasetConfig
from src.services.factor_service import FactorService
from src.services.market_data_service import MarketDataService


class MLDatasetBuilder:
    """Specialized dataset builder for Machine Learning Factor Combination models (XGBoost, Fama-MacBeth, IC-Weighting)."""

    def __init__(
        self,
        factor_service: Optional[FactorService] = None,
        market_data_service: Optional[MarketDataService] = None,
        config: Optional[DatasetConfig] = None,
    ) -> None:
        self.factor_service = factor_service or FactorService()
        self.market_data_service = market_data_service or MarketDataService()
        self.config = config or DatasetConfig()

    def build_ml_dataset(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Builds feature matrix X (date, ticker x factors) and forward target label y (date, ticker x return).
        Loads data exclusively from services/repositories connected to Neon.
        """
        factor_df = self.factor_service.get_factor_scores(start_date=start_date, end_date=end_date)
        returns_df = self.market_data_service.get_returns_matrix(freq=self.config.target_freq, start_date=start_date, end_date=end_date)

        if factor_df.empty or returns_df.empty:
            return pd.DataFrame(), pd.DataFrame()

        # Pivot factors wide: (date, ticker) -> factor columns
        feature_matrix = factor_df.pivot(index=["date", "ticker"], columns="factor_name", values="final_score")
        target_series = returns_df.reset_index().set_index(["date", "ticker"])["simple_return"]

        # Align indices
        aligned = feature_matrix.join(target_series, how="inner").dropna()
        X = aligned[feature_matrix.columns]
        y = aligned[["simple_return"]]

        return X, y
