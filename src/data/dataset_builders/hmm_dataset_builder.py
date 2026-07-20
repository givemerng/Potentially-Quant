from __future__ import annotations

from typing import List, Optional
import pandas as pd

from src.services.factor_service import FactorService
from src.services.market_data_service import MarketDataService


class HMMDatasetBuilder:
    """Specialized dataset builder for Hidden Markov Model (HMM) Market Regime Detection."""

    def __init__(
        self,
        factor_service: Optional[FactorService] = None,
        market_data_service: Optional[MarketDataService] = None,
    ) -> None:
        self.factor_service = factor_service or FactorService()
        self.market_data_service = market_data_service or MarketDataService()

    def build_hmm_features(
        self,
        series_names: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Assembles macro time-series and market returns for HMM state detection.
        Loads data exclusively from services/repositories connected to Neon.
        """
        series = series_names or ["DGS10", "DGS2", "VIXCLS", "BAMLH0A0HYM2", "NAPM"]
        macro_df = self.factor_service.get_macro_series(series_names=series, start_date=start_date, end_date=end_date)
        if macro_df.empty:
            return pd.DataFrame()

        pivoted = macro_df.pivot(index="date", columns="series_name", values="value").sort_index().ffill().bfill()
        return pivoted
