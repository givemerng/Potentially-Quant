from __future__ import annotations

from typing import List, Dict, Any, Optional
import pandas as pd

from src.data.repositories.regimes_repository import RegimesRepository


class RegimeService:
    """Service orchestrating HMM market regime probabilities and Bayesian factor IC updates."""

    def __init__(self, regimes_repo: Optional[RegimesRepository] = None) -> None:
        self.regimes_repo = regimes_repo or RegimesRepository()

    def get_market_regimes(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        return self.regimes_repo.get_market_regimes(start_date=start_date, end_date=end_date)

    def get_regime_factor_ic(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        return self.regimes_repo.get_regime_factor_ic(start_date=start_date, end_date=end_date)

    def save_market_regimes(self, rows: List[Dict[str, Any]]) -> int:
        return self.regimes_repo.save_market_regimes(rows)

    def save_regime_factor_ic(self, rows: List[Dict[str, Any]]) -> int:
        return self.regimes_repo.save_regime_factor_ic(rows)

    def save_regime_factor_weights(self, rows: List[Dict[str, Any]]) -> int:
        return self.regimes_repo.save_regime_factor_weights(rows)
