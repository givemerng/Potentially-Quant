from __future__ import annotations

from typing import List, Optional
import pandas as pd

from src.data.repositories.factors_repository import FactorsRepository
from src.data.repositories.fundamentals_repository import FundamentalsRepository


class FactorService:
    """Service orchestrating factor signals, evaluation metrics, Fama-French benchmarks, and fundamentals."""

    def __init__(
        self,
        factors_repo: Optional[FactorsRepository] = None,
        fundamentals_repo: Optional[FundamentalsRepository] = None,
    ) -> None:
        self.factors_repo = factors_repo or FactorsRepository()
        self.fundamentals_repo = fundamentals_repo or FundamentalsRepository()

    def get_factor_scores(
        self,
        factor_name: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        return self.factors_repo.get_factor_scores(factor_name=factor_name, start_date=start_date, end_date=end_date)

    def get_fama_french(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        return self.factors_repo.get_fama_french(start_date=start_date, end_date=end_date)

    def get_macro_series(
        self,
        series_names: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        return self.factors_repo.get_macro_series(series_names=series_names, start_date=start_date, end_date=end_date)

    def get_fundamentals(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        tickers: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        return self.fundamentals_repo.get_fundamentals(start_date=start_date, end_date=end_date, tickers=tickers)

    def get_earnings_calendar(self, tickers: Optional[List[str]] = None) -> pd.DataFrame:
        return self.fundamentals_repo.get_earnings_calendar(tickers=tickers)

    def get_insider_transactions(self, tickers: Optional[List[str]] = None) -> pd.DataFrame:
        return self.fundamentals_repo.get_insider_transactions(tickers=tickers)
