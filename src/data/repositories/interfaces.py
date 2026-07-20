from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import pandas as pd


class IPricesRepository(ABC):
    @abstractmethod
    def get_daily_prices(self, start_date: Optional[str] = None, end_date: Optional[str] = None, tickers: Optional[List[str]] = None) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_returns(self, freq: str = "monthly", start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_universe_membership(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        pass

    @abstractmethod
    def save_prices(self, rows: List[Dict[str, Any]]) -> int:
        pass

    @abstractmethod
    def save_returns(self, rows: List[Dict[str, Any]]) -> int:
        pass

    @abstractmethod
    def save_universe_membership(self, rows: List[Dict[str, Any]]) -> int:
        pass


class IMetadataRepository(ABC):
    @abstractmethod
    def get_ticker_metadata(self, tickers: Optional[List[str]] = None) -> pd.DataFrame:
        pass

    @abstractmethod
    def save_ticker_metadata(self, rows: List[Dict[str, Any]]) -> int:
        pass


class IFundamentalsRepository(ABC):
    @abstractmethod
    def get_fundamentals(self, start_date: Optional[str] = None, end_date: Optional[str] = None, tickers: Optional[List[str]] = None) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_earnings_calendar(self, tickers: Optional[List[str]] = None) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_insider_transactions(self, tickers: Optional[List[str]] = None) -> pd.DataFrame:
        pass

    @abstractmethod
    def save_fundamentals(self, rows: List[Dict[str, Any]]) -> int:
        pass


class IFactorsRepository(ABC):
    @abstractmethod
    def get_factor_scores(self, factor_name: Optional[str] = None, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_factor_metrics(self, factor_name: Optional[str] = None, stage: Optional[str] = None) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_fama_french(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_macro_series(self, series_names: Optional[List[str]] = None, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        pass

    @abstractmethod
    def save_factor_scores(self, rows: List[Dict[str, Any]]) -> int:
        pass


class IRegimesRepository(ABC):
    @abstractmethod
    def get_market_regimes(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_regime_factor_ic(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_regime_factor_weights(self, run_id: Optional[str] = None) -> pd.DataFrame:
        pass

    @abstractmethod
    def save_market_regimes(self, rows: List[Dict[str, Any]]) -> int:
        pass


class IPortfolioRepository(ABC):
    @abstractmethod
    def get_backtest_results(self, backtest_name: Optional[str] = None) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_composite_scores(self, method: Optional[str] = None, run_id: Optional[str] = None) -> pd.DataFrame:
        pass

    @abstractmethod
    def save_backtest_results(self, results_df: pd.DataFrame, weights_df: pd.DataFrame, trades_df: pd.DataFrame, metrics_df: pd.DataFrame, backtest_name: str) -> Dict[str, int]:
        pass
