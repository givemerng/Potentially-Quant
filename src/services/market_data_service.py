from __future__ import annotations

from typing import List, Optional
import pandas as pd

from src.data.repositories.prices_repository import PricesRepository
from src.data.repositories.metadata_repository import MetadataRepository
from src.domain.models import TickerMetadata


class MarketDataService:
    """Service orchestrating market prices, ticker metadata, and universe filtering."""

    def __init__(
        self,
        prices_repo: Optional[PricesRepository] = None,
        metadata_repo: Optional[MetadataRepository] = None,
    ) -> None:
        self.prices_repo = prices_repo or PricesRepository()
        self.metadata_repo = metadata_repo or MetadataRepository()

    def get_price_history(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        tickers: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        return self.prices_repo.get_daily_prices(start_date=start_date, end_date=end_date, tickers=tickers)

    def get_returns_matrix(self, freq: str = "monthly", start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        return self.prices_repo.get_returns(freq=freq, start_date=start_date, end_date=end_date)

    def get_universe_membership(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        return self.prices_repo.get_universe_membership(start_date=start_date, end_date=end_date)

    def get_ticker_metadata(self, tickers: Optional[List[str]] = None) -> pd.DataFrame:
        return self.metadata_repo.get_ticker_metadata(tickers=tickers)

    def get_domain_metadata(self, tickers: Optional[List[str]] = None) -> List[TickerMetadata]:
        return self.metadata_repo.get_domain_metadata(tickers=tickers)
