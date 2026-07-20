from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Optional
import pandas as pd
from sqlalchemy import select

from src.data.db import ticker_metadata_table
from src.data.repositories.base import BaseRepository
from src.data.repositories.interfaces import IMetadataRepository
from src.domain.models import TickerMetadata


class MetadataRepository(BaseRepository, IMetadataRepository):
    """Repository for managing ticker reference metadata with LRU caching."""

    def get_ticker_metadata(self, tickers: Optional[List[str]] = None) -> pd.DataFrame:
        stmt = select(
            ticker_metadata_table.c.ticker,
            ticker_metadata_table.c.sector,
            ticker_metadata_table.c.industry,
            ticker_metadata_table.c.shares_outstanding,
        )

        if tickers:
            stmt = stmt.where(ticker_metadata_table.c.ticker.in_(tickers))

        return self.fetch_dataframe(stmt)

    def get_domain_metadata(self, tickers: Optional[List[str]] = None) -> List[TickerMetadata]:
        df = self.get_ticker_metadata(tickers=tickers)
        if df.empty:
            return []
        return [
            TickerMetadata(
                ticker=row["ticker"],
                sector=row.get("sector"),
                industry=row.get("industry"),
                shares_outstanding=row.get("shares_outstanding"),
            )
            for _, row in df.iterrows()
        ]

    def save_ticker_metadata(self, rows: List[Dict[str, Any]]) -> int:
        return self.bulk_upsert(ticker_metadata_table, rows)
