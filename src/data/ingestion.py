from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd

from src.config import AppConfig
from src.data.db import (
    create_schema, get_engine, macro_table, prices_table, rebuild_schema, upsert_rows,
    ticker_metadata_table, fundamentals_table, fama_french_table,
    insider_transactions_table, earnings_calendar_table
)
from src.data.downloader import YahooFinanceDownloader
from src.data.fred_client import FredDownloader
from src.data.fama_french_client import FamaFrenchDownloader


@dataclass
class PipelineSummary:
    ticker_count: int
    start_date: str
    end_date: str
    price_rows_upserted: int
    macro_rows_upserted: int
    metadata_rows_upserted: int = 0
    fundamentals_rows_upserted: int = 0
    ff_rows_upserted: int = 0
    earnings_calendar_rows_upserted: int = 0
    insider_transactions_rows_upserted: int = 0
    failed: bool = False

    @property
    def total_rows_upserted(self) -> int:
        return (
            self.price_rows_upserted
            + self.macro_rows_upserted
            + self.metadata_rows_upserted
            + self.fundamentals_rows_upserted
            + self.ff_rows_upserted
            + self.earnings_calendar_rows_upserted
            + self.insider_transactions_rows_upserted
        )


class DataIngestionPipeline:
    def __init__(self, config: AppConfig, logger) -> None:
        self.config = config
        self.logger = logger
        self.engine = get_engine()
        self.yahoo = YahooFinanceDownloader(
            logger=logger,
            retries=config.download.retries,
            retry_delay_seconds=config.download.retry_delay_seconds,
            max_workers=config.download.max_workers,
        )
        self.fred = FredDownloader(
            logger=logger,
            retries=config.download.retries,
            retry_delay_seconds=config.download.retry_delay_seconds,
        )
        self.fama_french = FamaFrenchDownloader(logger=logger)

    def run(self) -> PipelineSummary:
        if self.config.rebuild_on_run:
            self.logger.warning("Rebuild mode enabled; dropping and recreating schema")
            rebuild_schema(self.engine)
        else:
            create_schema(self.engine)
        price_rows_upserted = 0
        macro_rows_upserted = 0
        metadata_rows_upserted = 0
        fundamentals_rows_upserted = 0
        ff_rows_upserted = 0
        earnings_calendar_rows_upserted = 0
        insider_transactions_rows_upserted = 0
        failed = False

        try:
            prices = self.yahoo.fetch_price_data(
                tickers=self.config.stock_universe,
                start=self.config.start_date,
                end=self.config.end_date,
                batch_size=self.config.download.batch_size,
            )
            price_rows_upserted = self._store_prices(prices)
        except Exception as exc:
            failed = True
            self.logger.exception("Price ingestion failed: %s", exc)

        try:
            macro = self.fred.fetch_series(
                series_list=self.config.fred_series,
                start=self.config.start_date,
                end=self.config.end_date,
            )
            macro_rows_upserted = self._store_macro(macro)
        except Exception as exc:
            failed = True
            self.logger.exception("Macro ingestion failed: %s", exc)

        try:
            metadata_df, fundamentals_df = self.yahoo.fetch_metadata_and_fundamentals(
                tickers=self.config.stock_universe
            )
            metadata_rows_upserted = self._store_metadata(metadata_df)
            fundamentals_rows_upserted = self._store_fundamentals(fundamentals_df)
        except Exception as exc:
            failed = True
            self.logger.exception("Metadata/Fundamental ingestion failed: %s", exc)

        try:
            earnings_df = self.yahoo.fetch_earnings_calendar(tickers=self.config.stock_universe)
            earnings_calendar_rows_upserted = self._store_earnings_calendar(earnings_df)
        except Exception as exc:
            self.logger.warning("Earnings calendar ingestion failed: %s. Continuing.", exc)

        try:
            insider_df = self.yahoo.fetch_insider_transactions(tickers=self.config.stock_universe)
            insider_transactions_rows_upserted = self._store_insider_transactions(insider_df)
        except Exception as exc:
            self.logger.warning("Insider transactions ingestion failed: %s. Continuing.", exc)

        try:
            ff_df = self.fama_french.download_daily_factors()
            ff_rows_upserted = self._store_fama_french(ff_df)
        except Exception as exc:
            self.logger.warning("Fama-French ingestion failed: %s. Continuing pipeline.", exc)

        return PipelineSummary(
            ticker_count=len(self.config.stock_universe),
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            price_rows_upserted=price_rows_upserted,
            macro_rows_upserted=macro_rows_upserted,
            metadata_rows_upserted=metadata_rows_upserted,
            fundamentals_rows_upserted=fundamentals_rows_upserted,
            ff_rows_upserted=ff_rows_upserted,
            earnings_calendar_rows_upserted=earnings_calendar_rows_upserted,
            insider_transactions_rows_upserted=insider_transactions_rows_upserted,
            failed=failed,
        )

    def _store_prices(self, prices: pd.DataFrame) -> int:
        if prices.empty:
            self.logger.warning("No price rows to store")
            return 0

        reset = prices.reset_index()
        reset["date"] = pd.to_datetime(reset["date"]).dt.date
        rows = reset.to_dict(orient="records")
        upserted = upsert_rows(self.engine, prices_table, rows)
        self.logger.info("Upserted %s price rows", upserted)
        return upserted

    def _store_macro(self, macro: pd.DataFrame) -> int:
        if macro.empty:
            self.logger.warning("No macro rows to store")
            return 0

        melted = macro.reset_index().melt(id_vars="date", var_name="series_name", value_name="value")
        melted = melted.dropna(subset=["value"])
        melted["date"] = pd.to_datetime(melted["date"]).dt.date
        rows = melted.to_dict(orient="records")
        upserted = upsert_rows(self.engine, macro_table, rows)
        self.logger.info("Upserted %s macro rows", upserted)
        return upserted

    def _store_metadata(self, df: pd.DataFrame) -> int:
        if df.empty:
            self.logger.warning("No metadata rows to store")
            return 0
        rows = df.to_dict(orient="records")
        upserted = upsert_rows(self.engine, ticker_metadata_table, rows)
        self.logger.info("Upserted %s metadata rows", upserted)
        return upserted

    def _store_fundamentals(self, df: pd.DataFrame) -> int:
        if df.empty:
            self.logger.warning("No fundamental rows to store")
            return 0
        rows = df.to_dict(orient="records")
        upserted = upsert_rows(self.engine, fundamentals_table, rows)
        self.logger.info("Upserted %s fundamental rows", upserted)
        return upserted

    def _store_earnings_calendar(self, df: pd.DataFrame) -> int:
        if df.empty:
            self.logger.warning("No earnings calendar rows to store")
            return 0
        rows = df.to_dict(orient="records")
        upserted = upsert_rows(self.engine, earnings_calendar_table, rows)
        self.logger.info("Upserted %s earnings calendar rows", upserted)
        return upserted

    def _store_insider_transactions(self, df: pd.DataFrame) -> int:
        if df.empty:
            self.logger.warning("No insider transaction rows to store")
            return 0
        rows = df.to_dict(orient="records")
        upserted = upsert_rows(self.engine, insider_transactions_table, rows)
        self.logger.info("Upserted %s insider transaction rows", upserted)
        return upserted

    def _store_fama_french(self, df: pd.DataFrame) -> int:
        if df.empty:
            self.logger.warning("No Fama-French rows to store")
            return 0
        reset = df.reset_index(drop=True)
        reset["date"] = pd.to_datetime(reset["date"]).dt.date
        rows = reset.to_dict(orient="records")
        upserted = upsert_rows(self.engine, fama_french_table, rows)
        self.logger.info("Upserted %s Fama-French rows", upserted)
        return upserted

