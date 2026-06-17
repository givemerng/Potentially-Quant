from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd

from src.config import AppConfig
from src.data.db import create_schema, get_engine, macro_table, prices_table, rebuild_schema, upsert_rows
from src.data.downloader import YahooFinanceDownloader
from src.data.fred_client import FredDownloader


@dataclass
class PipelineSummary:
    ticker_count: int
    start_date: str
    end_date: str
    price_rows_upserted: int
    macro_rows_upserted: int
    failed: bool = False

    @property
    def total_rows_upserted(self) -> int:
        return self.price_rows_upserted + self.macro_rows_upserted


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

    def run(self) -> PipelineSummary:
        if self.config.rebuild_on_run:
            self.logger.warning("Rebuild mode enabled; dropping and recreating schema")
            rebuild_schema(self.engine)
        else:
            create_schema(self.engine)
        price_rows_upserted = 0
        macro_rows_upserted = 0
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

        return PipelineSummary(
            ticker_count=len(self.config.stock_universe),
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            price_rows_upserted=price_rows_upserted,
            macro_rows_upserted=macro_rows_upserted,
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
