from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable, List

import pandas as pd
import yfinance as yf
from tqdm import tqdm


class YahooFinanceDownloader:
    def __init__(self, logger, retries: int = 3, retry_delay_seconds: int = 2, max_workers: int = 4) -> None:
        self.logger = logger
        self.retries = retries
        self.retry_delay_seconds = retry_delay_seconds
        self.max_workers = max_workers

    def fetch_price_data(self, tickers: List[str], start: str, end: str, batch_size: int = 25) -> pd.DataFrame:
        if not tickers:
            return pd.DataFrame()

        batches = list(self._chunked(tickers, batch_size))
        frames: list[pd.DataFrame] = []

        self.logger.info("Starting Yahoo Finance download for %s tickers across %s batches", len(tickers), len(batches))
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(batches))) as executor:
            future_map = {
                executor.submit(self._download_batch, batch, start, end): batch for batch in batches
            }
            for future in tqdm(as_completed(future_map), total=len(future_map), desc="Yahoo batches"):
                batch = future_map[future]
                try:
                    frame = future.result()
                    if not frame.empty:
                        frames.append(frame)
                    self.logger.info("Completed batch with %s tickers", len(batch))
                except Exception as exc:
                    self.logger.exception("Yahoo batch failed for tickers=%s: %s", batch, exc)

        if not frames:
            self.logger.warning("No price data was downloaded")
            return pd.DataFrame()

        combined = pd.concat(frames).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
        combined = combined.dropna(subset=["close", "adj_close"], how="all")
        return combined

    def _download_batch(self, tickers: List[str], start: str, end: str) -> pd.DataFrame:
        last_exception: Exception | None = None

        for attempt in range(1, self.retries + 1):
            try:
                self.logger.info("Downloading batch attempt %s/%s for %s tickers", attempt, self.retries, len(tickers))
                raw = yf.download(
                    tickers=tickers,
                    start=start,
                    end=end,
                    auto_adjust=False,
                    progress=False,
                    group_by="ticker",
                    threads=False,
                )
                return self._normalize_download(raw, tickers)
            except Exception as exc:
                last_exception = exc
                self.logger.warning("Yahoo download attempt %s failed: %s", attempt, exc)
                time.sleep(self.retry_delay_seconds * attempt)

        raise RuntimeError(f"Failed to download Yahoo batch after {self.retries} attempts") from last_exception

    def _normalize_download(self, raw: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
        if raw.empty:
            return pd.DataFrame()

        frames: list[pd.DataFrame] = []
        for ticker in tickers:
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    ticker_frame = raw[ticker].copy()
                else:
                    ticker_frame = raw.copy()

                ticker_frame = ticker_frame.rename(
                    columns={
                        "Open": "open",
                        "High": "high",
                        "Low": "low",
                        "Close": "close",
                        "Adj Close": "adj_close",
                        "Volume": "volume",
                    }
                )
                for column in ["open", "high", "low", "close", "adj_close", "volume"]:
                    if column not in ticker_frame.columns:
                        ticker_frame[column] = pd.NA

                ticker_frame = ticker_frame[["open", "high", "low", "close", "adj_close", "volume"]]
                ticker_frame["ticker"] = ticker
                ticker_frame.index = pd.to_datetime(ticker_frame.index).tz_localize(None)
                ticker_frame.index.name = "date"
                ticker_frame["volume"] = ticker_frame["volume"].fillna(0).astype("int64")
                frames.append(ticker_frame.reset_index())
            except KeyError:
                self.logger.warning("Ticker %s missing from Yahoo response", ticker)

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        combined = combined.dropna(subset=["open", "high", "low", "close", "adj_close"], how="all")
        combined["ticker"] = combined["ticker"].astype(str)
        combined = combined.set_index(["date", "ticker"]).sort_index()
        return combined

    @staticmethod
    def _chunked(values: List[str], batch_size: int) -> Iterable[List[str]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        total_batches = math.ceil(len(values) / batch_size)
        for batch_number in range(total_batches):
            start = batch_number * batch_size
            yield values[start : start + batch_size]
