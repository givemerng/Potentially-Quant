from __future__ import annotations

import os
import time
from typing import List

import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred


class FredDownloader:
    def __init__(self, logger, retries: int = 3, retry_delay_seconds: int = 2) -> None:
        self.logger = logger
        self.retries = retries
        self.retry_delay_seconds = retry_delay_seconds
        load_dotenv()
        api_key = os.getenv("FRED_API_KEY")
        self.client = Fred(api_key=api_key) if api_key else None

    def fetch_series(self, series_list: List[str], start: str, end: str) -> pd.DataFrame:
        if not self.client:
            self.logger.warning("FRED API key not configured; skipping macro download")
            return pd.DataFrame()

        series_frames: list[pd.Series] = []
        for series_name in series_list:
            series = self._fetch_single_series(series_name, start, end)
            if series is not None and not series.empty:
                series_frames.append(series.rename(series_name))

        if not series_frames:
            return pd.DataFrame()

        frame = pd.concat(series_frames, axis=1).sort_index()
        frame.index = pd.to_datetime(frame.index).tz_localize(None)
        frame = frame.loc[(frame.index >= pd.Timestamp(start)) & (frame.index <= pd.Timestamp(end))]
        frame = frame.ffill()
        frame.index.name = "date"
        return frame

    def _fetch_single_series(self, series_name: str, start: str, end: str) -> pd.Series | None:
        last_exception: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                self.logger.info("Fetching FRED series %s (%s/%s)", series_name, attempt, self.retries)
                data = self.client.get_series(series_name, observation_start=start, observation_end=end)
                series = pd.Series(data, name=series_name).sort_index()
                return pd.to_numeric(series, errors="coerce")
            except Exception as exc:
                last_exception = exc
                self.logger.warning("FRED fetch failed for %s on attempt %s: %s", series_name, attempt, exc)
                time.sleep(self.retry_delay_seconds * attempt)

        self.logger.error("Skipping FRED series %s after repeated failures: %s", series_name, last_exception)
        return None
