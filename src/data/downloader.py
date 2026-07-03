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

    def fetch_metadata_and_fundamentals(self, tickers: List[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Fetch ticker metadata and quarterly fundamentals in parallel.

        Returns:
            metadata_df: columns [ticker, sector, industry, shares_outstanding]
            fundamentals_df: columns [date, ticker, book_value, gross_profit, total_assets, eps]
        """
        self.logger.info("Starting metadata and fundamental fetch for %d tickers", len(tickers))
        meta_records = []
        fund_frames = []

        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(tickers))) as executor:
            future_map = {
                executor.submit(self._fetch_single_ticker_info_and_fundamentals, ticker): ticker
                for ticker in tickers
            }
            for future in tqdm(as_completed(future_map), total=len(future_map), desc="Metadata & Fundamentals"):
                ticker = future_map[future]
                try:
                    meta, fund = future.result()
                    if meta:
                        meta_records.append(meta)
                    if fund is not None and not fund.empty:
                        fund_frames.append(fund)
                except Exception as exc:
                    self.logger.error("Failed to fetch info and fundamentals for %s: %s", ticker, exc)

        metadata_df = pd.DataFrame(meta_records)
        if metadata_df.empty:
            metadata_df = pd.DataFrame(columns=["ticker", "sector", "industry", "shares_outstanding"])

        fundamentals_df = pd.concat(fund_frames, ignore_index=True) if fund_frames else pd.DataFrame(
            columns=["date", "ticker", "book_value", "gross_profit", "total_assets", "eps"]
        )

        return metadata_df, fundamentals_df

    def _fetch_single_ticker_info_and_fundamentals(self, ticker: str) -> tuple[dict | None, pd.DataFrame | None]:
        self.logger.info("Fetching yfinance info/fundamentals for %s", ticker)
        yf_ticker = yf.Ticker(ticker)

        # Get metadata
        meta = {"ticker": ticker, "sector": None, "industry": None, "shares_outstanding": None}
        try:
            info = yf_ticker.info
            if info:
                meta["sector"] = info.get("sector")
                meta["industry"] = info.get("industry")
                meta["shares_outstanding"] = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding")
        except Exception as exc:
            self.logger.warning("Error fetching info for %s: %s. Using default metadata.", ticker, exc)

        # Get fundamentals
        fund_df = None
        try:
            bs = yf_ticker.quarterly_balance_sheet
            fin = yf_ticker.quarterly_financials
            cf = yf_ticker.quarterly_cashflow
 
            if bs.empty:
                bs = yf_ticker.balance_sheet
            if fin.empty:
                fin = yf_ticker.financials
            if cf.empty:
                cf = yf_ticker.cashflow
 
            if not bs.empty or not fin.empty or not cf.empty:
                # Combine rows
                combined = pd.concat([bs, fin, cf], axis=0)
                # Keep index strings clean
                combined.index = [str(idx).strip() for idx in combined.index]
 
                # Helper to extract metrics case-insensitively
                def extract_metric(candidates: list[str]) -> pd.Series | None:
                    idx_lower = [idx.lower() for idx in combined.index]
                    for cand in candidates:
                        cand_l = cand.lower().strip()
                        if cand_l in idx_lower:
                            pos = idx_lower.index(cand_l)
                            return combined.iloc[pos]
                    return None
 
                book_value = extract_metric(["stockholders equity", "total stockholders equity", "common stock equity", "stockholders_equity"])
                gross_profit = extract_metric(["gross profit", "grossprofit", "gross_profit"])
                total_assets = extract_metric(["total assets", "totalassets", "total_assets"])
                eps = extract_metric(["diluted eps", "basic eps", "diluted_eps", "basic_eps", "eps", "net income"])
                ebitda = extract_metric(["ebitda", "normalized ebitda", "normalized_ebitda"])
                total_debt = extract_metric(["total debt", "total_debt"])
                operating_cash_flow = extract_metric(["operating cash flow", "operating_cash_flow", "cash flow from continuing operating activities"])
                capital_expenditures = extract_metric(["capital expenditure", "capital_expenditures", "capital_expenditure", "purchase of ppe", "purchase of property plant and equipment"])
                net_income = extract_metric(["net income", "net_income", "net income from continuing operations"])
 
                # Align to columns (reporting dates)
                temp_df = pd.DataFrame(index=combined.columns)
                temp_df["book_value"] = book_value if book_value is not None else pd.NA
                temp_df["gross_profit"] = gross_profit if gross_profit is not None else pd.NA
                temp_df["total_assets"] = total_assets if total_assets is not None else pd.NA
                temp_df["eps"] = eps if eps is not None else pd.NA
                temp_df["ebitda"] = ebitda if ebitda is not None else pd.NA
                temp_df["total_debt"] = total_debt if total_debt is not None else pd.NA
                temp_df["operating_cash_flow"] = operating_cash_flow if operating_cash_flow is not None else pd.NA
                temp_df["capital_expenditures"] = capital_expenditures.abs() if capital_expenditures is not None else pd.NA
                temp_df["net_income"] = net_income if net_income is not None else pd.NA
 
                temp_df = temp_df.rename_axis("date").reset_index()
                temp_df["date"] = pd.to_datetime(temp_df["date"]).dt.date
                temp_df["ticker"] = ticker
                fund_df = temp_df
        except Exception as exc:
            self.logger.warning("Error fetching quarterly financial statement data for %s: %s", ticker, exc)
 
        return meta, fund_df

    def fetch_earnings_calendar(self, tickers: List[str]) -> pd.DataFrame:
        self.logger.info("Starting earnings calendar fetch for %d tickers", len(tickers))
        records = []
        for ticker in tickers:
            try:
                yf_ticker = yf.Ticker(ticker)
                ed = yf_ticker.earnings_dates
                if ed is not None and not ed.empty:
                    for dt, row in ed.iterrows():
                        clean_dt = pd.Timestamp(dt).tz_localize(None).date()
                        records.append({
                            "ticker": ticker,
                            "date": clean_dt,
                            "eps_estimate": float(row["EPS Estimate"]) if not pd.isna(row["EPS Estimate"]) else None,
                            "reported_eps": float(row["Reported EPS"]) if not pd.isna(row["Reported EPS"]) else None,
                            "surprise_pct": float(row["Surprise(%)"]) if not pd.isna(row["Surprise(%)"]) else None,
                        })
            except Exception as exc:
                self.logger.warning("Failed to fetch earnings calendar for %s: %s", ticker, exc)
        return pd.DataFrame(records)

    def fetch_insider_transactions(self, tickers: List[str]) -> pd.DataFrame:
        self.logger.info("Starting insider transactions fetch for %d tickers", len(tickers))
        records = []
        for ticker in tickers:
            try:
                yf_ticker = yf.Ticker(ticker)
                it = yf_ticker.insider_transactions
                if it is not None and not it.empty:
                    for _, row in it.iterrows():
                        dt = pd.to_datetime(row["Start Date"]).date()
                        records.append({
                            "ticker": ticker,
                            "date": dt,
                            "insider": str(row["Insider"]) if not pd.isna(row["Insider"]) else None,
                            "position": str(row["Position"]) if not pd.isna(row["Position"]) else None,
                            "shares": float(row["Shares"]) if not pd.isna(row["Shares"]) else None,
                            "value": float(row["Value"]) if not pd.isna(row["Value"]) else None,
                            "text": str(row["Text"]) if not pd.isna(row["Text"]) else None,
                        })
            except Exception as exc:
                self.logger.warning("Failed to fetch insider transactions for %s: %s", ticker, exc)
        return pd.DataFrame(records)

    @staticmethod
    def _chunked(values: List[str], batch_size: int) -> Iterable[List[str]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        total_batches = math.ceil(len(values) / batch_size)
        for batch_number in range(total_batches):
            start = batch_number * batch_size
            yield values[start : start + batch_size]

