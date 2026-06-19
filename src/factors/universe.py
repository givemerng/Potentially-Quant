"""Point-in-time universe filter for the Regime Alpha Engine.

Applies three exclusion rules on a strictly point-in-time basis:
  1. Market-cap proxy below $1B (dollar-volume proxy, no look-ahead).
  2. Daily dollar volume below universe median (cross-sectional filter).
  3. Within N months of first available price date (IPO exclusion).

Results are stored in the ``universe_membership`` table.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy.engine import Engine

from src.config import UniverseFilterConfig
from src.data.db import universe_membership_table, upsert_rows


@dataclass
class UniverseSummary:
    total_ticker_dates: int
    in_universe_ticker_dates: int
    rows_stored: int

    @property
    def inclusion_rate(self) -> float:
        if self.total_ticker_dates == 0:
            return 0.0
        return self.in_universe_ticker_dates / self.total_ticker_dates


class UniverseFilter:
    """Point-in-time universe membership filter.

    All filter criteria use only information available on or before the
    filter date — no look-ahead bias is introduced.

    Market-cap note:
        Historical shares-outstanding data requires Compustat or similar.
        As a point-in-time safe proxy we use:
            market_cap_proxy = price × trailing_avg_daily_volume
        This approximates dollar-market-cap without needing shares-outstanding.
    """

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Core filter
    # ------------------------------------------------------------------

    def compute_membership(
        self,
        prices: pd.DataFrame,
        config: UniverseFilterConfig,
    ) -> pd.DataFrame:
        """Compute point-in-time universe membership for every (date, ticker).

        Args:
            prices: DataFrame with MultiIndex (date, ticker) and columns
                    including ``adj_close`` and ``volume``.
            config: Universe filter configuration thresholds.

        Returns:
            DataFrame with columns:
            [ticker, date, in_universe, market_cap_proxy,
             avg_dollar_vol, days_since_first_price]
        """
        if prices.empty:
            self.logger.warning("Empty prices DataFrame — returning empty membership.")
            return pd.DataFrame(
                columns=["ticker", "date", "in_universe", "market_cap_proxy",
                         "avg_dollar_vol", "days_since_first_price"]
            )

        prices = prices.copy().sort_index()

        # Pivot to wide format for vectorised ops
        close_wide = prices["adj_close"].unstack("ticker").sort_index()
        vol_wide = prices["volume"].unstack("ticker").sort_index()

        # Dollar volume = price × volume per day
        dollar_vol_wide = close_wide * vol_wide

        # --- Feature 1: Trailing avg daily dollar volume (lookback window) ---
        lookback = config.dollar_vol_lookback_days
        avg_dollar_vol = dollar_vol_wide.rolling(window=lookback, min_periods=max(1, lookback // 2)).mean()

        # --- Feature 2: Market-cap proxy = price × trailing avg volume ---
        market_cap_proxy = close_wide * avg_dollar_vol.div(close_wide).fillna(0)
        # Simpler point-in-time safe formula: price × trailing_avg_vol * price
        # i.e. market_cap_proxy = close^2 × trailing_avg_vol / close = close × trailing_avg_vol
        # Actually: market_cap_proxy (in $) = avg_dollar_vol (already in $, per day)
        # We use avg_dollar_vol itself as the proxy (larger = larger cap)
        market_cap_proxy_bn = avg_dollar_vol / 1e9  # convert to $B

        # --- Feature 3: Days since first available price (IPO exclusion) ---
        first_date_per_ticker = close_wide.apply(lambda col: col.first_valid_index())
        days_since_first: pd.DataFrame = pd.DataFrame(
            index=close_wide.index, columns=close_wide.columns, dtype=float
        )
        for ticker in close_wide.columns:
            first = first_date_per_ticker[ticker]
            if first is None:
                days_since_first[ticker] = np.nan
            else:
                days_since_first[ticker] = (
                    pd.to_datetime(close_wide.index) - pd.Timestamp(first)
                ).days

        # --- Apply filters (all point-in-time) ---
        ipo_months_days = config.ipo_exclusion_months * 30.44  # approximate

        # Cap filter: proxy in $B must exceed threshold
        cap_ok = market_cap_proxy_bn >= config.min_market_cap_bn

        # Dollar volume percentile filter (cross-sectional per date)
        dollar_vol_threshold = avg_dollar_vol.quantile(
            config.min_dollar_vol_percentile / 100, axis=1
        )
        dvol_ok = avg_dollar_vol.ge(dollar_vol_threshold, axis=0)

        # IPO filter
        ipo_ok = days_since_first >= ipo_months_days

        # A stock must pass all three filters AND have non-NaN close
        has_price = close_wide.notna()
        in_universe = cap_ok & dvol_ok & ipo_ok & has_price

        # --- Stack back to long format ---
        records = []
        for ticker in close_wide.columns:
            ticker_dates = close_wide.index[has_price[ticker]]
            for date in ticker_dates:
                records.append(
                    {
                        "ticker": ticker,
                        "date": date.date(),
                        "in_universe": bool(in_universe.loc[date, ticker]),
                        "market_cap_proxy": float(market_cap_proxy_bn.loc[date, ticker])
                        if not pd.isna(market_cap_proxy_bn.loc[date, ticker])
                        else None,
                        "avg_dollar_vol": float(avg_dollar_vol.loc[date, ticker])
                        if not pd.isna(avg_dollar_vol.loc[date, ticker])
                        else None,
                        "days_since_first_price": float(days_since_first.loc[date, ticker])
                        if not pd.isna(days_since_first.loc[date, ticker])
                        else None,
                    }
                )

        result = pd.DataFrame(records)
        in_count = int(result["in_universe"].sum()) if not result.empty else 0
        self.logger.info(
            "Universe filter: %d / %d (date, ticker) pairs in-universe",
            in_count,
            len(result),
        )
        return result

    # ------------------------------------------------------------------
    # Vectorised version (faster, used internally)
    # ------------------------------------------------------------------

    def compute_membership_fast(
        self,
        prices: pd.DataFrame,
        config: UniverseFilterConfig,
    ) -> pd.DataFrame:
        """Faster vectorised version returning a long DataFrame.

        Uses stacking instead of a Python loop — much faster for large universes.
        Produces the same schema as ``compute_membership``.
        """
        if prices.empty:
            return pd.DataFrame(
                columns=["ticker", "date", "in_universe", "market_cap_proxy",
                         "avg_dollar_vol", "days_since_first_price"]
            )

        prices = prices.copy().sort_index()
        close_wide = prices["adj_close"].unstack("ticker").sort_index()
        vol_wide = prices["volume"].unstack("ticker").sort_index()
        dollar_vol_wide = close_wide * vol_wide

        lookback = config.dollar_vol_lookback_days
        avg_dollar_vol = dollar_vol_wide.rolling(window=lookback, min_periods=max(1, lookback // 2)).mean()
        market_cap_proxy_bn = avg_dollar_vol / 1e9

        first_date_per_ticker = close_wide.apply(lambda col: col.first_valid_index())
        date_index = pd.to_datetime(close_wide.index)
        days_since_first = pd.DataFrame(index=close_wide.index, columns=close_wide.columns, dtype=float)
        for ticker in close_wide.columns:
            first = first_date_per_ticker[ticker]
            days_since_first[ticker] = (date_index - pd.Timestamp(first)).days if first is not None else np.nan

        ipo_months_days = config.ipo_exclusion_months * 30.44
        cap_ok = market_cap_proxy_bn >= config.min_market_cap_bn
        dollar_vol_threshold = avg_dollar_vol.quantile(config.min_dollar_vol_percentile / 100, axis=1)
        dvol_ok = avg_dollar_vol.ge(dollar_vol_threshold, axis=0)
        ipo_ok = days_since_first >= ipo_months_days
        has_price = close_wide.notna()
        in_universe = (cap_ok & dvol_ok & ipo_ok & has_price).astype(float)

        # Stack all features together
        stacked = pd.DataFrame(
            {
                "in_universe": in_universe.stack(future_stack=True),
                "market_cap_proxy": market_cap_proxy_bn.stack(future_stack=True),
                "avg_dollar_vol": avg_dollar_vol.stack(future_stack=True),
                "days_since_first_price": days_since_first.stack(future_stack=True),
            }
        )
        stacked = stacked[has_price.stack(future_stack=True)]  # only rows with a price
        stacked = stacked.reset_index()
        stacked.columns = ["date", "ticker", "in_universe", "market_cap_proxy",
                           "avg_dollar_vol", "days_since_first_price"]
        stacked["in_universe"] = stacked["in_universe"].astype(bool)
        stacked["date"] = pd.to_datetime(stacked["date"]).dt.date

        in_count = int(stacked["in_universe"].sum())
        self.logger.info(
            "Universe filter (fast): %d / %d (date, ticker) pairs in-universe",
            in_count,
            len(stacked),
        )
        return stacked

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def store_membership(self, engine: Engine, membership: pd.DataFrame) -> int:
        """Upsert membership records into the ``universe_membership`` table.

        Args:
            engine: SQLAlchemy engine.
            membership: Output of ``compute_membership_fast``.

        Returns:
            Number of rows upserted.
        """
        if membership.empty:
            self.logger.warning("No universe membership rows to store.")
            return 0

        rows = membership.to_dict(orient="records")
        upserted = upsert_rows(engine, universe_membership_table, rows)
        self.logger.info("Upserted %d universe membership rows", upserted)
        return upserted

    # ------------------------------------------------------------------
    # High-level runner
    # ------------------------------------------------------------------

    def run(
        self,
        engine: Engine,
        prices: pd.DataFrame,
        config: UniverseFilterConfig,
    ) -> UniverseSummary:
        """Compute universe membership and persist to DB.

        Args:
            engine: Active SQLAlchemy engine.
            prices: Raw price DataFrame from the ingestion pipeline.
            config: Universe filter configuration.

        Returns:
            ``UniverseSummary`` with counts and inclusion rate.
        """
        membership = self.compute_membership_fast(prices, config)
        rows_stored = self.store_membership(engine, membership)

        return UniverseSummary(
            total_ticker_dates=len(membership),
            in_universe_ticker_dates=int(membership["in_universe"].sum()) if not membership.empty else 0,
            rows_stored=rows_stored,
        )
