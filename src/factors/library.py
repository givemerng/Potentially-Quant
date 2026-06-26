from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy.engine import Engine

from src.factors.base import BaseAlpha


class PriceMomentum(BaseAlpha):
    """Price Momentum factor: Cumulative return over the past N months, excluding the last month.

    Often represented as 12-1 month momentum.
    """

    def compute(
        self,
        engine: Engine,
        prices_df: pd.DataFrame,
        metadata_df: pd.DataFrame,
        fundamentals_df: pd.DataFrame,
        date: pd.Timestamp,
        config: dict,
    ) -> pd.Series:
        if prices_df.empty:
            return pd.Series(dtype=float)

        # 1. Pivot prices to wide format (dates x tickers)
        close_wide = prices_df["adj_close"].unstack("ticker").sort_index()
        
        # 2. Resample to month-end frequency
        close_m = close_wide.ffill().resample("ME").last()

        # 3. Compute momentum: Price(t-1) / Price(t-lookback) - 1.0
        lookback = config.get("momentum_lookback_months", 12)
        if lookback < 2:
            self.logger.warning("momentum_lookback_months must be >= 2. Using 12.")
            lookback = 12

        mom = close_m.shift(1) / close_m.shift(lookback) - 1.0

        if date not in mom.index:
            self.logger.warning("Date %s not found in resampled Momentum series index", date)
            return pd.Series(np.nan, index=close_wide.columns)

        return mom.loc[date]


class ThreeMonthMomentum(BaseAlpha):
    """Three-Month Momentum factor: Cumulative return over the past 3 months, excluding the last month."""

    def compute(
        self,
        engine: Engine,
        prices_df: pd.DataFrame,
        metadata_df: pd.DataFrame,
        fundamentals_df: pd.DataFrame,
        date: pd.Timestamp,
        config: dict,
    ) -> pd.Series:
        if prices_df.empty:
            return pd.Series(dtype=float)

        close_wide = prices_df["adj_close"].unstack("ticker").sort_index()
        close_m = close_wide.ffill().resample("ME").last()

        lookback = config.get("short_momentum_lookback_months", 3)
        if lookback < 2:
            self.logger.warning("short_momentum_lookback_months must be >= 2. Using 3.")
            lookback = 3

        mom = close_m.shift(1) / close_m.shift(lookback) - 1.0

        if date not in mom.index:
            return pd.Series(np.nan, index=close_wide.columns)

        return mom.loc[date]


class BookToPrice(BaseAlpha):
    """Book-to-Price (B/P) factor: Book value of equity / Market Capitalization."""

    def compute(
        self,
        engine: Engine,
        prices_df: pd.DataFrame,
        metadata_df: pd.DataFrame,
        fundamentals_df: pd.DataFrame,
        date: pd.Timestamp,
        config: dict,
    ) -> pd.Series:
        if prices_df.empty:
            return pd.Series(dtype=float)

        # Pivot stock prices to month-end
        close_wide = prices_df["adj_close"].unstack("ticker").sort_index()
        close_m = close_wide.ffill().resample("ME").last()

        if fundamentals_df.empty or metadata_df.empty:
            self.logger.warning("No fundamental or metadata data available for BookToPrice.")
            return pd.Series(np.nan, index=close_wide.columns)

        # Extract Book Value and pivot to wide format
        book_df = fundamentals_df.dropna(subset=["book_value"])
        if book_df.empty:
            return pd.Series(np.nan, index=close_wide.columns)

        book_wide = book_df.pivot(index="date", columns="ticker", values="book_value")
        # Align book value dates to price dates by forward-filling using index union
        union_idx = book_wide.index.union(close_m.index)
        book_wide_m = book_wide.reindex(union_idx).ffill().reindex(close_m.index)

        # Fetch shares outstanding
        meta_indexed = metadata_df.set_index("ticker")
        shares = meta_indexed["shares_outstanding"].reindex(close_wide.columns)

        # Market Cap = Price × Shares
        market_cap = close_m.multiply(shares, axis=1)

        # B/P = Book Value / Market Cap
        bp = book_wide_m / market_cap

        if date not in bp.index:
            return pd.Series(np.nan, index=close_wide.columns)

        return bp.loc[date]


class GrossProfitability(BaseAlpha):
    """Gross Profitability factor: Gross Profit / Total Assets."""

    def compute(
        self,
        engine: Engine,
        prices_df: pd.DataFrame,
        metadata_df: pd.DataFrame,
        fundamentals_df: pd.DataFrame,
        date: pd.Timestamp,
        config: dict,
    ) -> pd.Series:
        if prices_df.empty:
            return pd.Series(dtype=float)

        close_wide = prices_df["adj_close"].unstack("ticker").sort_index()
        close_m = close_wide.ffill().resample("ME").last()

        if fundamentals_df.empty:
            self.logger.warning("No fundamental data available for GrossProfitability.")
            return pd.Series(np.nan, index=close_wide.columns)

        # Extract gross profit and assets
        gp_df = fundamentals_df.dropna(subset=["gross_profit"])
        assets_df = fundamentals_df.dropna(subset=["total_assets"])

        if gp_df.empty or assets_df.empty:
            return pd.Series(np.nan, index=close_wide.columns)

        gp_wide = gp_df.pivot(index="date", columns="ticker", values="gross_profit")
        assets_wide = assets_df.pivot(index="date", columns="ticker", values="total_assets")

        # Forward fill to month-end price dates using index union
        union_gp = gp_wide.index.union(close_m.index)
        gp_wide_m = gp_wide.reindex(union_gp).ffill().reindex(close_m.index)

        union_assets = assets_wide.index.union(close_m.index)
        assets_wide_m = assets_wide.reindex(union_assets).ffill().reindex(close_m.index)

        # Gross Profitability = Gross Profit / Total Assets
        gp_factor = gp_m = gp_wide_m / assets_wide_m

        if date not in gp_factor.index:
            return pd.Series(np.nan, index=close_wide.columns)

        return gp_factor.loc[date]


class LowVolatility(BaseAlpha):
    """Low Volatility factor: Standard deviation of daily returns over the past N trading days.

    Note: Standard low vol factor is inversely related to vol (e.g. -1 * volatility),
    so we return the standard deviation of daily returns. Portfolios can sort ascending.
    """

    def compute(
        self,
        engine: Engine,
        prices_df: pd.DataFrame,
        metadata_df: pd.DataFrame,
        fundamentals_df: pd.DataFrame,
        date: pd.Timestamp,
        config: dict,
    ) -> pd.Series:
        if prices_df.empty:
            return pd.Series(dtype=float)

        # Get daily pricing
        close_wide = prices_df["adj_close"].unstack("ticker").sort_index()
        
        # Calculate daily simple returns
        daily_returns = close_wide.pct_change(fill_method=None)

        # Compute standard deviation over the rolling window (e.g. 252 days)
        lookback = config.get("volatility_lookback_days", 252)
        min_periods = max(1, lookback // 2)
        rolling_std = daily_returns.rolling(window=lookback, min_periods=min_periods).std()

        # Resample std to month-end frequency using index union to avoid losing intermediate values
        close_m_index = close_wide.resample("ME").last().index
        union_vol = rolling_std.index.union(close_m_index)
        vol_m = rolling_std.reindex(union_vol).ffill().reindex(close_m_index)

        if date not in vol_m.index:
            return pd.Series(np.nan, index=close_wide.columns)

        return vol_m.loc[date]
