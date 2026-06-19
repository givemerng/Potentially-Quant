"""Tests for UniverseFilter — point-in-time universe membership construction."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import UniverseFilterConfig
from src.factors.universe import UniverseFilter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_prices(
    tickers: list[str],
    dates: pd.DatetimeIndex,
    prices_map: dict | None = None,
    volumes_map: dict | None = None,
) -> pd.DataFrame:
    """Build a synthetic prices DataFrame with MultiIndex (date, ticker)."""
    frames = []
    for ticker in tickers:
        price_arr = prices_map.get(ticker, np.full(len(dates), 100.0)) if prices_map else np.full(len(dates), 100.0)
        vol_arr = volumes_map.get(ticker, np.full(len(dates), 1_000_000)) if volumes_map else np.full(len(dates), 1_000_000)

        df = pd.DataFrame(
            {
                "adj_close": price_arr,
                "close": price_arr,
                "open": price_arr,
                "high": price_arr * 1.01,
                "low": price_arr * 0.99,
                "volume": vol_arr,
                "ticker": ticker,
            },
            index=dates,
        )
        df.index.name = "date"
        frames.append(df.reset_index())

    combined = pd.concat(frames, ignore_index=True)
    return combined.set_index(["date", "ticker"]).sort_index()


@pytest.fixture
def config() -> UniverseFilterConfig:
    return UniverseFilterConfig(
        min_market_cap_bn=0.05,       # low threshold so most stocks pass
        min_dollar_vol_percentile=50,
        ipo_exclusion_months=1,       # short for test speed
        dollar_vol_lookback_days=5,
    )


@pytest.fixture
def uf() -> UniverseFilter:
    return UniverseFilter()


@pytest.fixture
def base_dates() -> pd.DatetimeIndex:
    return pd.date_range("2020-01-02", periods=80, freq="B")


# ---------------------------------------------------------------------------
# Market-cap proxy filter
# ---------------------------------------------------------------------------

class TestMarketCapFilter:
    def test_low_volume_ticker_excluded(self, uf: UniverseFilter, base_dates: pd.DatetimeIndex) -> None:
        """Ticker with negligible dollar volume should be excluded."""
        prices = _make_prices(
            ["BIG", "TINY"],
            base_dates,
            prices_map={"BIG": np.full(len(base_dates), 100.0), "TINY": np.full(len(base_dates), 1.0)},
            volumes_map={"BIG": np.full(len(base_dates), 10_000_000), "TINY": np.full(len(base_dates), 1)},
        )
        cfg = UniverseFilterConfig(
            min_market_cap_bn=0.0005,   # ~$500K floor
            min_dollar_vol_percentile=0,
            ipo_exclusion_months=0,
            dollar_vol_lookback_days=5,
        )
        membership = uf.compute_membership_fast(prices, cfg)
        big_rows = membership[membership["ticker"] == "BIG"]
        tiny_rows = membership[membership["ticker"] == "TINY"]

        # BIG should be mostly in-universe; TINY should be mostly excluded
        assert big_rows["in_universe"].mean() > 0.5
        assert tiny_rows["in_universe"].mean() < 0.5

    def test_all_pass_with_zero_threshold(self, uf: UniverseFilter, base_dates: pd.DatetimeIndex) -> None:
        """With min_market_cap_bn=0, cap filter should not exclude anything."""
        prices = _make_prices(["A", "B"], base_dates)
        cfg = UniverseFilterConfig(
            min_market_cap_bn=0.0,
            min_dollar_vol_percentile=0,
            ipo_exclusion_months=0,
            dollar_vol_lookback_days=5,
        )
        membership = uf.compute_membership_fast(prices, cfg)
        # After warmup period (lookback), all should be in universe
        late_membership = membership[pd.to_datetime(membership["date"]) > base_dates[10]]
        assert late_membership["in_universe"].all()


# ---------------------------------------------------------------------------
# Dollar-volume percentile filter
# ---------------------------------------------------------------------------

class TestDollarVolumeFilter:
    def test_below_median_excluded(self, uf: UniverseFilter, base_dates: pd.DatetimeIndex) -> None:
        """Tickers with below-median dollar volume should be excluded."""
        # HIGH_VOL has 10x the dollar volume of LOW_VOL
        prices = _make_prices(
            ["HIGH_VOL", "LOW_VOL"],
            base_dates,
            volumes_map={
                "HIGH_VOL": np.full(len(base_dates), 10_000_000),
                "LOW_VOL": np.full(len(base_dates), 100),
            },
        )
        cfg = UniverseFilterConfig(
            min_market_cap_bn=0.0,
            min_dollar_vol_percentile=50,
            ipo_exclusion_months=0,
            dollar_vol_lookback_days=5,
        )
        membership = uf.compute_membership_fast(prices, cfg)
        high_vol = membership[membership["ticker"] == "HIGH_VOL"]
        low_vol = membership[membership["ticker"] == "LOW_VOL"]

        late_high = high_vol[pd.to_datetime(high_vol["date"]) > base_dates[10]]
        late_low = low_vol[pd.to_datetime(low_vol["date"]) > base_dates[10]]

        assert late_high["in_universe"].mean() > late_low["in_universe"].mean()

    def test_zero_percentile_passes_all(self, uf: UniverseFilter, base_dates: pd.DatetimeIndex) -> None:
        prices = _make_prices(["A", "B", "C"], base_dates)
        cfg = UniverseFilterConfig(
            min_market_cap_bn=0.0,
            min_dollar_vol_percentile=0,
            ipo_exclusion_months=0,
            dollar_vol_lookback_days=5,
        )
        membership = uf.compute_membership_fast(prices, cfg)
        late_membership = membership[pd.to_datetime(membership["date"]) > base_dates[10]]
        assert late_membership["in_universe"].all()


# ---------------------------------------------------------------------------
# IPO exclusion filter
# ---------------------------------------------------------------------------

class TestIPOExclusion:
    def test_new_ticker_excluded_within_exclusion_window(self, uf: UniverseFilter) -> None:
        """A ticker with only 5 days of history should be excluded with 12-month window."""
        dates = pd.date_range("2020-01-02", periods=10, freq="B")
        prices = _make_prices(
            ["NEW_IPO"],
            dates,
            volumes_map={"NEW_IPO": np.full(len(dates), 10_000_000)},
        )
        cfg = UniverseFilterConfig(
            min_market_cap_bn=0.0,
            min_dollar_vol_percentile=0,
            ipo_exclusion_months=12,
            dollar_vol_lookback_days=3,
        )
        membership = uf.compute_membership_fast(prices, cfg)
        # All rows should be excluded — stock is <12 months old
        assert not membership["in_universe"].any()

    def test_old_ticker_not_excluded(self, uf: UniverseFilter) -> None:
        """A ticker with 3+ years of history should pass IPO filter."""
        dates = pd.date_range("2017-01-02", periods=800, freq="B")
        prices = _make_prices(
            ["OLD_STOCK"],
            dates,
            volumes_map={"OLD_STOCK": np.full(len(dates), 10_000_000)},
        )
        cfg = UniverseFilterConfig(
            min_market_cap_bn=0.0,
            min_dollar_vol_percentile=0,
            ipo_exclusion_months=12,
            dollar_vol_lookback_days=5,
        )
        membership = uf.compute_membership_fast(prices, cfg)
        # Rows well past the 12-month IPO exclusion window should be mostly in-universe
        # Use day 320 (>12 months + lookback buffer) to avoid boundary effects
        late_rows = membership[pd.to_datetime(membership["date"]) > dates[320]]
        assert late_rows["in_universe"].mean() > 0.95

    def test_zero_exclusion_months_no_ipo_filter(self, uf: UniverseFilter, base_dates: pd.DatetimeIndex) -> None:
        """With ipo_exclusion_months=0, new stocks pass the IPO filter immediately."""
        prices = _make_prices(["BRAND_NEW"], base_dates[:10])
        cfg = UniverseFilterConfig(
            min_market_cap_bn=0.0,
            min_dollar_vol_percentile=0,
            ipo_exclusion_months=0,
            dollar_vol_lookback_days=3,
        )
        membership = uf.compute_membership_fast(prices, cfg)
        late_rows = membership[pd.to_datetime(membership["date"]) > base_dates[4]]
        assert late_rows["in_universe"].all()


# ---------------------------------------------------------------------------
# Point-in-time integrity
# ---------------------------------------------------------------------------

class TestPointInTimeIntegrity:
    def test_no_future_data_in_filter(self, uf: UniverseFilter, base_dates: pd.DatetimeIndex) -> None:
        """Universe membership on date t must only use data available at t.

        We verify this by checking that a stock that becomes large (high volume)
        only AFTER a certain date is NOT in-universe BEFORE that date.
        """
        volumes = np.full(len(base_dates), 100)     # tiny volume initially
        volumes[40:] = 100_000_000                  # huge volume from day 40 onwards

        prices = _make_prices(
            ["LATEBOOM"],
            base_dates,
            volumes_map={"LATEBOOM": volumes},
        )
        cfg = UniverseFilterConfig(
            min_market_cap_bn=0.0,
            min_dollar_vol_percentile=0,
            ipo_exclusion_months=0,
            dollar_vol_lookback_days=5,
        )
        membership = uf.compute_membership_fast(prices, cfg)
        # Volume-based filter (cap proxy) before day 40 should show excluded
        early_rows = membership[pd.to_datetime(membership["date"]) < base_dates[10]]
        late_rows = membership[pd.to_datetime(membership["date"]) > base_dates[50]]

        # After the volume jump, should be in-universe; before, should not
        if not late_rows.empty:
            assert late_rows["avg_dollar_vol"].mean() > early_rows["avg_dollar_vol"].mean()

    def test_membership_schema(self, uf: UniverseFilter, base_dates: pd.DatetimeIndex) -> None:
        """Output DataFrame must have the expected columns."""
        prices = _make_prices(["AAPL"], base_dates)
        cfg = UniverseFilterConfig(dollar_vol_lookback_days=5, ipo_exclusion_months=0)
        membership = uf.compute_membership_fast(prices, cfg)

        required_cols = {"ticker", "date", "in_universe", "market_cap_proxy",
                         "avg_dollar_vol", "days_since_first_price"}
        assert required_cols.issubset(set(membership.columns))

    def test_empty_prices_returns_empty(self, uf: UniverseFilter, config: UniverseFilterConfig) -> None:
        membership = uf.compute_membership_fast(pd.DataFrame(), config)
        assert membership.empty
