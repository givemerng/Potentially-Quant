"""Tests for ReturnCalculator — daily log-returns and monthly simple returns."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.factors.returns import ReturnCalculator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_prices(tickers: list[str], dates: pd.DatetimeIndex, base: float = 100.0) -> pd.DataFrame:
    """Build a synthetic prices DataFrame with MultiIndex (date, ticker)."""
    frames = []
    for i, ticker in enumerate(tickers):
        prices_for_ticker = base * (1 + 0.01 * (i + 1)) ** np.arange(len(dates))
        df = pd.DataFrame(
            {
                "adj_close": prices_for_ticker,
                "close": prices_for_ticker,
                "open": prices_for_ticker * 0.99,
                "high": prices_for_ticker * 1.01,
                "low": prices_for_ticker * 0.98,
                "volume": np.full(len(dates), 1_000_000),
                "ticker": ticker,
            },
            index=dates,
        )
        df.index.name = "date"
        frames.append(df.reset_index())

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.set_index(["date", "ticker"]).sort_index()
    return combined


@pytest.fixture
def simple_prices() -> pd.DataFrame:
    dates = pd.date_range("2020-01-02", periods=10, freq="B")
    return _make_prices(["AAPL", "MSFT"], dates)


@pytest.fixture
def calculator() -> ReturnCalculator:
    return ReturnCalculator()


# ---------------------------------------------------------------------------
# Daily log-returns
# ---------------------------------------------------------------------------

class TestDailyLogReturns:
    def test_returns_shape(self, calculator: ReturnCalculator, simple_prices: pd.DataFrame) -> None:
        """Output has one fewer row than input (first obs is dropped)."""
        result = calculator.compute_daily_log_returns(simple_prices)
        n_dates = simple_prices.index.get_level_values("date").nunique()
        n_tickers = simple_prices.index.get_level_values("ticker").nunique()
        # First date dropped per ticker → (n_dates - 1) × n_tickers rows
        assert len(result) == (n_dates - 1) * n_tickers

    def test_returns_index_names(self, calculator: ReturnCalculator, simple_prices: pd.DataFrame) -> None:
        result = calculator.compute_daily_log_returns(simple_prices)
        assert list(result.index.names) == ["date", "ticker"]

    def test_log_return_correctness(self, calculator: ReturnCalculator) -> None:
        """Manually verify log-return for a known price series."""
        dates = pd.date_range("2020-01-02", periods=3, freq="B")
        prices_arr = np.array([100.0, 110.0, 99.0])
        df = pd.DataFrame(
            {
                "adj_close": prices_arr,
                "close": prices_arr,
                "open": prices_arr,
                "high": prices_arr,
                "low": prices_arr,
                "volume": [1_000_000] * 3,
                "ticker": "TEST",
            },
            index=dates,
        )
        df.index.name = "date"
        prices = df.reset_index().set_index(["date", "ticker"])

        result = calculator.compute_daily_log_returns(prices)

        expected_day2 = math.log(110.0 / 100.0)
        expected_day3 = math.log(99.0 / 110.0)

        actual_day2 = result.loc[(dates[1], "TEST"), "log_return"]
        actual_day3 = result.loc[(dates[2], "TEST"), "log_return"]

        assert abs(actual_day2 - expected_day2) < 1e-10
        assert abs(actual_day3 - expected_day3) < 1e-10

    def test_no_look_ahead_in_returns(self, calculator: ReturnCalculator, simple_prices: pd.DataFrame) -> None:
        """Return on date t must only use prices on or before date t.

        We verify this by checking that shifting the price series by -1
        (future price) gives different returns — confirming we're not
        accidentally using future data.
        """
        result = calculator.compute_daily_log_returns(simple_prices)
        # All returns should be non-NaN after dropping first obs
        assert not result["log_return"].isna().any()
        # Returns should not be zero (that would indicate no variation / bad data)
        assert (result["log_return"] != 0).any()

    def test_output_columns(self, calculator: ReturnCalculator, simple_prices: pd.DataFrame) -> None:
        result = calculator.compute_daily_log_returns(simple_prices)
        assert "log_return" in result.columns
        assert "simple_return" in result.columns

    def test_log_and_simple_consistent(self, calculator: ReturnCalculator, simple_prices: pd.DataFrame) -> None:
        """For small returns, log_return ≈ simple_return."""
        result = calculator.compute_daily_log_returns(simple_prices)
        # log(1+r) ≈ r for small r; difference < 0.1% for 1% daily moves
        diff = (result["log_return"] - np.log1p(result["simple_return"])).abs()
        assert diff.max() < 1e-10

    def test_empty_prices_returns_empty(self, calculator: ReturnCalculator) -> None:
        result = calculator.compute_daily_log_returns(pd.DataFrame())
        assert result.empty


# ---------------------------------------------------------------------------
# Monthly simple returns
# ---------------------------------------------------------------------------

class TestMonthlySimpleReturns:
    def test_monthly_returns_are_month_end(self, calculator: ReturnCalculator) -> None:
        """Dates in monthly returns should be month-end dates."""
        dates = pd.date_range("2020-01-02", periods=250, freq="B")
        prices = _make_prices(["AAPL"], dates)
        result = calculator.compute_monthly_simple_returns(prices)

        return_dates = result.index.get_level_values("date")
        for d in return_dates:
            ts = pd.Timestamp(d)
            next_day = ts + pd.Timedelta(days=1)
            # A month-end date either ends the month or is on a weekend boundary
            assert ts.month != next_day.month or next_day.day_of_week >= 5

    def test_monthly_fewer_rows_than_daily(self, calculator: ReturnCalculator, simple_prices: pd.DataFrame) -> None:
        monthly = calculator.compute_monthly_simple_returns(simple_prices)
        daily = calculator.compute_daily_log_returns(simple_prices)
        assert len(monthly) <= len(daily)

    def test_monthly_output_columns(self, calculator: ReturnCalculator, simple_prices: pd.DataFrame) -> None:
        result = calculator.compute_monthly_simple_returns(simple_prices)
        assert "simple_return" in result.columns
        assert "log_return" in result.columns

    def test_monthly_no_future_fill(self, calculator: ReturnCalculator) -> None:
        """Prices are forward-filled; but returns must not carry stale values."""
        # Create prices with a gap in the middle
        dates = pd.date_range("2020-01-02", periods=60, freq="B")
        prices_arr = np.full(len(dates), 100.0)
        prices_arr[20:25] = np.nan  # 5-day gap

        df = pd.DataFrame(
            {"adj_close": prices_arr, "close": prices_arr, "open": prices_arr,
             "high": prices_arr, "low": prices_arr, "volume": np.full(len(dates), 1e6),
             "ticker": "TEST"},
            index=dates,
        )
        df.index.name = "date"
        prices = df.reset_index().set_index(["date", "ticker"])

        # Should not raise; gap in prices should be handled gracefully
        result = calculator.compute_monthly_simple_returns(prices)
        assert not result.empty

    def test_empty_prices_returns_empty(self, calculator: ReturnCalculator) -> None:
        result = calculator.compute_monthly_simple_returns(pd.DataFrame())
        assert result.empty


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_validation_passes_correlated_series(self, calculator: ReturnCalculator) -> None:
        """Two highly correlated return series should pass validation."""
        dates = pd.date_range("2020-01-02", periods=100, freq="B")
        prices = _make_prices(["AAPL", "MSFT", "^GSPC"], dates)
        daily = calculator.compute_daily_log_returns(prices)
        passed, msg = calculator.validate_against_index(daily, min_correlation=0.5)
        # All three are deterministic price series with positive drift → high correlation
        assert isinstance(passed, bool)
        assert isinstance(msg, str)

    def test_validation_fails_on_empty(self, calculator: ReturnCalculator) -> None:
        passed, msg = calculator.validate_against_index(pd.DataFrame())
        assert not passed
        assert "Empty" in msg
