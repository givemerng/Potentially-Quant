"""Tests for ReturnMatrix — the core research sandbox."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.factors.return_matrix import ReturnMatrix


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_wide(n_dates: int = 24, n_tickers: int = 3, seed: int = 42) -> pd.DataFrame:
    """Create a wide returns DataFrame (dates × tickers) with realistic values."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-31", periods=n_dates, freq="ME")
    tickers = [f"T{i}" for i in range(n_tickers)]
    data = rng.normal(loc=0.005, scale=0.04, size=(n_dates, n_tickers))
    return pd.DataFrame(data, index=dates, columns=tickers)


@pytest.fixture
def wide_df() -> pd.DataFrame:
    return _make_wide()


@pytest.fixture
def rm(wide_df: pd.DataFrame) -> ReturnMatrix:
    return ReturnMatrix(wide_df)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_index_is_datetime(self, rm: ReturnMatrix) -> None:
        assert isinstance(rm.data.index, pd.DatetimeIndex)

    def test_index_is_sorted(self, rm: ReturnMatrix) -> None:
        assert rm.data.index.is_monotonic_increasing

    def test_shape_preserved(self, wide_df: pd.DataFrame, rm: ReturnMatrix) -> None:
        assert rm.shape == wide_df.shape

    def test_from_long_roundtrip(self, wide_df: pd.DataFrame) -> None:
        """Build from long format and verify values match original wide DataFrame."""
        long = wide_df.stack(future_stack=True).reset_index()
        long.columns = ["date", "ticker", "log_return"]

        rm = ReturnMatrix.from_long(long, return_col="log_return")

        for col in wide_df.columns:
            pd.testing.assert_series_equal(
                rm.data[col].sort_index(),
                wide_df[col].sort_index(),
                check_names=False,
                check_freq=False,
                rtol=1e-10,
            )

    def test_from_long_missing_columns_raises(self) -> None:
        with pytest.raises(ValueError, match="Missing columns"):
            ReturnMatrix.from_long(pd.DataFrame({"date": [], "ticker": []}), return_col="log_return")

    def test_repr(self, rm: ReturnMatrix) -> None:
        r = repr(rm)
        assert "ReturnMatrix" in r
        assert "tickers=" in r


# ---------------------------------------------------------------------------
# Gap filling
# ---------------------------------------------------------------------------

class TestFillGaps:
    def test_none_method_preserves_nans(self) -> None:
        data = _make_wide()
        data.iloc[0, 0] = np.nan
        rm = ReturnMatrix(data).fill_gaps(method="none")
        assert np.isnan(rm.data.iloc[0, 0])

    def test_zero_method_fills_nans(self) -> None:
        data = _make_wide()
        data.iloc[0, 0] = np.nan
        rm = ReturnMatrix(data).fill_gaps(method="zero")
        assert rm.data.iloc[0, 0] == 0.0

    def test_drop_cols_removes_bad_ticker(self) -> None:
        data = _make_wide(n_dates=30, n_tickers=3)
        data.iloc[:10, 0] = np.nan  # 10 consecutive NaNs in ticker T0
        rm = ReturnMatrix(data).fill_gaps(method="drop_cols", max_consecutive=5)
        assert "T0" not in rm.tickers

    def test_drop_cols_keeps_good_tickers(self) -> None:
        data = _make_wide(n_dates=30, n_tickers=3)
        data.iloc[:3, 0] = np.nan  # only 3 consecutive NaNs — below threshold
        original_tickers = list(data.columns)
        rm = ReturnMatrix(data).fill_gaps(method="drop_cols", max_consecutive=5)
        assert original_tickers[0] in rm.tickers  # T0 should survive

    def test_invalid_method_raises(self, rm: ReturnMatrix) -> None:
        with pytest.raises(ValueError, match="Unknown gap-filling method"):
            rm.fill_gaps(method="invalid_method")

    def test_fill_gaps_returns_new_instance(self, rm: ReturnMatrix) -> None:
        """fill_gaps should not mutate the original ReturnMatrix."""
        new_rm = rm.fill_gaps(method="zero")
        assert new_rm is not rm


# ---------------------------------------------------------------------------
# Universe trimming
# ---------------------------------------------------------------------------

class TestTrimByUniverse:
    def test_excluded_ticker_date_becomes_nan(self) -> None:
        dates = pd.date_range("2020-01-31", periods=3, freq="ME")
        data = pd.DataFrame(
            {"A": [0.01, 0.02, 0.03], "B": [0.04, 0.05, 0.06]},
            index=dates,
        )
        rm = ReturnMatrix(data)

        membership = pd.DataFrame(
            [
                {"date": dates[0], "ticker": "A", "in_universe": True},
                {"date": dates[0], "ticker": "B", "in_universe": False},  # excluded
                {"date": dates[1], "ticker": "A", "in_universe": True},
                {"date": dates[1], "ticker": "B", "in_universe": True},
                {"date": dates[2], "ticker": "A", "in_universe": True},
                {"date": dates[2], "ticker": "B", "in_universe": True},
            ]
        )

        trimmed = rm.trim_by_universe(membership)

        # B on date[0] should be NaN (excluded from universe)
        assert np.isnan(trimmed.data.loc[dates[0], "B"])
        # A on date[0] should be unchanged
        assert trimmed.data.loc[dates[0], "A"] == pytest.approx(0.01)
        # B on date[1] should be present
        assert not np.isnan(trimmed.data.loc[dates[1], "B"])

    def test_trim_does_not_mutate_original(self) -> None:
        dates = pd.date_range("2020-01-31", periods=2, freq="ME")
        data = pd.DataFrame({"A": [0.01, 0.02]}, index=dates)
        rm = ReturnMatrix(data.copy())
        membership = pd.DataFrame([
            {"date": dates[0], "ticker": "A", "in_universe": False},
            {"date": dates[1], "ticker": "A", "in_universe": True},
        ])
        _ = rm.trim_by_universe(membership)
        # Original should be unchanged
        assert rm.data.loc[dates[0], "A"] == pytest.approx(0.01)

    def test_unknown_ticker_in_membership_ignored(self) -> None:
        dates = pd.date_range("2020-01-31", periods=2, freq="ME")
        data = pd.DataFrame({"A": [0.01, 0.02]}, index=dates)
        rm = ReturnMatrix(data)
        membership = pd.DataFrame([
            {"date": dates[0], "ticker": "UNKNOWN", "in_universe": True},
            {"date": dates[1], "ticker": "UNKNOWN", "in_universe": True},
        ])
        # Should not raise; unknown ticker just has no effect
        trimmed = rm.trim_by_universe(membership)
        assert "A" in trimmed.data.columns


# ---------------------------------------------------------------------------
# Cross-sectional statistics
# ---------------------------------------------------------------------------

class TestCrossSectionalStats:
    def test_stats_keys_present(self, rm: ReturnMatrix) -> None:
        date = rm.dates[0]
        stats = rm.cross_sectional_stats(date)
        expected_keys = {"mean", "std", "median", "skewness", "kurtosis", "n_stocks", "pct_positive"}
        assert expected_keys == set(stats.keys())

    def test_n_stocks_correct(self, rm: ReturnMatrix) -> None:
        date = rm.dates[0]
        stats = rm.cross_sectional_stats(date)
        assert stats["n_stocks"] == rm.shape[1]

    def test_mean_in_range(self, rm: ReturnMatrix) -> None:
        date = rm.dates[0]
        stats = rm.cross_sectional_stats(date)
        # Mean of normal(0.005, 0.04) should be close to 0.005
        assert -0.5 < stats["mean"] < 0.5

    def test_pct_positive_between_0_and_1(self, rm: ReturnMatrix) -> None:
        date = rm.dates[0]
        stats = rm.cross_sectional_stats(date)
        assert 0.0 <= stats["pct_positive"] <= 1.0

    def test_unknown_date_raises(self, rm: ReturnMatrix) -> None:
        with pytest.raises(KeyError):
            rm.cross_sectional_stats(pd.Timestamp("1900-01-01"))

    def test_all_nan_date_returns_nans(self) -> None:
        dates = pd.date_range("2020-01-31", periods=2, freq="ME")
        data = pd.DataFrame({"A": [np.nan, 0.01], "B": [np.nan, 0.02]}, index=dates)
        rm = ReturnMatrix(data)
        stats = rm.cross_sectional_stats(dates[0])
        assert np.isnan(stats["mean"])


# ---------------------------------------------------------------------------
# Annual stats
# ---------------------------------------------------------------------------

class TestAnnualStats:
    def test_returns_dataframe(self, rm: ReturnMatrix) -> None:
        result = rm.annual_stats()
        assert isinstance(result, pd.DataFrame)

    def test_annual_stats_columns(self, rm: ReturnMatrix) -> None:
        result = rm.annual_stats()
        expected = {"n_obs", "mean", "std", "skewness", "excess_kurtosis"}
        assert expected.issubset(set(result.columns))

    def test_index_is_year(self) -> None:
        data = _make_wide(n_dates=36)
        rm = ReturnMatrix(data)
        result = rm.annual_stats()
        assert result.index.name == "year"
        for year in result.index:
            assert isinstance(year, (int, np.integer))

    def test_normal_distribution_near_zero_excess_kurtosis(self) -> None:
        """For large normal samples, excess kurtosis should be near 0."""
        rng = np.random.default_rng(123)
        dates = pd.date_range("2015-01-31", periods=120, freq="ME")
        tickers = [f"T{i}" for i in range(100)]
        data = pd.DataFrame(
            rng.normal(0, 0.04, size=(120, 100)),
            index=dates,
            columns=tickers,
        )
        rm = ReturnMatrix(data)
        result = rm.annual_stats()
        # Excess kurtosis for normal should be close to 0 with large N
        for _, row in result.iterrows():
            assert abs(row["excess_kurtosis"]) < 1.5

    def test_annual_stats_empty_returns_empty(self) -> None:
        rm = ReturnMatrix(pd.DataFrame())
        result = rm.annual_stats()
        assert result.empty
