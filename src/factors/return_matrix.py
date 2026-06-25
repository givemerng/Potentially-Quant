"""ReturnMatrix — the core research sandbox for cross-sectional factor work.

Wraps a wide-format DataFrame (dates × tickers) with helpers for gap-filling,
universe trimming, and distributional analysis.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


class ReturnMatrix:
    """Assets × dates return matrix for cross-sectional factor research.

    The underlying ``data`` attribute is a DataFrame with:
    - Row index: ``pd.DatetimeIndex`` (trading dates or month-ends).
    - Columns: ticker strings.
    - Values: returns (log or simple) as floats.

    Design principle: **never forward-fill returns**. Only prices should
    be carried forward before computing returns. This class enforces that
    by operating exclusively on returns.
    """

    def __init__(self, data: pd.DataFrame) -> None:
        """
        Args:
            data: Wide DataFrame (dates × tickers) of return values.
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            data = data.copy()
            data.index = pd.to_datetime(data.index)
        self.data: pd.DataFrame = data.sort_index()

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_long(cls, long_df: pd.DataFrame, return_col: str = "log_return") -> "ReturnMatrix":
        """Build a ReturnMatrix from a long-format DataFrame.

        Args:
            long_df: DataFrame with columns [date, ticker, <return_col>].
            return_col: Name of the return column to pivot.

        Returns:
            A new ReturnMatrix instance.
        """
        required = {"date", "ticker", return_col}
        missing = required - set(long_df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        wide = long_df.pivot(index="date", columns="ticker", values=return_col)
        wide.index = pd.to_datetime(wide.index)
        wide.index = wide.index.as_unit("ns")  # strip freq metadata for clean roundtrip
        wide.index.name = "date"
        wide.columns.name = "ticker"
        return cls(wide)

    # ------------------------------------------------------------------
    # Gap handling (prices only, never returns)
    # ------------------------------------------------------------------

    def fill_gaps(self, method: str = "none", max_consecutive: int = 5) -> "ReturnMatrix":
        """Handle gaps in the return matrix.

        Supported methods:
        - ``'none'``: Leave NaN as-is (recommended for returns).
        - ``'zero'``: Fill NaN with 0 (treat missing trading days as flat).
        - ``'drop_cols'``: Drop tickers with more than ``max_consecutive``
          consecutive NaN observations.

        Note:
            **Never forward-fill returns** — that would propagate stale
            signals. Only prices are forward-filled before computing returns.

        Args:
            method: Gap-filling strategy (``'none'``, ``'zero'``,
                    ``'drop_cols'``).
            max_consecutive: Threshold for ``'drop_cols'`` mode.

        Returns:
            New ReturnMatrix with gaps handled.
        """
        if method == "none":
            return ReturnMatrix(self.data.copy())
        elif method == "zero":
            return ReturnMatrix(self.data.fillna(0.0))
        elif method == "drop_cols":
            def max_consecutive_nans(series: pd.Series) -> int:
                mask = series.isna()
                groups = (mask != mask.shift()).cumsum()
                return int(mask.groupby(groups).sum().max())

            bad_tickers = [
                col for col in self.data.columns
                if max_consecutive_nans(self.data[col]) > max_consecutive
            ]
            cleaned = self.data.drop(columns=bad_tickers)
            return ReturnMatrix(cleaned)
        else:
            raise ValueError(f"Unknown gap-filling method: {method!r}. Use 'none', 'zero', or 'drop_cols'.")

    # ------------------------------------------------------------------
    # Universe trimming
    # ------------------------------------------------------------------

    def trim_by_universe(self, membership: pd.DataFrame) -> "ReturnMatrix":
        """Mask returns for tickers not in the universe on each date.

        Uses point-in-time membership: for each (date, ticker) pair,
        if ``in_universe`` is False (or the pair is absent from
        ``membership``), the return is set to NaN.

        Args:
            membership: DataFrame with columns [date, ticker, in_universe].
                        ``date`` should be date-like (converted internally).

        Returns:
            New ReturnMatrix with non-universe returns masked as NaN.
        """
        membership = membership.copy()
        membership["date"] = pd.to_datetime(membership["date"])

        # Build a wide boolean mask aligned to our return matrix
        mask_wide = membership.pivot(index="date", columns="ticker", values="in_universe")
        mask_wide = mask_wide.reindex(index=self.data.index, columns=self.data.columns)

        # Where mask is False or NaN, set returns to NaN
        masked = self.data.where(mask_wide == True)  # noqa: E712
        return ReturnMatrix(masked)

    # ------------------------------------------------------------------
    # Cross-sectional statistics
    # ------------------------------------------------------------------

    def cross_sectional_stats(self, date: pd.Timestamp) -> Dict[str, float]:
        """Compute cross-sectional distribution stats for a single date.

        Args:
            date: The date to analyse.

        Returns:
            Dictionary with keys: mean, std, median, skewness, kurtosis,
            n_stocks, pct_positive.
        """
        date = pd.Timestamp(date)
        if date not in self.data.index:
            raise KeyError(f"Date {date.date()} not found in ReturnMatrix.")

        row = self.data.loc[date].dropna()
        if row.empty:
            return {k: float("nan") for k in ["mean", "std", "median", "skewness", "kurtosis", "n_stocks", "pct_positive"]}

        return {
            "mean": float(row.mean()),
            "std": float(row.std()),
            "median": float(row.median()),
            "skewness": float(row.skew()),
            "kurtosis": float(row.kurt()),  # excess kurtosis
            "n_stocks": int(row.count()),
            "pct_positive": float((row > 0).mean()),
        }

    def annual_stats(self) -> pd.DataFrame:
        """Compute cross-sectional skewness and kurtosis by calendar year.

        Useful for the fat-tail analysis required by the Week 2 deliverable.
        Pools all cross-sectional return observations within each year and
        computes distribution moments on the pooled sample.

        Returns:
            DataFrame indexed by year with columns:
            [n_obs, mean, std, skewness, kurtosis, excess_kurtosis].
        """
        if self.data.empty:
            return pd.DataFrame(
                columns=["n_obs", "mean", "std", "skewness", "excess_kurtosis"]
            ).rename_axis("year")
        records = []
        years = self.data.index.year.unique()

        for year in sorted(years):
            year_mask = self.data.index.year == year
            pooled = self.data.loc[year_mask].values.flatten()
            pooled = pooled[~np.isnan(pooled)]

            if len(pooled) < 10:
                continue

            records.append(
                {
                    "year": int(year),
                    "n_obs": len(pooled),
                    "mean": float(np.mean(pooled)),
                    "std": float(np.std(pooled, ddof=1)),
                    "skewness": float(pd.Series(pooled).skew()),
                    "excess_kurtosis": float(pd.Series(pooled).kurt()),
                }
            )

        return pd.DataFrame(records).set_index("year")

    def cross_sectional_stats_over_time(self) -> pd.DataFrame:
        """Compute cross-sectional distribution stats for each date."""
        if self.data.empty:
            return pd.DataFrame(
                columns=["mean", "std", "median", "skewness", "kurtosis", "n_stocks", "pct_positive"]
            ).rename_axis("date")

        records = []
        for date in self.data.index:
            stats_for_date = self.cross_sectional_stats(date)
            stats_for_date["date"] = pd.Timestamp(date)
            records.append(stats_for_date)

        result = pd.DataFrame(records).set_index("date").sort_index()
        result.index.name = "date"
        return result

    def to_long(self, value_name: str = "value") -> pd.DataFrame:
        """Convert the wide matrix to long format."""
        long_df = self.data.stack(future_stack=True).rename(value_name).reset_index()
        long_df.columns = ["date", "ticker", value_name]
        return long_df

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @property
    def tickers(self) -> list[str]:
        """List of tickers in the matrix."""
        return list(self.data.columns)

    @property
    def dates(self) -> pd.DatetimeIndex:
        """Sorted DatetimeIndex of the matrix."""
        return self.data.index

    @property
    def shape(self) -> tuple[int, int]:
        """(n_dates, n_tickers) shape of the matrix."""
        return self.data.shape

    def __repr__(self) -> str:
        return (
            f"ReturnMatrix(dates={len(self.data)}, tickers={len(self.data.columns)}, "
            f"start={self.data.index.min().date()}, end={self.data.index.max().date()})"
        )
