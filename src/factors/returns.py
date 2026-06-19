"""Return calculation module for the Regime Alpha Engine.

Computes daily log-returns and monthly simple returns from price data,
validates against a benchmark index, and persists results to the DB.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy.engine import Engine

from src.data.db import returns_table, upsert_rows


@dataclass
class ReturnSummary:
    daily_rows_stored: int
    monthly_rows_stored: int
    validation_passed: bool
    validation_message: str


class ReturnCalculator:
    """Compute and store daily log-returns and monthly simple returns.

    All computations are strictly backward-looking (no look-ahead bias):
    return at date t uses only prices available on or before date t.
    """

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Core computations
    # ------------------------------------------------------------------

    def compute_daily_log_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Compute daily log-returns per ticker from a prices DataFrame.

        Args:
            prices: DataFrame with MultiIndex (date, ticker) and at least
                    an ``adj_close`` column.

        Returns:
            DataFrame with columns [log_return, simple_return] and the
            same (date, ticker) MultiIndex. The first observation per
            ticker is dropped (NaN).
        """
        if prices.empty:
            return pd.DataFrame(columns=["log_return", "simple_return"])

        adj = (
            prices["adj_close"]
            .unstack("ticker")  # shape: dates x tickers
            .sort_index()
        )
        log_ret = np.log(adj / adj.shift(1)).dropna(how="all")
        simple_ret = (adj / adj.shift(1) - 1).dropna(how="all")

        log_stacked = log_ret.stack(future_stack=True).rename("log_return")
        simple_stacked = simple_ret.stack(future_stack=True).rename("simple_return")

        result = pd.concat([log_stacked, simple_stacked], axis=1).dropna(how="all")
        result.index.names = ["date", "ticker"]
        self.logger.info("Computed %d daily return rows for %d tickers", len(result), adj.shape[1])
        return result

    def compute_monthly_simple_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Compute month-end simple returns per ticker.

        Prices are resampled to month-end using the last available
        adj_close; returns use only the resampled price series
        (no forward-filling of returns — only prices are carried forward).

        Args:
            prices: DataFrame with MultiIndex (date, ticker) and ``adj_close``.

        Returns:
            DataFrame with columns [log_return, simple_return] and
            (date, ticker) MultiIndex at month-end frequency.
        """
        if prices.empty:
            return pd.DataFrame(columns=["log_return", "simple_return"])

        adj = prices["adj_close"].unstack("ticker").sort_index()

        # Forward-fill price gaps within month, then take month-end price
        monthly_prices = adj.ffill().resample("ME").last()
        monthly_simple = monthly_prices / monthly_prices.shift(1) - 1
        monthly_log = np.log(monthly_prices / monthly_prices.shift(1))

        monthly_simple = monthly_simple.dropna(how="all")
        monthly_log = monthly_log.dropna(how="all")

        simple_stacked = monthly_simple.stack(future_stack=True).rename("simple_return")
        log_stacked = monthly_log.stack(future_stack=True).rename("log_return")

        result = pd.concat([log_stacked, simple_stacked], axis=1).dropna(how="all")
        result.index.names = ["date", "ticker"]
        self.logger.info(
            "Computed %d monthly return rows (%d months)",
            len(result),
            len(monthly_simple),
        )
        return result

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_against_index(
        self,
        returns_df: pd.DataFrame,
        index_ticker: str = "^GSPC",
        min_correlation: float = 0.3,
    ) -> tuple[bool, str]:
        """Sanity-check: mean cross-sectional return should correlate with SPX.

        We compute the equal-weight mean return across all tickers per date
        and correlate it with the SPX log-return. A correlation below
        ``min_correlation`` suggests a data or computation error.

        Args:
            returns_df: Output of ``compute_daily_log_returns``.
            index_ticker: Benchmark ticker in the same DataFrame (if present).
            min_correlation: Minimum acceptable Pearson correlation.

        Returns:
            (passed: bool, message: str)
        """
        if returns_df.empty:
            return False, "Empty return DataFrame — nothing to validate."

        try:
            ret_wide = returns_df["log_return"].unstack("ticker")

            if index_ticker in ret_wide.columns:
                spx = ret_wide[index_ticker]
                eq_weight = ret_wide.drop(columns=[index_ticker]).mean(axis=1)
            else:
                # If SPX not present, use mean vs first ticker as a loose check
                eq_weight = ret_wide.mean(axis=1)
                spx = ret_wide.iloc[:, 0]

            corr = eq_weight.corr(spx)
            if pd.isna(corr):
                return False, "Correlation is NaN — insufficient overlapping data."
            if corr < min_correlation:
                msg = f"Cross-sectional mean vs benchmark correlation = {corr:.3f} < {min_correlation}. Check data."
                self.logger.warning(msg)
                return False, msg

            msg = f"Validation passed: cross-sectional mean vs benchmark correlation = {corr:.3f}"
            self.logger.info(msg)
            return True, msg

        except Exception as exc:  # noqa: BLE001
            msg = f"Validation error: {exc}"
            self.logger.warning(msg)
            return False, msg

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def store_returns(self, engine: Engine, returns_df: pd.DataFrame, freq: str) -> int:
        """Upsert returns into the ``returns`` table.

        Args:
            engine: SQLAlchemy engine.
            returns_df: DataFrame with (date, ticker) index and
                        [log_return, simple_return] columns.
            freq: ``'daily'`` or ``'monthly'``.

        Returns:
            Number of rows upserted.
        """
        if returns_df.empty:
            self.logger.warning("No %s return rows to store.", freq)
            return 0

        reset = returns_df.reset_index()
        reset["date"] = pd.to_datetime(reset["date"]).dt.date
        reset["freq"] = freq
        rows = reset.to_dict(orient="records")

        upserted = upsert_rows(engine, returns_table, rows)
        self.logger.info("Upserted %d %s return rows", upserted, freq)
        return upserted

    # ------------------------------------------------------------------
    # High-level runner
    # ------------------------------------------------------------------

    def run(self, engine: Engine, prices: pd.DataFrame) -> ReturnSummary:
        """Compute daily + monthly returns, validate, and store in DB.

        Args:
            engine: Active SQLAlchemy engine.
            prices: Raw price DataFrame from the ingestion pipeline.

        Returns:
            ``ReturnSummary`` with row counts and validation status.
        """
        daily = self.compute_daily_log_returns(prices)
        monthly = self.compute_monthly_simple_returns(prices)

        validation_passed, validation_message = self.validate_against_index(daily)

        daily_rows = self.store_returns(engine, daily, freq="daily")
        monthly_rows = self.store_returns(engine, monthly, freq="monthly")

        return ReturnSummary(
            daily_rows_stored=daily_rows,
            monthly_rows_stored=monthly_rows,
            validation_passed=validation_passed,
            validation_message=validation_message,
        )
