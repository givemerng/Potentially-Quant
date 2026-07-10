"""Shared feature preparation pipeline for all combination methods.

Ensures all three composite strategies consume identical, consistently
formatted inputs: aligned dates, consistent factor ordering, handled
missing values.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


class FeaturePreprocessor:
    """Prepares a standardised (dates × tickers × factors) feature matrix
    and an aligned forward-return series from raw DB extracts.
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def prepare(
        self,
        factor_scores_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        score_column: str = "final_score",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, List[str]]:
        """Build aligned feature matrix and forward-return series.

        Args:
            factor_scores_df: Long-format factor scores with columns
                              ``[date, ticker, factor_name, <score_column>]``.
            returns_df: Long-format monthly returns with columns
                        ``[date, ticker, simple_return]``.
            score_column: Which score column to pivot (default ``final_score``).

        Returns:
            feature_panel: MultiIndex DataFrame ``(date, ticker) → factor columns``.
                           Each row = one stock on one date; each column = one factor.
            forward_returns_panel: Series ``(date, ticker) → simple_return_{t+1}``.
            eval_dates: Sorted array of all evaluation dates.
            factor_names: Ordered list of factor column names.
        """
        self.logger.info("Preparing feature matrix from factor scores (column=%s)", score_column)

        if factor_scores_df.empty:
            raise ValueError("factor_scores_df is empty")
        if returns_df.empty:
            raise ValueError("returns_df is empty")

        # --- Ensure date types are consistent ---
        scores = factor_scores_df.copy()
        scores["date"] = pd.to_datetime(scores["date"])

        rets = returns_df.copy()
        rets["date"] = pd.to_datetime(rets["date"])

        # --- Determine consistent factor ordering ---
        factor_names = sorted(scores["factor_name"].unique().tolist())
        self.logger.info("Detected %d factors: %s", len(factor_names), factor_names)

        # --- Pivot scores to wide: (date, ticker) → factor columns ---
        pivoted = scores.pivot_table(
            index=["date", "ticker"],
            columns="factor_name",
            values=score_column,
            aggfunc="first",
        )
        # Enforce consistent column ordering
        pivoted = pivoted.reindex(columns=factor_names)

        # --- Build forward returns: return at t+1 aligned with factor at t ---
        # Create a wide return matrix (dates × tickers)
        ret_wide = rets.pivot_table(
            index="date",
            columns="ticker",
            values="simple_return",
            aggfunc="first",
        ).sort_index()

        # Shift returns back by 1 period → forward_return[t] = actual_return[t+1]
        forward_ret_wide = ret_wide.shift(-1)

        # Melt back to long format (date, ticker) → forward_return
        forward_ret_long = forward_ret_wide.stack(future_stack=True)
        forward_ret_long.name = "forward_return"
        forward_ret_long.index.names = ["date", "ticker"]

        # --- Align: keep only (date, ticker) pairs present in BOTH ---
        common_idx = pivoted.index.intersection(forward_ret_long.index)
        feature_panel = pivoted.loc[common_idx].copy()
        forward_returns_panel = forward_ret_long.loc[common_idx].copy()

        # --- Handle missing factor values: fill NaN with cross-sectional median ---
        for col in factor_names:
            if feature_panel[col].isna().any():
                medians = feature_panel.groupby(level="date")[col].transform("median")
                feature_panel[col] = feature_panel[col].fillna(medians)

        # Drop any remaining rows with NaN in any factor column
        valid_mask = feature_panel.notna().all(axis=1) & forward_returns_panel.notna()
        feature_panel = feature_panel.loc[valid_mask]
        forward_returns_panel = forward_returns_panel.loc[valid_mask]

        # --- Sorted evaluation dates ---
        eval_dates = sorted(feature_panel.index.get_level_values("date").unique())

        self.logger.info(
            "Feature matrix ready: %d (date, ticker) observations across %d dates and %d factors",
            len(feature_panel),
            len(eval_dates),
            len(factor_names),
        )

        return feature_panel, forward_returns_panel, eval_dates, factor_names
