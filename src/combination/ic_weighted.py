"""IC-Weighted Composite: factor weights proportional to trailing rolling IC."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.combination.base import BaseComposite


class ICWeightedComposite(BaseComposite):
    """Combine factor z-scores using trailing Spearman rank IC as weights.

    At each evaluation date *t*, the rolling IC of each factor is computed over
    the most recent ``ic_lookback_months`` periods.  Factors with negative
    rolling IC are zeroed out, and the remaining weights are normalised to
    sum to 1.  The composite score is a weighted average of the factor
    z-scores.
    """

    method_name: str = "ic_weighted"

    def __init__(self, logger: logging.Logger | None = None) -> None:
        super().__init__(logger=logger)
        # Stores factor weights at each evaluation date
        self._weights_history: List[dict] = []

    # ------------------------------------------------------------------
    # Interface implementation
    # ------------------------------------------------------------------

    def fit(
        self,
        factor_scores: pd.DataFrame,
        forward_returns: pd.Series,
        eval_date: pd.Timestamp,
        config: dict,
    ) -> None:
        """Compute rolling IC weights for each factor up to *eval_date*.

        The computed weights are stored internally and used by ``predict``.
        """
        lookback = config.get("ic_lookback_months", 36)

        # Factor scores: MultiIndex (date, ticker) → factor columns
        available_dates = sorted(factor_scores.index.get_level_values("date").unique())
        # Only use dates up to and including eval_date
        historical_dates = [d for d in available_dates if d <= eval_date]

        # Use the most recent `lookback` dates
        window_dates = historical_dates[-lookback:]
        if len(window_dates) < 3:
            self.logger.warning(
                "Insufficient dates (%d) for IC computation at %s",
                len(window_dates), eval_date.date(),
            )
            self._current_weights = {}
            return

        factor_names = factor_scores.columns.tolist()
        ic_means: Dict[str, float] = {}

        for factor in factor_names:
            ic_values = []
            for d in window_dates:
                try:
                    scores_d = factor_scores.loc[d, factor].dropna()
                    rets_d = forward_returns.loc[d].reindex(scores_d.index).dropna()
                    common = scores_d.index.intersection(rets_d.index)
                    if len(common) >= 5:
                        ic = scores_d.loc[common].corr(rets_d.loc[common], method="spearman")
                        if not pd.isna(ic):
                            ic_values.append(ic)
                except (KeyError, TypeError):
                    continue

            ic_means[factor] = float(np.mean(ic_values)) if ic_values else 0.0

        # Zero out negative ICs, then normalise
        raw_weights = {f: max(0.0, ic) for f, ic in ic_means.items()}
        total = sum(raw_weights.values())
        if total > 0:
            self._current_weights = {f: w / total for f, w in raw_weights.items()}
        else:
            # Fall back to equal weight if all ICs are negative
            n = len(factor_names)
            self._current_weights = {f: 1.0 / n for f in factor_names}

        # Record weight history for persistence
        for factor, weight in self._current_weights.items():
            self._weights_history.append({
                "date": eval_date,
                "factor_name": factor,
                "weight": weight,
            })

    def predict(
        self,
        factor_scores: pd.DataFrame,
        eval_date: pd.Timestamp,
    ) -> pd.Series:
        """Compute the IC-weighted composite score for each ticker at *eval_date*."""
        if not hasattr(self, "_current_weights") or not self._current_weights:
            return pd.Series(dtype=float)

        try:
            cross_section = factor_scores.loc[eval_date]
        except KeyError:
            return pd.Series(dtype=float)

        composite = pd.Series(0.0, index=cross_section.index)
        for factor, weight in self._current_weights.items():
            if factor in cross_section.columns:
                composite += weight * cross_section[factor].fillna(0.0)

        composite.name = "composite_score"
        return composite

    def get_weights_dataframe(self) -> pd.DataFrame:
        """Return the full weight history as a DataFrame."""
        if not self._weights_history:
            return pd.DataFrame(columns=["date", "factor_name", "weight"])
        return pd.DataFrame(self._weights_history)
