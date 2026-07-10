"""Fama-MacBeth OLS Composite: cross-sectional regressions averaged over time."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.combination.base import BaseComposite


class FamaMacBethComposite(BaseComposite):
    """Fama-MacBeth two-pass regression composite.

    **Pass 1 (cross-sectional)**: At each evaluation date *t*, run an OLS
    regression of forward returns on the cross-section of factor z-scores:

        return_{t+1,i} = β₀_t + β₁_t · f₁_{t,i} + ... + β_K_t · f_K_{t,i} + ε

    **Pass 2 (time-series)**: Average the β coefficients across the trailing
    window to get stable factor premium estimates.

    The composite score at date *t* is then the weighted sum of factor
    z-scores using the averaged β̄ coefficients as weights.
    """

    method_name: str = "fama_macbeth"

    def __init__(self, logger: logging.Logger | None = None) -> None:
        super().__init__(logger=logger)
        self._coeff_history: List[dict] = []
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
        """Run cross-sectional regressions on historical dates, then average coefficients.

        Only data strictly before ``eval_date`` (with forward returns available)
        is used to compute the time-series average of β coefficients.
        """
        lookback = config.get("ic_lookback_months", 36)

        # Dates available for regression (need both scores and forward returns)
        available_dates = sorted(factor_scores.index.get_level_values("date").unique())
        historical_dates = [d for d in available_dates if d <= eval_date]
        window_dates = historical_dates[-lookback:]

        factor_names = factor_scores.columns.tolist()
        all_betas: Dict[str, List[float]] = {f: [] for f in factor_names}

        for d in window_dates:
            try:
                X_d = factor_scores.loc[d].dropna(how="any")
                y_d = forward_returns.loc[d].reindex(X_d.index).dropna()
                common = X_d.index.intersection(y_d.index)

                if len(common) < len(factor_names) + 2:
                    continue

                X_reg = sm.add_constant(X_d.loc[common])
                y_reg = y_d.loc[common]

                model = sm.OLS(y_reg, X_reg)
                results = model.fit()

                for i, f in enumerate(factor_names):
                    beta = float(results.params.iloc[i + 1])  # skip constant
                    all_betas[f].append(beta)

                # Store individual date coefficients for diagnostics
                coeff_record = {"date": d}
                for i, f in enumerate(factor_names):
                    coeff_record[f] = float(results.params.iloc[i + 1])
                self._coeff_history.append(coeff_record)

            except Exception as exc:
                self.logger.warning("FM regression failed on %s: %s", d.date(), exc)
                continue

        # Time-series average of β coefficients (Pass 2)
        avg_betas = {}
        for f, betas in all_betas.items():
            avg_betas[f] = float(np.mean(betas)) if betas else 0.0

        # Normalise to sum of absolute values = 1
        total_abs = sum(abs(v) for v in avg_betas.values())
        if total_abs > 0:
            self._current_weights = {f: v / total_abs for f, v in avg_betas.items()}
        else:
            n = len(factor_names)
            self._current_weights = {f: 1.0 / n for f in factor_names}

        # Record weight history
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
        """Compute the FM composite score for each ticker at *eval_date*."""
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

    def get_coefficients_dataframe(self) -> pd.DataFrame:
        """Return the rolling Fama-MacBeth coefficients for diagnostics."""
        if not self._coeff_history:
            return pd.DataFrame()
        return pd.DataFrame(self._coeff_history)
