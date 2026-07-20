from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


class RegimeFactorAnalyzer:
    """Computes regime-conditional Information Coefficients (IC) and rolling ICIR statistics."""

    def __init__(
        self,
        ic_lookback_months: int = 36,
        min_regime_samples: int = 5,
    ) -> None:
        self.ic_lookback_months = ic_lookback_months
        self.min_regime_samples = min_regime_samples

    def compute_daily_cross_sectional_ic(
        self,
        factors_df: pd.DataFrame,
        returns_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Computes cross-sectional Spearman rank IC for each date and factor.

        Args:
            factors_df: MultiIndex (date, ticker) or wide DataFrame of factor scores.
            returns_df: MultiIndex (date, ticker) or wide DataFrame of forward returns.

        Returns:
            DataFrame indexed by date, with columns as factor names containing IC values.
        """
        if isinstance(factors_df.index, pd.MultiIndex):
            factors_flat = factors_df.reset_index()
        else:
            factors_flat = factors_df.copy()

        if isinstance(returns_df.index, pd.MultiIndex):
            returns_flat = returns_df.reset_index()
        else:
            returns_flat = returns_df.copy()

        if "factor_name" in factors_flat.columns and "final_score" in factors_flat.columns:
            pivoted_factors = factors_flat.pivot(index=["date", "ticker"], columns="factor_name", values="final_score")
        else:
            pivoted_factors = factors_df.copy()

        ret_col = "simple_return" if "simple_return" in returns_flat.columns else ("return" if "return" in returns_flat.columns else returns_flat.columns[-1])
        if "date" in returns_flat.columns and "ticker" in returns_flat.columns:
            returns_series = returns_flat.set_index(["date", "ticker"])[ret_col]
        else:
            returns_series = returns_df.stack()

        aligned = pivoted_factors.join(returns_series.rename("fwd_return"), how="inner").dropna(subset=["fwd_return"])

        dates = sorted(aligned.index.get_level_values("date").unique())
        factor_cols = [c for c in pivoted_factors.columns if c != "fwd_return"]

        ic_records = []
        for d in dates:
            day_data = aligned.xs(d, level="date")
            if len(day_data) < 2:
                continue
            fwd_ret = day_data["fwd_return"]
            row_dict = {"date": d}
            for factor in factor_cols:
                scores = day_data[factor].dropna()
                common_idx = scores.index.intersection(fwd_ret.index)
                if len(common_idx) >= 2:
                    corr, _ = spearmanr(scores.loc[common_idx], fwd_ret.loc[common_idx])
                    row_dict[factor] = corr if not np.isnan(corr) else 0.0
                else:
                    row_dict[factor] = np.nan
            ic_records.append(row_dict)

        if not ic_records:
            return pd.DataFrame(columns=factor_cols, index=pd.DatetimeIndex([], name="date"))

        ic_df = pd.DataFrame(ic_records).set_index("date").sort_index()
        return ic_df


    def compute_regime_conditional_stats(
        self,
        ic_df: pd.DataFrame,
        regimes_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Computes aggregate factor x regime IC matrix and rolling ICIR.

        Args:
            ic_df: DataFrame of daily/monthly ICs indexed by date, columns are factors.
            regimes_df: DataFrame indexed by date with 'regime_id', 'regime_label'.

        Returns:
            Tuple of (regime_ic_matrix, rolling_stats_df)
        """
        merged = ic_df.join(regimes_df[["regime_id", "regime_label"]], how="inner")

        regime_labels = sorted(merged["regime_label"].dropna().unique())
        factors = [c for c in ic_df.columns if c in merged.columns]

        summary_rows = []
        for factor in factors:
            row = {"factor_name": factor}
            for label in regime_labels:
                sub = merged[merged["regime_label"] == label][factor].dropna()
                row[label] = float(sub.mean()) if len(sub) >= self.min_regime_samples else np.nan
            summary_rows.append(row)

        summary_matrix = pd.DataFrame(summary_rows).set_index("factor_name")

        dates = sorted(merged.index.unique())
        rolling_records = []

        for i, current_date in enumerate(dates):
            start_date = current_date - pd.DateOffset(months=self.ic_lookback_months)
            window_df = merged.loc[(merged.index >= start_date) & (merged.index <= current_date)]

            for factor in factors:
                for reg_id in sorted(merged["regime_id"].dropna().unique()):
                    reg_sub = window_df[window_df["regime_id"] == reg_id][factor].dropna()
                    reg_label_matches = merged.loc[merged["regime_id"] == reg_id, "regime_label"]
                    reg_label = str(reg_label_matches.iloc[0]) if not reg_label_matches.empty else f"Regime_{reg_id}"

                    count = len(reg_sub)
                    if count >= self.min_regime_samples:
                        sample_ic = float(reg_sub.mean())
                        std_ic = float(reg_sub.std())
                        rolling_icir = float(sample_ic / std_ic) if std_ic > 1e-6 else 0.0
                    else:
                        sample_ic = np.nan
                        rolling_icir = np.nan

                    rolling_records.append({
                        "date": current_date,
                        "factor_name": factor,
                        "regime_id": int(reg_id),
                        "regime_label": reg_label,
                        "sample_ic": sample_ic,
                        "rolling_icir": rolling_icir,
                        "sample_count": count,
                    })

        rolling_df = pd.DataFrame(rolling_records)
        return summary_matrix, rolling_df
