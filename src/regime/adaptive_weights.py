from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AdaptiveWeightGenerator:
    """Generates dynamic factor weights with direction sign flip, clipping constraints, and soft regime blending."""

    def __init__(
        self,
        max_factor_weight: float = 0.25,
        min_weight_threshold: float = 0.02,
    ) -> None:
        self.max_factor_weight = max_factor_weight
        self.min_weight_threshold = min_weight_threshold

    def generate_weights(
        self,
        posterior_df: pd.DataFrame,
        regimes_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generates dynamic factor weights and directions.

        Args:
            posterior_df: DataFrame with date, factor_name, regime_id, posterior_ic.
            regimes_df: DataFrame with date, prob_0, prob_1, prob_2, prob_3.

        Returns:
            Tuple of (weights_df, direction_df, metadata_df)
        """
        if posterior_df.empty or regimes_df.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        dates = sorted(posterior_df["date"].unique())
        factors = sorted(posterior_df["factor_name"].unique())
        regime_ids = sorted(posterior_df["regime_id"].unique())

        prob_cols = [f"prob_{k}" for k in regime_ids if f"prob_{k}" in regimes_df.columns]

        weights_records = []
        direction_records = []
        metadata_records = []

        for d in dates:
            d_post = posterior_df[posterior_df["date"] == d]
            d_reg = regimes_df.loc[regimes_df.index == d] if d in regimes_df.index else regimes_df[regimes_df["date"] == d]

            if d_reg.empty:
                # Fallback to equal probabilities across regimes
                reg_probs = {k: 1.0 / len(regime_ids) for k in regime_ids}
            else:
                row_reg = d_reg.iloc[0]
                reg_probs = {}
                for k in regime_ids:
                    col = f"prob_{k}"
                    reg_probs[k] = float(row_reg[col]) if col in row_reg and not pd.isna(row_reg[col]) else 0.0

            # 1. Compute soft-probability blended posterior IC per factor
            blended_ics = {}
            for factor in factors:
                f_post = d_post[d_post["factor_name"] == factor]
                b_ic = 0.0
                for k in regime_ids:
                    r_ic = f_post[f_post["regime_id"] == k]["posterior_ic"]
                    ic_val = float(r_ic.iloc[0]) if not r_ic.empty and not pd.isna(r_ic.iloc[0]) else 0.0
                    b_ic += reg_probs.get(k, 0.0) * ic_val
                blended_ics[factor] = b_ic

            # 2. Extract direction and raw weight magnitude
            raw_weights = {}
            directions = {}
            for factor in factors:
                ic_val = blended_ics[factor]
                raw_weights[factor] = abs(ic_val)
                directions[factor] = 1.0 if ic_val >= 0 else -1.0

            # 3. Apply Weight Constraint Layer
            # Filter threshold
            constrained_weights = {}
            for factor, w in raw_weights.items():
                if w < self.min_weight_threshold:
                    constrained_weights[factor] = 0.0
                else:
                    constrained_weights[factor] = min(w, self.max_factor_weight)

            # Normalize to sum to 1.0
            tot_w = sum(constrained_weights.values())
            final_weights = {}
            if tot_w > 1e-6:
                for factor, w in constrained_weights.items():
                    final_weights[factor] = w / tot_w
            else:
                # Equal weight fallback
                eq_w = 1.0 / len(factors)
                for factor in factors:
                    final_weights[factor] = eq_w

            w_row = {"date": d, **final_weights}
            d_row = {"date": d, **directions}
            weights_records.append(w_row)
            direction_records.append(d_row)

            # Metadata for persistence
            top_reg_id = max(reg_probs, key=reg_probs.get) if reg_probs else 0
            top_prob = reg_probs.get(top_reg_id, 0.0)
            for factor in factors:
                metadata_records.append({
                    "date": d,
                    "factor_name": factor,
                    "regime_probability": top_prob,
                    "dynamic_weight": final_weights[factor],
                    "posterior_ic": blended_ics[factor],
                })

        weights_df = pd.DataFrame(weights_records).set_index("date").sort_index()
        direction_df = pd.DataFrame(direction_records).set_index("date").sort_index()
        metadata_df = pd.DataFrame(metadata_records)

        return weights_df, direction_df, metadata_df
