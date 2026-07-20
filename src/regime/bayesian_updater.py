from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BayesianUpdater:
    """Computes Bayesian posterior IC, variance, and effective sample size using prior and exponentially weighted observations."""

    def __init__(
        self,
        prior_weight: float = 0.3,
        decay_halflife: float = 12.0,
        default_prior_ic: float = 0.02,
        default_prior_var: float = 0.01,
    ) -> None:
        self.prior_weight = prior_weight
        self.decay_halflife = decay_halflife
        self.default_prior_ic = default_prior_ic
        self.default_prior_var = default_prior_var

    def update_posteriors(
        self,
        rolling_df: pd.DataFrame,
        ic_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Computes Bayesian posterior IC estimates for each date, factor, and regime.

        Args:
            rolling_df: DataFrame with date, factor_name, regime_id, regime_label, sample_ic, rolling_icir, sample_count.
            ic_df: DataFrame of raw cross-sectional ICs indexed by date.

        Returns:
            DataFrame with posterior_ic, posterior_variance, effective_sample_size added.
        """
        if rolling_df.empty:
            return rolling_df

        df = rolling_df.copy()

        # Compute global prior IC per factor from full history or default
        global_priors = {}
        global_vars = {}
        for factor in ic_df.columns:
            series = ic_df[factor].dropna()
            if len(series) > 0:
                global_priors[factor] = float(series.mean())
                global_vars[factor] = float(series.var()) if len(series) > 1 else self.default_prior_var
            else:
                global_priors[factor] = self.default_prior_ic
                global_vars[factor] = self.default_prior_var

        posterior_ics = []
        posterior_vars = []
        eff_sample_sizes = []

        for _, row in df.iterrows():
            factor = row["factor_name"]
            sample_ic = row["sample_ic"]
            count = row["sample_count"]

            prior_ic = global_priors.get(factor, self.default_prior_ic)
            prior_var = max(global_vars.get(factor, self.default_prior_var), 1e-5)

            if pd.isna(sample_ic) or count <= 0:
                post_ic = prior_ic
                post_var = prior_var
                n_eff = 0.0
            else:
                # Exponential decay weight adjustment
                # Assuming exponential decay halflife reduces effective weight over older samples
                lambda_decay = np.log(2.0) / max(self.decay_halflife, 1.0)
                # Effective sample size approximation with decay
                decay_sum = (1.0 - np.exp(-lambda_decay * count)) / (1.0 - np.exp(-lambda_decay)) if lambda_decay > 1e-6 else float(count)
                n_eff = float(decay_sum)

                # Bayesian shrinkage convex combination
                alpha = self.prior_weight
                post_ic = float(alpha * prior_ic + (1.0 - alpha) * sample_ic)

                # Posterior variance update: 1 / (1/sigma_prior^2 + n_eff / sigma_sample^2)
                sample_var = prior_var  # Use prior variance as robust observation variance proxy
                inv_post_var = (1.0 / prior_var) + (n_eff / max(sample_var, 1e-5))
                post_var = float(1.0 / inv_post_var)

            posterior_ics.append(post_ic)
            posterior_vars.append(post_var)
            eff_sample_sizes.append(n_eff)

        df["posterior_ic"] = posterior_ics
        df["posterior_variance"] = posterior_vars
        df["effective_sample_size"] = eff_sample_sizes

        return df
