from __future__ import annotations

import numpy as np
import pandas as pd


class PortfolioConstructor:
    """Constructs target portfolio weights from factor scores subject to exposure constraints."""

    def __init__(self, logger=None) -> None:
        self.logger = logger

    def construct_weights(
        self,
        factor_scores: pd.Series,
        method: str = "equal_weight_long_only",
        max_position_size: float = 0.05,
        long_only: bool = True,
        net_exposure: float = 1.0,
        gross_exposure: float = 1.0,
    ) -> pd.Series:
        """Calculate target portfolio weights on a single date.

        Args:
            factor_scores: pd.Series mapping tickers to factor scores.
            method: Weighting method name.
            max_position_size: Maximum absolute weight allowed for any single stock.
            long_only: If True, only positive weights are allowed.
            net_exposure: Target sum of weights.
            gross_exposure: Target sum of absolute weights.

        Returns:
            pd.Series of weights indexed by ticker (non-selected have weight 0.0).
        """
        clean_scores = factor_scores.dropna()
        if clean_scores.empty:
            return pd.Series(dtype=float)

        n_assets = len(clean_scores)
        
        # Decide number of assets to select in each leg (top 20% / bottom 20% by default)
        # If too few assets, choose at least 1
        k = max(1, int(np.ceil(n_assets * 0.20)))

        sorted_scores = clean_scores.sort_values(ascending=False)

        weights = pd.Series(0.0, index=factor_scores.index)

        if method == "equal_weight_long_only" or long_only:
            # Select top K assets
            long_tickers = sorted_scores.index[:k]
            if len(long_tickers) > 0:
                weights.loc[long_tickers] = net_exposure / len(long_tickers)
            # Enforce max position size on long leg
            weights = self.limit_position_sizes(weights, max_position_size)

        elif method == "equal_weight_long_short":
            # Select top K assets for long leg, bottom K assets for short leg
            long_tickers = sorted_scores.index[:k]
            short_tickers = sorted_scores.index[-k:]

            # Exclude overlapping tickers if K is large
            overlap = set(long_tickers).intersection(set(short_tickers))
            if overlap:
                long_tickers = [t for t in long_tickers if t not in overlap]
                short_tickers = [t for t in short_tickers if t not in overlap]

            n_long = len(long_tickers)
            n_short = len(short_tickers)

            # Determine leg exposure values: L - S = net_exposure, L + S = gross_exposure
            # Under long_only=False, we allow short leg (S > 0)
            l_target = 0.5 * (gross_exposure + net_exposure)
            s_target = 0.5 * (gross_exposure - net_exposure)

            if n_long > 0:
                weights.loc[long_tickers] = l_target / n_long
            if n_short > 0:
                weights.loc[short_tickers] = -s_target / n_short

            # Enforce max position size on both legs separately
            weights = self.limit_position_sizes(weights, max_position_size)

        else:
            raise ValueError(f"Unknown weighting method: {method}")

        return weights

    def limit_position_sizes(self, weights: pd.Series, max_size: float) -> pd.Series:
        """Clip weights to max_size while maintaining the leg exposure sums."""
        if max_size >= 1.0 or weights.empty:
            return weights

        longs = weights[weights > 0]
        if not longs.empty:
            longs = self._redistribute_leg(longs, max_size, longs.sum())

        shorts = weights[weights < 0]
        if not shorts.empty:
            shorts_abs = -shorts
            shorts_clipped = self._redistribute_leg(shorts_abs, max_size, shorts_abs.sum())
            shorts = -shorts_clipped

        # Re-combine longs and shorts and fill unselected with 0.0
        combined = pd.concat([longs, shorts])
        return combined.reindex(weights.index, fill_value=0.0)

    def _redistribute_leg(self, leg: pd.Series, max_size: float, target_sum: float) -> pd.Series:
        """Iterative redistribution of excess weight from clipped assets."""
        s = leg.copy()
        
        # If target_sum is 0, just return zeros
        if np.isclose(target_sum, 0.0):
            return pd.Series(0.0, index=leg.index)

        # Cap max_size dynamically to ensure mathematical feasibility:
        # e.g., if there are N assets, max_size cannot be less than target_sum / N
        min_possible_max = target_sum / len(leg)
        eff_max_size = max(max_size, min_possible_max)

        for _ in range(20):
            over = s > eff_max_size
            if not over.any():
                break
            
            s[over] = eff_max_size
            under = s < eff_max_size
            if not under.any():
                # All assets are capped, cannot redistribute further
                break

            remaining_target = target_sum - s[over].sum()
            current_under_sum = s[under].sum()

            if current_under_sum > 1e-12:
                s[under] = s[under] * (remaining_target / current_under_sum)
            else:
                # Distribute equally among under-allocated assets if their sum is 0
                s[under] = remaining_target / under.sum()
                
        return s
