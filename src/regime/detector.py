from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM

from src.data.db import market_regimes_table, upsert_rows

logger = logging.getLogger(__name__)





class MarketRegimeDetector:
    """Hidden Markov Model detector for financial market regimes.

    Constructs macro/market features, fits GaussianHMM, evaluates optimal state count via BIC,
    assigns economic regime labels, and computes regime persistence.
    """

    def __init__(
        self,
        n_components: int = 4,
        covariance_type: str = "full",
        n_iter: int = 100,
        random_state: int = 42,
    ):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state

        self.model: Optional[GaussianHMM] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = []
        self.regime_labels: Dict[int, str] = {}

    def build_feature_matrix(
        self,
        prices_df: pd.DataFrame,
        fred_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Construct macro/market feature panel for regime detection.

        Features:
          1. vix_level: VIXCLS from FRED (or proxy from returns)
          2. vix_change: 1-day diff of VIX
          3. yield_curve_slope: DGS10 - DGS2
          4. credit_spread: BAMLH0A0HYM2
          5. spx_momentum: 252-day price return of market benchmark
          6. realized_vol: 21-day annualized daily volatility
          7. gdp_growth: 365-day percentage change of GDP (or trailing 252-day)

        Args:
            prices_df: DataFrame of daily stock prices with columns ['date', 'ticker', 'close'] or wide format
            fred_df: DataFrame of FRED economic series indexed by date (or with 'date' column)

        Returns:
            DataFrame of aligned, clean daily feature series.
        """
        # 1. Market Return & Volatility from prices_df
        if "ticker" in prices_df.columns and "close" in prices_df.columns:
            # Pivot to wide format
            wide_prices = prices_df.pivot(index="date", columns="ticker", values="close")
            market_price = wide_prices.median(axis=1)
        elif isinstance(prices_df.index, pd.DatetimeIndex) or "date" in prices_df.columns:
            if "date" in prices_df.columns:
                prices_df = prices_df.set_index("date")
            market_price = prices_df.select_dtypes(include=[np.number]).median(axis=1)
        else:
            raise ValueError("prices_df must contain date and price data")

        market_price = market_price.sort_index().ffill().bfill()
        daily_returns = market_price.pct_change()

        # Feature: Realized Volatility (21-day rolling, annualized)
        realized_vol = daily_returns.rolling(21, min_periods=5).std() * np.sqrt(252)

        # Feature: SPX / Market 12M Momentum (252-day return)
        spx_momentum = market_price.pct_change(252)

        features = pd.DataFrame({
            "spx_momentum": spx_momentum,
            "realized_vol": realized_vol,
        }, index=market_price.index)

        # 2. Merge FRED macro features if available
        if fred_df is not None and not fred_df.empty:
            fred = fred_df.copy()
            if "date" in fred.columns:
                fred = fred.set_index("date")
            fred.index = pd.to_datetime(fred.index)
            features.index = pd.to_datetime(features.index)

            # VIX level & change
            if "VIXCLS" in fred.columns:
                features["vix_level"] = fred["VIXCLS"].reindex(features.index).ffill().bfill()
            else:
                # Proxy VIX level as realized_vol * 100
                features["vix_level"] = realized_vol * 100

            features["vix_change"] = features["vix_level"].diff()

            # Yield curve slope (10Y - 2Y)
            if "DGS10" in fred.columns and "DGS2" in fred.columns:
                dgs10 = fred["DGS10"].reindex(features.index).ffill().bfill()
                dgs2 = fred["DGS2"].reindex(features.index).ffill().bfill()
                features["yield_curve_slope"] = dgs10 - dgs2
            else:
                features["yield_curve_slope"] = 1.5  # Neutral default proxy

            # Credit spread (HY - IG)
            if "BAMLH0A0HYM2" in fred.columns:
                features["credit_spread"] = fred["BAMLH0A0HYM2"].reindex(features.index).ffill().bfill()
            else:
                features["credit_spread"] = 4.0  # Default proxy

            # GDP YoY Growth
            if "GDP" in fred.columns:
                gdp = fred["GDP"].reindex(features.index).ffill().bfill()
                features["gdp_growth"] = gdp.pct_change(252).fillna(0.02)
            else:
                features["gdp_growth"] = 0.02
        else:
            # Fallback if no FRED data provided
            features["vix_level"] = realized_vol * 100
            features["vix_change"] = features["vix_level"].diff()
            features["yield_curve_slope"] = 1.5
            features["credit_spread"] = 4.0
            features["gdp_growth"] = 0.02

        # Clean NaNs resulting from rolling windows
        features = features.ffill().bfill().dropna()
        self.feature_names = list(features.columns)
        return features

    def fit(self, feature_df: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        """Fit GaussianHMM on normalized feature matrix.

        Returns:
            Tuple of (hard_state_series, soft_probabilities_dataframe)
        """
        if feature_df.empty:
            raise ValueError("Feature matrix is empty.")

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(feature_df.values)

        self.model = GaussianHMM(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        self.model.fit(X_scaled)

        hard_states = self.model.predict(X_scaled)
        soft_probs = self.model.predict_proba(X_scaled)

        state_series = pd.Series(hard_states, index=feature_df.index, name="regime_id")

        prob_cols = [f"prob_{i}" for i in range(self.n_components)]
        prob_df = pd.DataFrame(soft_probs, index=feature_df.index, columns=prob_cols)

        # Assign economic regime labels
        self.regime_labels = self.label_regimes(feature_df, state_series)

        logger.info(f"Fitted GaussianHMM ({self.n_components} states). Labels: {self.regime_labels}")
        return state_series, prob_df

    def select_optimal_states(
        self, feature_df: pd.DataFrame, max_states: int = 6
    ) -> Dict[str, Any]:
        """Perform BIC model selection over state counts [2, max_states].

        BIC = -2 * log_likelihood + p * log(N)
        """
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(feature_df.values)
        N, d = X_scaled.shape

        bic_scores = {}
        models = {}

        for k in range(2, max_states + 1):
            model = GaussianHMM(
                n_components=k,
                covariance_type=self.covariance_type,
                n_iter=self.n_iter,
                random_state=self.random_state,
            )
            model.fit(X_scaled)
            log_likelihood = model.score(X_scaled)

            # Number of parameters p
            if self.covariance_type == "full":
                cov_params = k * d * (d + 1) / 2
            elif self.covariance_type == "diag":
                cov_params = k * d
            else:
                cov_params = k

            trans_params = k * (k - 1)
            mean_params = k * d
            init_params = k - 1
            p = cov_params + trans_params + mean_params + init_params

            bic = -2 * log_likelihood + p * np.log(N)
            bic_scores[k] = float(bic)
            models[k] = model

        optimal_k = min(bic_scores, key=bic_scores.get)
        logger.info(f"BIC selection results: {bic_scores}. Optimal states k={optimal_k}")

        return {
            "bic_scores": bic_scores,
            "optimal_k": optimal_k,
        }

    def label_regimes(
        self, feature_df: pd.DataFrame, state_series: pd.Series
    ) -> Dict[int, str]:
        """Auto-assign intuitive economic labels to HMM states based on cluster means."""
        k_states = self.n_components
        labels = {}

        means = {}
        for k in range(k_states):
            mask = state_series == k
            if mask.sum() == 0:
                means[k] = {col: 0.0 for col in feature_df.columns}
            else:
                means[k] = feature_df.loc[mask].mean().to_dict()

        if k_states != 4:
            for k in range(k_states):
                labels[k] = f"Regime {k}"
            return labels

        # Specific heuristic for 4-state model
        # 1. Bear / High-Vol: highest realized_vol or highest vix_level
        sorted_by_vol = sorted(range(4), key=lambda k: means[k].get("realized_vol", 0.0), reverse=True)
        bear_state = sorted_by_vol[0]
        labels[bear_state] = "Bear / High-Vol"

        remaining = [k for k in range(4) if k != bear_state]

        # 2. Bull Market: highest spx_momentum among remaining
        sorted_by_mom = sorted(remaining, key=lambda k: means[k].get("spx_momentum", -999.0), reverse=True)
        bull_state = sorted_by_mom[0]
        labels[bull_state] = "Bull Market"

        remaining = [k for k in remaining if k != bull_state]

        # 3. Rate Shock / Stagnation: lowest yield_curve_slope or lowest gdp_growth
        sorted_by_yield = sorted(remaining, key=lambda k: means[k].get("yield_curve_slope", 999.0))
        rate_shock_state = sorted_by_yield[0]
        labels[rate_shock_state] = "Rate Shock / Stagnation"

        remaining = [k for k in remaining if k != rate_shock_state]
        neutral_state = remaining[0]
        labels[neutral_state] = "Neutral / Transition"

        return labels

    def compute_transition_matrix(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Compute state transition probability matrix and expected regime durations."""
        if self.model is None:
            raise ValueError("Model has not been fitted.")

        trans_mat = self.model.transmat_
        labels = [self.regime_labels.get(i, f"State {i}") for i in range(self.n_components)]

        trans_df = pd.DataFrame(trans_mat, index=labels, columns=labels)

        # Duration = 1 / (1 - P_ii)
        diag = np.diag(trans_mat)
        durations = []
        for p_ii in diag:
            dur = 1.0 / (1.0 - p_ii) if p_ii < 1.0 else np.inf
            durations.append(dur)

        duration_series = pd.Series(durations, index=labels, name="avg_duration_days")
        return trans_df, duration_series

    def save_to_db(self, engine: Any, regime_df: pd.DataFrame) -> int:
        """Upsert regime dataframe into PostgreSQL database.

        Expected columns in regime_df: ['date', 'regime_id', 'regime_label', 'prob_0', 'prob_1', 'prob_2', 'prob_3']
        """
        rows = []
        for idx, row in regime_df.reset_index().iterrows():
            d = row["date"]
            if isinstance(d, pd.Timestamp):
                d = d.date()

            record = {
                "date": d,
                "regime_id": int(row["regime_id"]),
                "regime_label": str(row.get("regime_label", f"State {row['regime_id']}")),
                "prob_0": float(row.get("prob_0", 0.0)),
                "prob_1": float(row.get("prob_1", 0.0)),
                "prob_2": float(row.get("prob_2", 0.0)),
                "prob_3": float(row.get("prob_3", 0.0)),
            }
            rows.append(record)

        count = upsert_rows(engine, market_regimes_table, rows)
        logger.info(f"Upserted {count} regime records into market_regimes table.")
        return count
