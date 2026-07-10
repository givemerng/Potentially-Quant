"""XGBoost Composite: walk-forward gradient-boosted tree model."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover – optional at import time
    xgb = None  # type: ignore[assignment]

from src.combination.base import BaseComposite


class XGBoostComposite(BaseComposite):
    """Strictly walk-forward XGBoost regressor composite.

    At every rebalance date, the model is retrained on an *expanding window*
    of historical observations (subject to a configurable purge gap) and then
    used to predict the next-period cross-sectional returns.

    The model is never fit once on the entire dataset; it is retrained at
    every rebalance date to ensure no look-ahead bias.
    """

    method_name: str = "xgboost"

    def __init__(self, logger: logging.Logger | None = None) -> None:
        super().__init__(logger=logger)
        self.model: Optional[object] = None
        self._feature_names: List[str] = []

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
        """Retrain the XGBoost model using an expanding window up to *eval_date*.

        Respects the configured purge gap between the last training sample
        and the evaluation date.
        """
        if xgb is None:
            raise ImportError("xgboost is required for XGBoostComposite")

        min_train = config.get("xgb_min_train_months", 36)
        purge_gap = config.get("xgb_purge_gap_months", 1)

        # Determine training dates: all dates strictly before (eval_date - purge_gap)
        cutoff = eval_date - pd.DateOffset(months=purge_gap)
        available_dates = sorted(factor_scores.index.get_level_values("date").unique())
        train_dates = [d for d in available_dates if d <= cutoff]

        if len(train_dates) < min_train:
            self.logger.info(
                "Skipping XGBoost fit at %s: only %d training dates available (need %d)",
                eval_date.date(), len(train_dates), min_train,
            )
            self.model = None
            return

        # Build training set
        train_idx = factor_scores.index.get_level_values("date").isin(train_dates)
        X_train = factor_scores.loc[train_idx].copy()
        y_train = forward_returns.loc[X_train.index].copy()

        # Drop rows with NaN in target
        valid = y_train.notna()
        X_train = X_train.loc[valid]
        y_train = y_train.loc[valid]

        if len(X_train) < min_train:
            self.model = None
            return

        self._feature_names = X_train.columns.tolist()

        # Train XGBoost regressor
        model = xgb.XGBRegressor(
            n_estimators=config.get("xgb_n_estimators", 200),
            max_depth=config.get("xgb_max_depth", 4),
            learning_rate=config.get("xgb_learning_rate", 0.05),
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        model.fit(X_train.values, y_train.values)
        self.model = model

        self.logger.info(
            "XGBoost trained at %s: %d samples, %d features, dates %s to %s",
            eval_date.date(), len(X_train), len(self._feature_names),
            train_dates[0].date(), train_dates[-1].date(),
        )

    def predict(
        self,
        factor_scores: pd.DataFrame,
        eval_date: pd.Timestamp,
    ) -> pd.Series:
        """Generate XGBoost-predicted composite scores at *eval_date*."""
        if self.model is None:
            return pd.Series(dtype=float)

        try:
            cross_section = factor_scores.loc[eval_date]
        except KeyError:
            return pd.Series(dtype=float)

        # Ensure column order matches training
        X_pred = cross_section.reindex(columns=self._feature_names).fillna(0.0)
        preds = self.model.predict(X_pred.values)

        result = pd.Series(preds, index=cross_section.index, name="composite_score")
        return result

    def get_model(self) -> Optional[object]:
        """Return the underlying XGBoost model (for SHAP analysis, serialization)."""
        return self.model

    def get_feature_names(self) -> List[str]:
        """Return the ordered feature names used in training."""
        return self._feature_names

    def save_model(self, path: str) -> None:
        """Serialize the latest trained XGBoost model to disk."""
        if self.model is not None and xgb is not None:
            self.model.save_model(path)
            self.logger.info("Saved XGBoost model to %s", path)
