"""SHAP feature importance analysis for the XGBoost composite model."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import shap
except ImportError:  # pragma: no cover
    shap = None  # type: ignore[assignment]

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None  # type: ignore[assignment]


class SHAPAnalyzer:
    """Compute and visualize SHAP feature importances for an XGBoost model."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.shap_values: Optional[np.ndarray] = None
        self.feature_names: list = []

    def analyze(
        self,
        model: object,
        X: pd.DataFrame,
        feature_names: list,
    ) -> pd.DataFrame:
        """Compute SHAP values for the given model and feature matrix.

        Args:
            model: Trained XGBoost model.
            X: Feature matrix (tickers × factors) used for explanation.
            feature_names: Ordered factor names matching X columns.

        Returns:
            DataFrame of SHAP values with columns matching factor names.
        """
        if shap is None:
            raise ImportError("shap package is required for SHAPAnalyzer")

        self.logger.info("Computing SHAP values for %d observations × %d features", len(X), len(feature_names))

        explainer = shap.TreeExplainer(model)
        self.shap_values = explainer.shap_values(X.values if hasattr(X, "values") else X)
        self.feature_names = feature_names

        shap_df = pd.DataFrame(self.shap_values, columns=feature_names)
        if hasattr(X, "index"):
            shap_df.index = X.index

        return shap_df

    def save_plots(self, X: pd.DataFrame, artifact_dir: str | Path) -> list[str]:
        """Generate and save SHAP summary and beeswarm plots.

        Args:
            X: Feature matrix used during ``analyze``.
            artifact_dir: Directory to save PNG artifacts.

        Returns:
            List of file paths written.
        """
        if self.shap_values is None:
            self.logger.warning("No SHAP values computed; call analyze() first")
            return []
        if plt is None:
            self.logger.warning("matplotlib not available; skipping SHAP plots")
            return []

        artifact_dir = Path(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        written: list[str] = []

        X_array = X.values if hasattr(X, "values") else X

        # Summary bar plot
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            shap.summary_plot(
                self.shap_values,
                X_array,
                feature_names=self.feature_names,
                plot_type="bar",
                show=False,
            )
            path = str(artifact_dir / "shap_summary.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close("all")
            written.append(path)
            self.logger.info("Saved SHAP summary bar plot to %s", path)
        except Exception as exc:
            self.logger.warning("Failed to generate SHAP summary bar plot: %s", exc)

        # Beeswarm plot
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            shap.summary_plot(
                self.shap_values,
                X_array,
                feature_names=self.feature_names,
                show=False,
            )
            path = str(artifact_dir / "shap_beeswarm.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close("all")
            written.append(path)
            self.logger.info("Saved SHAP beeswarm plot to %s", path)
        except Exception as exc:
            self.logger.warning("Failed to generate SHAP beeswarm plot: %s", exc)

        return written

    def get_importance_ranking(self) -> pd.Series:
        """Return factors ranked by mean |SHAP value| (descending)."""
        if self.shap_values is None:
            return pd.Series(dtype=float)

        mean_abs = np.abs(self.shap_values).mean(axis=0)
        ranking = pd.Series(mean_abs, index=self.feature_names, name="mean_abs_shap")
        return ranking.sort_values(ascending=False)
