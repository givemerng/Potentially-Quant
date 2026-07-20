"""Week 6 reporting: comparison tables, factor weight CSVs, and diagnostic plots."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:  # pragma: no cover
    plt = None  # type: ignore[assignment]
    sns = None  # type: ignore[assignment]


class Week6Reporter:
    """Generate Week 6 comparison reports and diagnostic artifacts."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def generate_comparison_report(
        self,
        method_metrics: Dict[str, Dict[str, float]],
        artifact_dir: str | Path,
    ) -> None:
        """Write a side-by-side comparison of all combination methods.

        Args:
            method_metrics: Dict mapping method_name → {metric_name: value}.
                            Expected metrics: ic_mean, icir, sharpe_ratio,
                            sortino_ratio, max_drawdown, calmar_ratio.
            artifact_dir: Directory to save CSV and markdown files.
        """
        artifact_dir = Path(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        # Build comparison DataFrame
        rows = []
        for method, metrics in method_metrics.items():
            row = {"method": method}
            row.update(metrics)
            rows.append(row)

        df = pd.DataFrame(rows)

        # Save CSV
        csv_path = artifact_dir / "composite_comparison.csv"
        df.to_csv(csv_path, index=False)
        self.logger.info("Saved comparison CSV to %s", csv_path)

        # Save Markdown
        md_path = artifact_dir / "composite_comparison.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Week 6 — Composite Method Comparison\n\n")
            f.write(self._to_markdown(df, index=False))
            f.write("\n")
        self.logger.info("Saved comparison markdown to %s", md_path)

    def _to_markdown(self, df: pd.DataFrame, index: bool = True) -> str:
        try:
            return df.to_markdown(index=index)
        except Exception:
            cols = list(df.columns)
            if index:
                cols = ["index"] + cols
            header = "| " + " | ".join(map(str, cols)) + " |"
            separator = "| " + " | ".join(["---"] * len(cols)) + " |"
            rows = []
            for idx, row in zip(df.index, df.values):
                val_list = [idx] + list(row) if index else list(row)
                rows.append("| " + " | ".join(str(v) for v in val_list) + " |")
            return "\n".join([header, separator] + rows)


    def save_factor_weights_csv(
        self,
        weights_df: pd.DataFrame,
        method: str,
        artifact_dir: str | Path,
    ) -> None:
        """Save rolling factor weights to CSV.

        Args:
            weights_df: DataFrame with columns [date, factor_name, weight].
            method: Combination method name.
            artifact_dir: Output directory.
        """
        artifact_dir = Path(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        path = artifact_dir / f"factor_weights_{method}.csv"
        weights_df.to_csv(path, index=False)
        self.logger.info("Saved %s factor weights to %s", method, path)

    def save_fm_coefficients_csv(
        self,
        coeff_df: pd.DataFrame,
        artifact_dir: str | Path,
    ) -> None:
        """Save rolling Fama-MacBeth coefficients for diagnostics.

        Args:
            coeff_df: DataFrame with date column and one column per factor.
            artifact_dir: Output directory.
        """
        artifact_dir = Path(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        path = artifact_dir / "fama_macbeth_coefficients.csv"
        coeff_df.to_csv(path, index=False)
        self.logger.info("Saved FM coefficients to %s", path)

    def save_shap_values_csv(
        self,
        shap_df: pd.DataFrame,
        artifact_dir: str | Path,
    ) -> None:
        """Save SHAP values to CSV."""
        artifact_dir = Path(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        path = artifact_dir / "shap_values.csv"
        shap_df.to_csv(path, index=True)
        self.logger.info("Saved SHAP values to %s", path)

    def plot_factor_correlation_matrix(
        self,
        factor_scores: pd.DataFrame,
        artifact_dir: str | Path,
    ) -> Optional[str]:
        """Generate and save a factor-factor correlation heatmap.

        Args:
            factor_scores: Wide DataFrame with factor columns.
            artifact_dir: Output directory.

        Returns:
            Path to the saved PNG, or None if plotting is unavailable.
        """
        if plt is None or sns is None:
            self.logger.warning("matplotlib/seaborn not available; skipping correlation plot")
            return None

        artifact_dir = Path(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        corr = factor_scores.corr(method="spearman")

        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            ax=ax,
            annot_kws={"size": 7},
        )
        ax.set_title("Factor Rank Correlation Matrix (Spearman)", fontsize=14)
        plt.tight_layout()

        path = str(artifact_dir / "factor_correlation_matrix.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        self.logger.info("Saved factor correlation matrix to %s", path)
        return path
