from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


class Week8Reporter:
    """Generates visualizations, comparison matrices, and summary reports for Week 8 Regime-Adaptive Strategy."""

    def __init__(self, output_dir: str | Path = "data/processed/week8") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_ic_heatmap(
        self,
        summary_ic_matrix: pd.DataFrame,
        filename: str = "factor_regime_ic_heatmap.png",
    ) -> Path:
        """Plots factor x regime IC heatmap."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            summary_ic_matrix,
            annot=True,
            fmt=".3f",
            cmap="vlag",
            center=0,
            cbar_kws={"label": "Regime-Conditional IC"},
            linewidths=0.5,
        )
        plt.title("Factor x Regime Conditional Rank IC", fontsize=14, pad=15)
        plt.xlabel("Economic Market Regime", fontsize=12)
        plt.ylabel("Alpha Factor", fontsize=12)
        plt.tight_layout()

        out_path = self.output_dir / filename
        plt.savefig(out_path, dpi=300)
        plt.close()
        logger.info("Saved IC heatmap to %s", out_path)
        return out_path

    def generate_weights_plot(
        self,
        weights_df: pd.DataFrame,
        filename: str = "regime_adaptive_weights.png",
    ) -> Path:
        """Plots dynamic factor weight trajectories over time."""
        plt.figure(figsize=(12, 6))
        for col in weights_df.columns:
            plt.plot(weights_df.index, weights_df[col], label=col, alpha=0.8, linewidth=1.5)

        plt.title("Regime-Adaptive Dynamic Factor Weights Drift", fontsize=14, pad=15)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Dynamic Allocation Weight", fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

        out_path = self.output_dir / filename
        plt.savefig(out_path, dpi=300)
        plt.close()
        logger.info("Saved dynamic weights plot to %s", out_path)
        return out_path

    def generate_posterior_ic_evolution(
        self,
        posteriors_df: pd.DataFrame,
        filename: str = "posterior_ic_evolution.png",
    ) -> Path:
        """Plots trajectory of posterior IC estimates over time per factor."""
        pivoted = posteriors_df.groupby(["date", "factor_name"])["posterior_ic"].mean().unstack()

        plt.figure(figsize=(12, 6))
        for col in pivoted.columns:
            plt.plot(pivoted.index, pivoted[col], label=col, alpha=0.7, linewidth=1.2)


        plt.axhline(0, color="black", linestyle="--", alpha=0.5)
        plt.title("Bayesian Posterior Factor IC Trajectory Over Time", fontsize=14, pad=15)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Posterior IC Estimate", fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

        out_path = self.output_dir / filename
        plt.savefig(out_path, dpi=300)
        plt.close()
        logger.info("Saved posterior IC evolution plot to %s", out_path)
        return out_path

    def generate_weights_heatmap(
        self,
        weights_df: pd.DataFrame,
        filename: str = "dynamic_weight_allocation_heatmap.png",
    ) -> Path:
        """Plots heatmap of dynamic factor weights over time (factors x dates)."""
        plt.figure(figsize=(14, 8))
        # Transpose so factors are rows, dates are columns
        sample_dates = weights_df.index[::max(1, len(weights_df) // 20)]  # Downsample x-axis ticks
        sns.heatmap(
            weights_df.T,
            cmap="YlGnBu",
            cbar_kws={"label": "Factor Weight"},
            xticklabels=[d.strftime("%Y-%m") if hasattr(d, "strftime") else str(d) for d in sample_dates],
        )
        plt.title("Dynamic Factor Weight Allocation Heatmap", fontsize=14, pad=15)
        plt.xlabel("Evaluation Date", fontsize=12)
        plt.ylabel("Alpha Factor", fontsize=12)
        plt.tight_layout()

        out_path = self.output_dir / filename
        plt.savefig(out_path, dpi=300)
        plt.close()
        logger.info("Saved weight allocation heatmap to %s", out_path)
        return out_path

    def generate_composite_comparison_report(
        self,
        comparison_df: pd.DataFrame,
        filename_csv: str = "week8_composite_comparison.csv",
        filename_md: str = "week8_composite_comparison.md",
    ) -> Tuple[Path, Path]:
        """Saves side-by-side strategy comparison table as CSV and Markdown."""
        csv_path = self.output_dir / filename_csv
        md_path = self.output_dir / filename_md

        comparison_df.to_csv(csv_path)

        try:
            md_table = comparison_df.to_markdown()
        except Exception:
            cols = list(comparison_df.columns)
            header = "| " + " | ".join(map(str, cols)) + " |"
            separator = "| " + " | ".join(["---"] * len(cols)) + " |"
            rows = ["| " + " | ".join(str(val) for val in row) + " |" for row in comparison_df.values]
            md_table = "\n".join([header, separator] + rows)

        md_content = [
            "# Week 8 Multi-Factor Strategy Performance Teardown",
            "",
            "Comparative evaluation of static composite baseline strategies vs. the **Regime-Adaptive Composite Model** across full history and out-of-sample (OOS) evaluation windows.",
            "",
            md_table,
            "",
            "> [!NOTE]",
            "> All metrics account for realistic transaction costs (commissions & bid-ask spreads) and rebalancing turnover penalties.",
        ]


        md_path.write_text("\n".join(md_content), encoding="utf-8")
        logger.info("Saved comparison tables to %s and %s", csv_path, md_path)
        return csv_path, md_path
