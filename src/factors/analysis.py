from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import Week2Config
from src.factors.return_matrix import ReturnMatrix


@dataclass
class Week2AnalysisSummary:
    return_matrix_shape: tuple[int, int]
    trimmed_matrix_non_null: int
    cross_sectional_dates: int
    annual_stats_rows: int
    artifacts_written: int


class Week2AnalysisPipeline:
    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger(__name__)

    def run(
        self,
        returns_df: pd.DataFrame,
        membership_df: pd.DataFrame,
        config: Week2Config,
    ) -> Week2AnalysisSummary:
        if returns_df.empty:
            raise ValueError("Week 2 analysis requires non-empty returns data.")
        if membership_df.empty:
            raise ValueError("Week 2 analysis requires non-empty universe membership data.")

        return_matrix = ReturnMatrix.from_long(returns_df.reset_index(), return_col=config.return_column)
        cleaned_matrix = return_matrix.fill_gaps(
            method=config.gap_fill_method,
            max_consecutive=config.max_consecutive_missing,
        )
        aligned_membership = self._align_membership_to_matrix(
            membership_df=membership_df,
            matrix_dates=cleaned_matrix.dates,
        )
        trimmed_matrix = cleaned_matrix.trim_by_universe(aligned_membership)
        cross_sectional_stats = trimmed_matrix.cross_sectional_stats_over_time()
        annual_stats = trimmed_matrix.annual_stats()

        artifacts_written = 0
        if config.save_artifacts:
            artifacts_written = self._write_artifacts(
                config=config,
                cleaned_matrix=cleaned_matrix,
                trimmed_matrix=trimmed_matrix,
                cross_sectional_stats=cross_sectional_stats,
                annual_stats=annual_stats,
            )

        return Week2AnalysisSummary(
            return_matrix_shape=trimmed_matrix.shape,
            trimmed_matrix_non_null=int(trimmed_matrix.data.notna().sum().sum()),
            cross_sectional_dates=len(cross_sectional_stats),
            annual_stats_rows=len(annual_stats),
            artifacts_written=artifacts_written,
        )

    def _align_membership_to_matrix(
        self,
        membership_df: pd.DataFrame,
        matrix_dates: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        membership = membership_df.copy()
        membership["date"] = pd.to_datetime(membership["date"])
        membership = membership.sort_values(["ticker", "date"])
        target_dates = pd.DataFrame({"date": pd.DatetimeIndex(matrix_dates).sort_values().unique()})

        aligned_frames: list[pd.DataFrame] = []
        for ticker, ticker_membership in membership.groupby("ticker", sort=False):
            aligned = pd.merge_asof(
                target_dates,
                ticker_membership[["date", "in_universe"]].sort_values("date"),
                on="date",
                direction="backward",
            )
            aligned["ticker"] = ticker
            aligned_frames.append(aligned.dropna(subset=["in_universe"]))

        result = pd.concat(aligned_frames, ignore_index=True) if aligned_frames else pd.DataFrame()
        if not result.empty:
            result["in_universe"] = result["in_universe"].astype(bool)
        return result

    def _write_artifacts(
        self,
        config: Week2Config,
        cleaned_matrix: ReturnMatrix,
        trimmed_matrix: ReturnMatrix,
        cross_sectional_stats: pd.DataFrame,
        annual_stats: pd.DataFrame,
    ) -> int:
        artifact_dir = Path(config.artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        cleaned_matrix.data.to_csv(artifact_dir / "return_matrix_cleaned.csv")
        trimmed_matrix.data.to_csv(artifact_dir / "return_matrix_trimmed.csv")
        cross_sectional_stats.to_csv(artifact_dir / "cross_sectional_stats.csv")
        annual_stats.to_csv(artifact_dir / "annual_stats.csv")
        self._plot_latest_histogram(trimmed_matrix, artifact_dir / "latest_cross_section_hist.png")
        self._plot_annual_tail_diagnostics(annual_stats, artifact_dir / "annual_tail_diagnostics.png")

        artifact_count = 6
        self.logger.info("Wrote %d Week 2 artifacts to %s", artifact_count, artifact_dir)
        return artifact_count

    def _plot_latest_histogram(self, matrix: ReturnMatrix, output_path: Path) -> None:
        import matplotlib.pyplot as plt
        import seaborn as sns

        latest_date = matrix.dates.max()
        latest_values = matrix.data.loc[latest_date].dropna()

        fig, ax = plt.subplots(figsize=(10, 6))
        if latest_values.empty:
            ax.text(0.5, 0.5, "No in-universe returns", ha="center", va="center", transform=ax.transAxes)
        else:
            sns.histplot(latest_values, bins=30, kde=True, ax=ax)
        ax.set_title(f"Cross-Sectional Return Distribution: {latest_date.date()}")
        ax.set_xlabel("Return")
        ax.set_ylabel("Frequency")
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

    def _plot_annual_tail_diagnostics(self, annual_stats: pd.DataFrame, output_path: Path) -> None:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        if annual_stats.empty:
            axes[0].text(0.5, 0.5, "No annual stats available", ha="center", va="center", transform=axes[0].transAxes)
            axes[1].text(0.5, 0.5, "No annual stats available", ha="center", va="center", transform=axes[1].transAxes)
        else:
            annual_stats["skewness"].plot(kind="bar", ax=axes[0], color="#1f77b4")
            axes[0].set_title("Annual Cross-Sectional Skewness")
            axes[0].set_ylabel("Skewness")

            annual_stats["excess_kurtosis"].plot(kind="bar", ax=axes[1], color="#d62728")
            axes[1].set_title("Annual Excess Kurtosis")
            axes[1].set_ylabel("Excess Kurtosis")
            axes[1].set_xlabel("Year")

        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
