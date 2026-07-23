"""Institutional Portfolio Performance Tear Sheet & Report Artifact Generator."""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, Optional, Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class PortfolioReporter:
    """Generates equity curves, drawdown charts, factor return attribution, and CSV summary reports."""

    def __init__(self, output_dir: str | Path = "data/processed/week9", logger: Optional[logging.Logger] = None) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def generate_report(
        self,
        daily_returns: pd.Series,
        metrics: Dict[str, float],
        weights_df: pd.DataFrame,
        opt_logs_df: pd.DataFrame,
        attribution_df: Optional[pd.DataFrame] = None,
        prefix: str = "cvar_optimized",
    ) -> None:
        """Generate plots and CSV artifacts."""
        # 1. Performance summary CSV
        if metrics:
            pd.DataFrame([metrics]).to_csv(self.output_dir / f"{prefix}_metrics.csv", index=False)

        # 2. Optimization log CSV
        if not opt_logs_df.empty:
            opt_logs_df.to_csv(self.output_dir / f"{prefix}_optimization_run_logs.csv", index=False)

        # 3. Weights CSV
        if not weights_df.empty:
            weights_df.to_csv(self.output_dir / f"{prefix}_rebalance_weights.csv", index=False)

        # 4. Attribution CSV
        if attribution_df is not None and not attribution_df.empty:
            attribution_df.to_csv(self.output_dir / f"{prefix}_return_attribution.csv", index=False)

        # 5. Equity Curve Plot
        if not daily_returns.empty:
            plt.figure(figsize=(10, 5))
            cum_rets = (1.0 + daily_returns).cumprod()
            plt.plot(cum_rets.index, cum_rets.values, label=f"{prefix} Equity Curve", color="#1f77b4", linewidth=1.5)
            plt.title(f"Portfolio Performance Teardown - {prefix.upper()}")
            plt.xlabel("Date")
            plt.ylabel("Cumulative Growth ($1.00 Base)")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.output_dir / f"{prefix}_equity_curve.png", dpi=150)
            plt.close()

        self.logger.info("Saved PortfolioReporter artifacts under %s", self.output_dir)
