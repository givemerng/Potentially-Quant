from __future__ import annotations

import logging
from pathlib import Path
import pandas as pd
import numpy as np


class Reporter:
    """Generates backtest reports, CSV files, tables, and visualization plots."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def generate_report(self, run_summary: dict, artifact_dir: str | Path) -> int:
        """Create all CSV and plot artifacts in the designated directory.

        Args:
            run_summary: Dict returned by VectorizedBacktester.run().
            artifact_dir: Path to the output directory.

        Returns:
            int: Number of artifacts successfully written.
        """
        backtest_name = run_summary["backtest_name"]
        self.logger.info("Generating report artifacts for %s in %s", backtest_name, artifact_dir)

        out_path = Path(artifact_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        results_df = pd.DataFrame(run_summary["results"])
        weights_df = pd.DataFrame(run_summary["weights"])
        trades_df = pd.DataFrame(run_summary["trades"])
        metrics = run_summary["metrics"]

        # Ensure datetime index on results for plotting and grouping
        results_df["date"] = pd.to_datetime(results_df["date"])

        # 1. Save main CSV files
        results_df.to_csv(out_path / "backtest_results.csv", index=False)
        weights_df.to_csv(out_path / "backtest_weights.csv", index=False)
        trades_df.to_csv(out_path / "backtest_trades.csv", index=False)

        # 2. Save metrics table
        metrics_df = pd.DataFrame([{"metric": k, "value": v} for k, v in metrics.items()])
        metrics_df.to_csv(out_path / "backtest_metrics.csv", index=False)

        # 3. Create annual performance stats
        annual_stats = self._create_annual_stats(results_df)
        annual_stats.to_csv(out_path / "annual_performance.csv")

        # 4. Create monthly returns grid
        monthly_grid = self._create_monthly_grid(results_df)
        monthly_grid.to_csv(out_path / "monthly_returns_grid.csv")

        # 5. Generate markdown summary report
        self._write_markdown_summary(backtest_name, metrics, annual_stats, monthly_grid, out_path / "summary.md")

        # 6. Generate plots
        self._plot_curves(results_df, backtest_name, out_path)

        artifact_count = 8  # 5 CSVs, 1 MD, 2 PNGs
        self.logger.info("Wrote %d backtest artifacts under %s", artifact_count, out_path)
        return artifact_count

    def _create_annual_stats(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Group results by year and compute annualized returns, volatility, and drawdowns."""
        if results_df.empty:
            return pd.DataFrame()

        df = results_df.copy()
        df["year"] = df["date"].dt.year

        records = {}
        # Identify frequency from spacing of dates (rough proxy)
        diffs = df["date"].diff().dropna()
        is_monthly = True
        if not diffs.empty:
            # If average spacing is less than 5 days, assume daily
            if diffs.mean().days < 5:
                is_monthly = False

        ann_factor = 12 if is_monthly else 252

        for year, group in df.groupby("year"):
            # Setup period might contain only 1 row with 0/setup return
            if len(group) <= 1 and (group["net_return"] <= 0.0).all() and (group["gross_return"] == 0.0).all():
                continue

            net_ret = (1.0 + group["net_return"]).prod() - 1.0
            gross_ret = (1.0 + group["gross_return"]).prod() - 1.0
            vol = group["net_return"].std(ddof=1) * np.sqrt(ann_factor) if len(group) > 1 else 0.0
            max_dd = group["drawdown"].min()
            turnover = group["turnover"].sum()

            records[year] = {
                "Net Return": net_ret,
                "Gross Return": gross_ret,
                "Volatility": vol,
                "Max Drawdown": max_dd,
                "Turnover": turnover,
            }

        return pd.DataFrame.from_dict(records, orient="index")

    def _create_monthly_grid(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Create a Year x Month return grid."""
        if results_df.empty:
            return pd.DataFrame()

        df = results_df.copy()
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month

        # Group and pivot
        monthly_rets = df.groupby(["year", "month"])["net_return"].apply(lambda x: (1.0 + x).prod() - 1.0)
        grid = monthly_rets.unstack("month").fillna(0.0)

        # Rename columns to standard abbreviated month names
        month_names = {
            1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
            7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
        }
        grid = grid.rename(columns=month_names)
        
        # Add a total column for the year
        grid["Full Year"] = df.groupby("year")["net_return"].apply(lambda x: (1.0 + x).prod() - 1.0)
        return grid

    def _write_markdown_summary(
        self,
        backtest_name: str,
        metrics: dict,
        annual_stats: pd.DataFrame,
        monthly_grid: pd.DataFrame,
        filepath: Path,
    ) -> None:
        """Save a markdown performance sheet."""
        with filepath.open("w", encoding="utf-8") as f:
            f.write(f"# Performance Summary Report: {backtest_name}\n\n")
            
            f.write("## Overall Metrics\n\n")
            f.write("| Metric | Value |\n")
            f.write("| :--- | :--- |\n")
            for k, v in metrics.items():
                name = k.replace("_", " ").title()
                if "ratio" in k.lower() or "sharpe" in k.lower() or "sortino" in k.lower() or "calmar" in k.lower():
                    val_str = f"{v:.4f}"
                elif "turnover" in k.lower() or "volatility" in k.lower() or "return" in k.lower() or "rate" in k.lower() or "drawdown" in k.lower():
                    val_str = f"{v:.2%}"
                else:
                    val_str = f"{v}"
                f.write(f"| {name} | {val_str} |\n")
            f.write("\n")

            f.write("## Annual Performance Table\n\n")
            if not annual_stats.empty:
                # Format to percentage strings
                fmt_df = annual_stats.copy()
                for col in fmt_df.columns:
                    fmt_df[col] = fmt_df[col].apply(lambda x: f"{x:.2%}")
                f.write(fmt_df.to_markdown())
            f.write("\n\n")

            f.write("## Monthly Returns Matrix (Net)\n\n")
            if not monthly_grid.empty:
                fmt_grid = monthly_grid.copy()
                for col in fmt_grid.columns:
                    fmt_grid[col] = fmt_grid[col].apply(lambda x: f"{x:.2%}")
                f.write(fmt_grid.to_markdown())
            f.write("\n")

    def _plot_curves(self, results_df: pd.DataFrame, backtest_name: str, out_path: Path) -> None:
        """Plot and save equity and drawdown curves."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
        except ImportError:
            self.logger.warning("matplotlib not installed; skipping plots.")
            return

        if results_df.empty:
            return

        dates = results_df["date"]
        
        # Calculate cumulative returns
        # Net portfolio value starts normalized to initial value
        net_equity = results_df["portfolio_value"]
        initial_val = net_equity.iloc[0] if len(net_equity) > 0 else 1.0
        # If setup period cost reduced initial_val, let's normalize gross to match
        # Let's plot actual dollar value
        
        # Gross equity (no transaction costs):
        gross_returns = results_df["gross_return"].copy()
        # The first period has 0 gross return
        gross_equity = initial_val * (1.0 + gross_returns).cumprod()

        # Drawdown
        drawdowns = results_df["drawdown"]

        # Plot 1: Equity Curve
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(dates, gross_equity, label="Gross Value (No TC)", color="grey", linestyle="--")
        ax.plot(dates, net_equity, label="Net Value (After TC)", color="#1f77b4", linewidth=2)
        ax.set_title(f"Cumulative Portfolio Value: {backtest_name}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Value ($)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.savefig(out_path / "equity_curve.png", dpi=150)
        plt.close()

        # Plot 2: Drawdown Curve
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.fill_between(dates, drawdowns, 0, color="red", alpha=0.3)
        ax.plot(dates, drawdowns, color="red", linewidth=1.5)
        ax.set_title(f"Portfolio Drawdown Profile: {backtest_name}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown")
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.savefig(out_path / "drawdown_curve.png", dpi=150)
        plt.close()
