from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sqlalchemy.engine import Engine

from src.data.db import factor_metrics_table, upsert_rows

logger = logging.getLogger(__name__)


class FactorEvaluator:
    """Harness to evaluate factor performance and store time-series / summary metrics."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def evaluate_factor(
        self,
        engine: Engine,
        factor_name: str,
        factor_scores_df: pd.DataFrame,
        returns_df: pd.DataFrame,
    ) -> dict[str, float]:
        """Compute performance metrics for a single factor and write them to the DB.

        Evaluates both raw_score and final_score separately.

        Args:
            engine: Active database engine.
            factor_name: Name of the factor.
            factor_scores_df: Long DataFrame of factor scores containing columns
                              [date, ticker, raw_score, final_score].
            returns_df: Long DataFrame of returns containing columns
                        [date, ticker, simple_return] at monthly frequency.

        Returns:
            dict: Summary metrics computed for the factor.
        """
        self.logger.info("Evaluating factor performance: %s", factor_name)

        if factor_scores_df.empty or returns_df.empty:
            self.logger.warning("Empty factor scores or returns dataframe for %s", factor_name)
            return {}

        # Pivot returns to wide format (dates x tickers)
        if not isinstance(returns_df.index, pd.MultiIndex):
            returns_indexed = returns_df.set_index(["date", "ticker"])
        else:
            returns_indexed = returns_df
        returns_wide = returns_indexed["simple_return"].unstack("ticker").sort_index()

        # Forward returns: shift returns back by 1 period so return at t+1 aligns with factor at t
        forward_returns = returns_wide.shift(-1)

        summary_results = {}

        # Evaluate both stages: 'raw' (using raw_score) and 'final' (using final_score)
        for stage, score_col in [("raw", "raw_score"), ("final", "final_score")]:
            if score_col not in factor_scores_df.columns:
                continue

            # Pivot factor scores to wide format
            scores_wide = factor_scores_df.pivot(index="date", columns="ticker", values=score_col).sort_index()

            # Align indexes: only evaluate on dates where we have both factors and forward returns
            common_dates = scores_wide.index.intersection(forward_returns.index).sort_values()
            if len(common_dates) < 3:
                self.logger.warning("Insufficient overlapping dates for factor evaluation on %s (%s)", factor_name, stage)
                continue

            # ------------------------------------------------------------------
            # 1. Compute rolling Information Coefficient (IC)
            # ------------------------------------------------------------------
            ic_records = []
            for date in common_dates:
                scores_t = scores_wide.loc[date].dropna()
                returns_t = forward_returns.loc[date].reindex(scores_t.index).dropna()
                
                # We need at least 3 tickers to compute a rank correlation
                if len(scores_t) >= 3 and len(returns_t) >= 3:
                    # Spearman rank correlation
                    ic_t = scores_t.corr(returns_t, method="spearman")
                    if not pd.isna(ic_t):
                        ic_records.append({
                            "date": date,
                            "factor_name": factor_name,
                            "stage": stage,
                            "metric_name": "ic",
                            "value": float(ic_t)
                        })

            if not ic_records:
                self.logger.warning("No valid IC values calculated for %s (%s)", factor_name, stage)
                continue

            ic_df = pd.DataFrame(ic_records)
            ic_series = ic_df.set_index("date")["value"]

            # Compute aggregate IC metrics
            ic_mean = float(ic_series.mean())
            ic_std = float(ic_series.std())
            icir = ic_mean / ic_std if ic_std != 0 else 0.0

            # ------------------------------------------------------------------
            # 2. Compute Quintile Returns and Sharpe Ratios
            # ------------------------------------------------------------------
            quintile_records = []
            ls_returns = []
            q1_returns = []
            q5_returns = []

            for date in common_dates:
                scores_t = scores_wide.loc[date].dropna()
                returns_t = forward_returns.loc[date].reindex(scores_t.index).dropna()
                
                if len(scores_t) >= 5:  # Need at least 5 stocks to split into quintiles
                    # Rank and split into quintiles (5 groups)
                    ranks = scores_t.rank(method="first")
                    quintiles = pd.qcut(ranks, 5, labels=False)
                    
                    q1_tickers = quintiles[quintiles == 0].index
                    q5_tickers = quintiles[quintiles == 4].index

                    ret_q1 = float(returns_t.reindex(q1_tickers).mean())
                    ret_q5 = float(returns_t.reindex(q5_tickers).mean())
                    ret_ls = ret_q5 - ret_q1

                    # Date of return is t+1 (month of performance)
                    next_month_date = date + pd.offsets.MonthEnd(1)

                    q1_returns.append(ret_q1)
                    q5_returns.append(ret_q5)
                    ls_returns.append(ret_ls)

                    # Store time-series returns in factor_metrics
                    quintile_records.extend([
                        {"date": next_month_date.date(), "factor_name": factor_name, "stage": stage, "metric_name": "return_q1", "value": ret_q1},
                        {"date": next_month_date.date(), "factor_name": factor_name, "stage": stage, "metric_name": "return_q5", "value": ret_q5},
                        {"date": next_month_date.date(), "factor_name": factor_name, "stage": stage, "metric_name": "return_ls", "value": ret_ls}
                    ])

            # Store quintile time-series returns in DB
            if quintile_records:
                upsert_rows(engine, factor_metrics_table, quintile_records)

            # Compute Quintile Sharpe ratios (annualized, assuming monthly return series)
            sharpe_q1 = np.sqrt(12) * np.mean(q1_returns) / np.std(q1_returns) if q1_returns and np.std(q1_returns) > 0 else 0.0
            sharpe_q5 = np.sqrt(12) * np.mean(q5_returns) / np.std(q5_returns) if q5_returns and np.std(q5_returns) > 0 else 0.0
            sharpe_ls = np.sqrt(12) * np.mean(ls_returns) / np.std(ls_returns) if ls_returns and np.std(ls_returns) > 0 else 0.0

            # ------------------------------------------------------------------
            # 3. Compute Autocorrelation Decay and Half-Life
            # ------------------------------------------------------------------
            avg_autocorrs = {}
            for lag in range(1, 7):
                lag_corrs = []
                for idx in range(len(scores_wide.index) - lag):
                    d_t = scores_wide.index[idx]
                    d_t_lag = scores_wide.index[idx + lag]
                    
                    s_t = scores_wide.loc[d_t].dropna()
                    s_t_lag = scores_wide.loc[d_t_lag].reindex(s_t.index).dropna()
                    
                    if len(s_t) >= 3 and len(s_t_lag) >= 3:
                        corr = s_t.corr(s_t_lag, method="spearman")
                        if not pd.isna(corr):
                            lag_corrs.append(corr)
                
                avg_autocorrs[lag] = np.mean(lag_corrs) if lag_corrs else 0.0

            # Estimate half-life using OLS: ln(autocorr(k)) = -lambda * k
            half_life = 0.0
            lags = []
            ln_corrs = []
            for lag, corr in avg_autocorrs.items():
                if corr > 0:  # Can only take log of positive values
                    lags.append(-lag)
                    ln_corrs.append(np.log(corr))
            
            if len(lags) >= 2:
                try:
                    # Regress through origin (no constant)
                    model = sm.OLS(ln_corrs, lags)
                    results = model.fit()
                    lam = float(results.params[0])
                    if lam > 0:
                        half_life = np.log(2) / lam
                except Exception as exc:
                    self.logger.warning("Failed to fit half-life OLS decay model: %s", exc)

            # ------------------------------------------------------------------
            # 4. Write all evaluation results to DB
            # ------------------------------------------------------------------
            last_date = common_dates[-1]

            # Collect summary metrics (indexed by the last evaluation date of the backtest)
            summary_records = [
                {"date": last_date.date(), "factor_name": factor_name, "stage": stage, "metric_name": "ic_mean", "value": ic_mean},
                {"date": last_date.date(), "factor_name": factor_name, "stage": stage, "metric_name": "ic_std", "value": ic_std},
                {"date": last_date.date(), "factor_name": factor_name, "stage": stage, "metric_name": "icir", "value": icir},
                {"date": last_date.date(), "factor_name": factor_name, "stage": stage, "metric_name": "sharpe_q1", "value": float(sharpe_q1)},
                {"date": last_date.date(), "factor_name": factor_name, "stage": stage, "metric_name": "sharpe_q5", "value": float(sharpe_q5)},
                {"date": last_date.date(), "factor_name": factor_name, "stage": stage, "metric_name": "sharpe_ls", "value": float(sharpe_ls)},
                {"date": last_date.date(), "factor_name": factor_name, "stage": stage, "metric_name": "half_life", "value": float(half_life)}
            ]

            # Add decay lags
            for lag, corr in avg_autocorrs.items():
                summary_records.append({
                    "date": last_date.date(),
                    "factor_name": factor_name,
                    "stage": stage,
                    "metric_name": f"autocorr_lag{lag}",
                    "value": float(corr)
                })

            # Store IC timeseries in DB
            ic_records_clean = [
                {**rec, "date": rec["date"].date()} for rec in ic_records
            ]
            upsert_rows(engine, factor_metrics_table, ic_records_clean)
            upsert_rows(engine, factor_metrics_table, summary_records)

            # Store summary metrics in the dictionary to return
            summary_results[f"{stage}_ic_mean"] = ic_mean
            summary_results[f"{stage}_icir"] = icir
            summary_results[f"{stage}_sharpe_ls"] = sharpe_ls
            summary_results[f"{stage}_half_life"] = half_life

            self.logger.info(
                "Completed %s (%s) evaluation: IC mean = %.4f, ICIR = %.4f, LS Sharpe = %.4f, Half-life = %.1f months",
                factor_name, stage, ic_mean, icir, sharpe_ls, half_life
            )

        return summary_results
