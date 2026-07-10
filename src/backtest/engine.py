from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from sqlalchemy.engine import Engine

from src.backtest.portfolio import PortfolioConstructor
from src.backtest.rebalancer import Rebalancer
from src.backtest.transaction_cost import LinearTransactionCostModel
from src.backtest.metrics import PortfolioMetrics
from src.data.db import (
    upsert_rows,
    portfolio_weights_table,
    trades_table,
    backtest_results_table,
    backtest_metrics_table,
)


class VectorizedBacktester:
    """Vectorized simulation engine for portfolio backtesting."""

    def __init__(
        self,
        initial_capital: float = 10000000.0,
        rebalance_frequency: str = "monthly",
        weighting_method: str = "equal_weight_long_only",
        max_position_size: float = 0.05,
        commission_bps: float = 5.0,
        bid_ask_spread_bps: float = 10.0,
        long_only: bool = True,
        net_exposure: float = 1.0,
        gross_exposure: float = 1.0,
        logger: logging.Logger | None = None,
    ) -> None:
        self.initial_capital = initial_capital
        self.rebalance_frequency = rebalance_frequency
        self.weighting_method = weighting_method
        self.max_position_size = max_position_size
        self.long_only = long_only
        self.net_exposure = net_exposure
        self.gross_exposure = gross_exposure
        self.logger = logger or logging.getLogger(self.__class__.__name__)

        self.portfolio_constructor = PortfolioConstructor(logger=self.logger)
        self.rebalancer = Rebalancer(frequency=rebalance_frequency, logger=self.logger)
        self.tc_model = LinearTransactionCostModel(
            commission_bps=commission_bps,
            bid_ask_spread_bps=bid_ask_spread_bps
        )

    def run(
        self,
        factor_scores_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        universe_membership_df: pd.DataFrame,
        backtest_name: str,
    ) -> dict:
        """Run the vectorized portfolio simulation.

        Args:
            factor_scores_df: DataFrame of factor scores (date, ticker, raw_score, final_score, factor_name).
            returns_df: DataFrame of asset returns (date, ticker, simple_return, log_return).
            universe_membership_df: DataFrame of universe membership flags (date, ticker, in_universe).
            backtest_name: Identifier for this backtest run (e.g. 'PriceMomentum_EW_LO').

        Returns:
            dict containing backtest results and performance summary.
        """
        self.logger.info("Running backtest for %s", backtest_name)

        if factor_scores_df.empty or returns_df.empty:
            self.logger.warning("Empty input data; backtest aborted.")
            return {}

        # Formats and copies to avoid warnings/side-effects
        factor_scores_df = factor_scores_df.copy()
        factor_scores_df["date"] = pd.to_datetime(factor_scores_df["date"])
        returns_df = returns_df.copy()
        returns_df["date"] = pd.to_datetime(returns_df["date"])

        # Filter to a single factor if multiple are present in factor_scores_df
        factor_names = factor_scores_df["factor_name"].unique()
        if len(factor_names) > 1:
            self.logger.warning("Multiple factors found in input. Using the first one: %s", factor_names[0])
            factor_scores_df = factor_scores_df[factor_scores_df["factor_name"] == factor_names[0]]

        scores_wide = factor_scores_df.pivot(index="date", columns="ticker", values="final_score").sort_index()
        returns_wide = returns_df.pivot(index="date", columns="ticker", values="simple_return").sort_index()

        if not universe_membership_df.empty:
            universe_membership_df = universe_membership_df.copy()
            universe_membership_df["date"] = pd.to_datetime(universe_membership_df["date"])
            universe_wide = universe_membership_df.pivot(index="date", columns="ticker", values="in_universe").sort_index()
        else:
            universe_wide = pd.DataFrame(True, index=scores_wide.index, columns=scores_wide.columns)

        # Determine overlapping rebalance dates
        all_dates = scores_wide.index.intersection(returns_wide.index).sort_values()
        rebalance_dates = self.rebalancer.get_rebalance_schedule(all_dates)

        if len(rebalance_dates) == 0:
            self.logger.warning("No rebalance dates found.")
            return {}

        portfolio_value = self.initial_capital
        prev_weights = pd.Series(0.0, index=scores_wide.columns)

        results_records = []
        weights_records = []
        trades_records = []

        tc_at_start = 0.0

        for i, date in enumerate(rebalance_dates):
            # Identify active universe
            try:
                active_mask = universe_wide.loc[date].fillna(False)
            except KeyError:
                active_mask = pd.Series(True, index=scores_wide.columns)

            # Get scores and mask out-of-universe tickers
            scores_t = scores_wide.loc[date].reindex(scores_wide.columns)
            scores_t[~active_mask] = np.nan

            # Construct weights
            target_weights = self.portfolio_constructor.construct_weights(
                factor_scores=scores_t,
                method=self.weighting_method,
                max_position_size=self.max_position_size,
                long_only=self.long_only,
                net_exposure=self.net_exposure,
                gross_exposure=self.gross_exposure,
            )
            target_weights = target_weights.reindex(scores_wide.columns, fill_value=0.0)

            # Record non-zero weights
            for ticker, w in target_weights.items():
                if w != 0.0:
                    weights_records.append({
                        "date": date.date(),
                        "ticker": ticker,
                        "backtest_name": backtest_name,
                        "weight": float(w),
                    })

            # Rebalancing execution and return calculation
            if i > 0:
                rets_t = returns_wide.loc[date].reindex(scores_wide.columns).fillna(0.0)
                gross_ret = float((prev_weights * rets_t).sum())

                # Drift weights
                if gross_ret > -0.999:
                    prev_weights_drift = prev_weights * (1.0 + rets_t) / (1.0 + gross_ret)
                else:
                    prev_weights_drift = prev_weights

                # Determine rebalance size and turnover
                turnover, weight_change = self.rebalancer.calculate_turnover(
                    prev_weights=prev_weights_drift,
                    target_weights=target_weights,
                    asset_returns=rets_t,
                )
                
                # Transaction cost paid at date
                tc = self.tc_model.calculate_cost(weight_change, turnover)
                
                # Net return of period: includes growth of initial assets minus transaction cost paid at the start
                net_ret = (1.0 + gross_ret) * (1.0 - tc_at_start) - 1.0
                portfolio_value = portfolio_value * (1.0 + net_ret)

                # Store non-zero trade details
                for ticker, dc in weight_change.items():
                    if dc != 0.0:
                        trades_records.append({
                            "date": date.date(),
                            "ticker": ticker,
                            "backtest_name": backtest_name,
                            "trade_type": "buy" if dc > 0 else "sell",
                            "weight_change": float(dc),
                            "turnover_contribution": float(abs(dc)),
                        })

                results_records.append({
                    "date": date.date(),
                    "backtest_name": backtest_name,
                    "gross_return": gross_ret,
                    "net_return": net_ret,
                    "portfolio_value": portfolio_value,
                    "drawdown": 0.0,
                    "turnover": turnover,
                    "transaction_costs": tc,
                })
                tc_at_start = tc
            else:
                # First period: setup portfolio
                turnover = float(target_weights.abs().sum())
                tc_at_start = self.tc_model.calculate_cost(target_weights, turnover)
                portfolio_value = portfolio_value * (1.0 - tc_at_start)

                results_records.append({
                    "date": date.date(),
                    "backtest_name": backtest_name,
                    "gross_return": 0.0,
                    "net_return": -tc_at_start,
                    "portfolio_value": portfolio_value,
                    "drawdown": 0.0,
                    "turnover": turnover,
                    "transaction_costs": tc_at_start,
                })

            prev_weights = target_weights

        # Compute drawdown post-loop
        if results_records:
            results_df = pd.DataFrame(results_records)
            vals = results_df["portfolio_value"]
            running_max = vals.cummax()
            dds = (vals - running_max) / running_max.replace(0.0, np.nan)
            for idx in range(len(results_records)):
                results_records[idx]["drawdown"] = float(dds.iloc[idx]) if not pd.isna(dds.iloc[idx]) else 0.0

        # Calculate metrics
        results_df = pd.DataFrame(results_records)
        net_rets = results_df["net_return"]
        gross_rets = results_df["gross_return"]
        turnovers = results_df["turnover"]

        eval_net_rets = net_rets.iloc[1:] if len(net_rets) > 1 else net_rets
        eval_gross_rets = gross_rets.iloc[1:] if len(gross_rets) > 1 else gross_rets

        metrics = PortfolioMetrics.compute_all(
            net_returns=eval_net_rets,
            gross_returns=eval_gross_rets,
            turnover_series=turnovers,
            frequency=self.rebalance_frequency,
        )

        return {
            "backtest_name": backtest_name,
            "metrics": metrics,
            "results": results_records,
            "weights": weights_records,
            "trades": trades_records,
        }

    def save_results(self, engine: Engine, run_summary: dict) -> dict[str, int]:
        """Upsert the backtest simulation results into PostgreSQL.

        Args:
            engine: Active database engine.
            run_summary: Dict returned by the self.run() method.

        Returns:
            dict mapping table names to the number of rows upserted.
        """
        backtest_name = run_summary["backtest_name"]
        self.logger.info("Saving backtest results for %s to database", backtest_name)

        rows_written = {}

        # 1. Weights
        weights = run_summary["weights"]
        rows_written["portfolio_weights"] = upsert_rows(engine, portfolio_weights_table, weights)

        # 2. Trades
        trades = run_summary["trades"]
        rows_written["trades"] = upsert_rows(engine, trades_table, trades)

        # 3. Results (returns, value, drawdown, costs)
        results = run_summary["results"]
        # Format date objects for database insert
        for r in results:
            r["date"] = pd.to_datetime(r["date"]).date()
        rows_written["backtest_results"] = upsert_rows(engine, backtest_results_table, results)

        # 4. Metrics
        metrics = run_summary["metrics"]
        metrics_rows = [
            {"backtest_name": backtest_name, "metric_name": k, "value": float(v) if not pd.isna(v) else None}
            for k, v in metrics.items()
        ]
        rows_written["backtest_metrics"] = upsert_rows(engine, backtest_metrics_table, metrics_rows)

        self.logger.info("Successfully saved backtest results: %s", rows_written)
        return rows_written
