"""Mean-Variance Markowitz Quadratic Portfolio Optimizer."""

from __future__ import annotations
import logging
from typing import Dict, List, Optional, Any
import cvxpy as cp
import numpy as np
import pandas as pd

from src.portfolio.optimizers.base import BasePortfolioOptimizer, OptimizationResult
from src.portfolio.constraints.manager import ConstraintManager
from src.portfolio.execution.solver import OptimizationSolverManager


class MeanVarianceOptimizer(BasePortfolioOptimizer):
    """Markowitz Mean-Variance Portfolio Optimizer with Risk Aversion."""

    def __init__(
        self,
        constraint_manager: Optional[ConstraintManager] = None,
        solver_manager: Optional[OptimizationSolverManager] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        super().__init__(logger=logger)
        self.constraint_manager = constraint_manager or ConstraintManager(logger=self.logger)
        self.solver_manager = solver_manager or OptimizationSolverManager(logger=self.logger)

    def optimize(
        self,
        expected_returns: pd.Series,
        covariance_df: pd.DataFrame,
        current_weights: Optional[pd.Series] = None,
        sector_map: Optional[pd.Series] = None,
        returns_panel: Optional[pd.DataFrame] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> OptimizationResult:
        cfg = config or {}
        tickers = list(expected_returns.index)
        N = len(tickers)

        if N == 0:
            return OptimizationResult(weights=pd.Series(dtype=float), status="empty_universe", solver_name="NONE")

        mu = expected_returns.fillna(0.0).values
        cov_matrix = covariance_df.reindex(index=tickers, columns=tickers).fillna(0.0).values
        risk_aversion = cfg.get("risk_aversion", 1.0)
        tc_lambda = cfg.get("tc_penalty_lambda", 1.0)
        cost_per_unit = (cfg.get("commission_bps", 5.0) + cfg.get("bid_ask_spread_bps", 10.0) / 2.0) / 10000.0

        w_prev = np.zeros(N)
        if current_weights is not None:
            w_prev = current_weights.reindex(tickers).fillna(0.0).values

        w = cp.Variable(N, name="w")

        constraints, _ = self.constraint_manager.assemble(
            w=w, tickers=tickers, sector_map=sector_map, scenario_matrix=None, config=cfg
        )

        expected_ret_term = mu @ w
        risk_term = 0.5 * risk_aversion * cp.quad_form(w, cov_matrix)
        tc_penalty_term = tc_lambda * cost_per_unit * cp.sum(cp.abs(w - w_prev))

        objective = cp.Maximize(expected_ret_term - risk_term - tc_penalty_term)
        problem = cp.Problem(objective, constraints)

        fallback_w = np.ones(N) / N
        fallback_series = pd.Series(fallback_w, index=tickers)

        weights, status, solver_used, obj_val = self.solver_manager.solve(
            problem=problem, w_var=w, tickers=tickers, fallback_weights=fallback_series
        )

        turnover = float(np.sum(np.abs(weights.values - w_prev)))

        return OptimizationResult(
            weights=weights,
            status=status,
            solver_name=solver_used,
            objective_value=obj_val,
            turnover=turnover,
            metadata={"risk_aversion": risk_aversion},
        )
