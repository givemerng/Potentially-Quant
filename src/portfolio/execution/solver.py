"""CVXPY Solver Abstraction Manager with Automatic Solver Fallback Chain."""

from __future__ import annotations
import logging
from typing import List, Optional, Tuple, Any
import cvxpy as cp
import numpy as np
import pandas as pd


class OptimizationSolverManager:
    """Solves CVXPY problems using a solver fallback chain (CLARABEL -> OSQP -> ECOS -> Fallback)."""

    DEFAULT_SOLVER_CHAIN = ["CLARABEL", "OSQP", "ECOS"]

    def __init__(
        self,
        solver_chain: Optional[List[str]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.solver_chain = solver_chain or self.DEFAULT_SOLVER_CHAIN
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def solve(
        self,
        problem: cp.Problem,
        w_var: cp.Variable,
        tickers: List[str],
        fallback_weights: Optional[pd.Series] = None,
    ) -> Tuple[pd.Series, str, str, Optional[float]]:
        """Attempt solving problem across solver chain. Returns (weights, status, solver_used, obj_val)."""
        installed_solvers = cp.installed_solvers()

        for solver_name in self.solver_chain:
            if solver_name not in installed_solvers:
                self.logger.debug("Solver %s not installed, skipping", solver_name)
                continue

            try:
                solver_attr = getattr(cp, solver_name)
                obj_val = problem.solve(solver=solver_attr, verbose=False)

                if problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) and w_var.value is not None:
                    raw_weights = np.array(w_var.value).flatten()
                    # Clean numerical noise & normalize
                    raw_weights = np.maximum(0.0, raw_weights)
                    weight_sum = np.sum(raw_weights)
                    if weight_sum > 0:
                        raw_weights = raw_weights / weight_sum

                    weights_series = pd.Series(raw_weights, index=tickers, name="weight")
                    self.logger.info("Optimization solved successfully with %s (status=%s)", solver_name, problem.status)
                    return weights_series, problem.status, solver_name, float(obj_val) if obj_val is not None else None
                else:
                    self.logger.warning("Solver %s returned non-optimal status: %s", solver_name, problem.status)
            except Exception as exc:
                self.logger.warning("Solver %s failed with exception: %s", solver_name, exc)

        # Fallback handling
        self.logger.error("All solvers in chain %s failed. Triggering fallback portfolio.", self.solver_chain)
        if fallback_weights is not None and not fallback_weights.empty:
            weights_series = fallback_weights.reindex(tickers).fillna(0.0)
            if weights_series.sum() > 0:
                weights_series = weights_series / weights_series.sum()
            return weights_series, "fallback_weights", "FALLBACK", None

        # Equal weight fallback
        eq_weights = pd.Series(1.0 / len(tickers), index=tickers, name="weight")
        return eq_weights, "fallback_equal_weight", "FALLBACK", None
