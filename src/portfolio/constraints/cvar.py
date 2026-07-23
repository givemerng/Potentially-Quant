"""CVaR tail risk constraint formulation."""

from __future__ import annotations
from typing import List, Tuple
import cvxpy as cp
import numpy as np


def build_cvar_constraint(
    w: cp.Variable,
    scenario_matrix: np.ndarray,
    alpha: float = 0.95,
    target_cvar: float = 0.20,
) -> Tuple[List[cp.Constraint], cp.Variable, cp.Variable]:
    """Formulate CVaR tail risk upper bound constraint via Rockafellar-Uryasev linear formulation.

    Parameters
    ----------
    w : cp.Variable
        N-dim asset weight vector.
    scenario_matrix : np.ndarray
        T x N matrix of scenario returns.
    alpha : float
        Confidence level (e.g. 0.95).
    target_cvar : float
        Upper bound on CVaR.

    Returns
    -------
    Tuple[List[cp.Constraint], cp.Variable, cp.Variable]
        (CVXPY Constraints list, gamma VaR variable, u loss auxiliary variable array)
    """
    T, N = scenario_matrix.shape
    gamma = cp.Variable(name="gamma")  # VaR variable
    u = cp.Variable(T, nonneg=True, name="u")  # Auxiliary loss variables

    losses = -scenario_matrix @ w
    constraints = [
        u >= losses - gamma,
        gamma + (1.0 / ((1.0 - alpha) * T)) * cp.sum(u) <= target_cvar,
    ]
    return constraints, gamma, u
