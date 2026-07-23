"""Position bounds & leverage constraints."""

from __future__ import annotations
from typing import List
import cvxpy as cp
import numpy as np


def build_position_constraints(
    w: cp.Variable,
    n_assets: int,
    max_position_size: float = 0.05,
    long_only: bool = True,
    net_exposure: float = 1.0,
) -> List[cp.Constraint]:
    """Build position weight constraints (0 <= w_i <= max_weight, sum(w) == net_exposure)."""
    constraints = []
    if long_only:
        constraints.append(w >= 0.0)
    constraints.append(w <= max_position_size)
    constraints.append(cp.sum(w) == net_exposure)
    return constraints
