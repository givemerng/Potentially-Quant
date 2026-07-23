"""Unified ConstraintManager for assembling optimization constraint sets."""

from __future__ import annotations
import logging
from typing import List, Optional, Tuple, Dict, Any
import cvxpy as cp
import numpy as np
import pandas as pd

from src.portfolio.constraints.position import build_position_constraints
from src.portfolio.constraints.sector import build_sector_constraints
from src.portfolio.constraints.cvar import build_cvar_constraint


class ConstraintManager:
    """Orchestrates assembly of position bounds, sector caps, and CVaR risk constraints."""

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def assemble(
        self,
        w: cp.Variable,
        tickers: List[str],
        sector_map: Optional[pd.Series] = None,
        scenario_matrix: Optional[np.ndarray] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[cp.Constraint], Dict[str, Any]]:
        """Assemble all constraints for CVXPY optimization.

        Returns
        -------
        Tuple[List[cp.Constraint], Dict[str, Any]]
            (List of CVXPY Constraints, Metadata dictionary of auxiliary variables)
        """
        cfg = config or {}
        max_pos = cfg.get("max_position_size", 0.05)
        max_sec = cfg.get("max_sector_exposure", 0.20)
        target_cvar = cfg.get("target_cvar", 0.20)
        alpha = cfg.get("alpha_confidence", 0.95)

        constraints: List[cp.Constraint] = []
        meta: Dict[str, Any] = {}

        # 1. Position constraints
        pos_constraints = build_position_constraints(
            w=w, n_assets=len(tickers), max_position_size=max_pos, long_only=True, net_exposure=1.0
        )
        constraints.extend(pos_constraints)

        # 2. Sector constraints
        if sector_map is not None:
            sec_constraints = build_sector_constraints(
                w=w, tickers=tickers, sector_map=sector_map, max_sector_exposure=max_sec
            )
            constraints.extend(sec_constraints)

        # 3. CVaR constraint
        if scenario_matrix is not None and scenario_matrix.shape[0] > 0:
            cvar_constraints, gamma, u = build_cvar_constraint(
                w=w, scenario_matrix=scenario_matrix, alpha=alpha, target_cvar=target_cvar
            )
            constraints.extend(cvar_constraints)
            meta["gamma"] = gamma
            meta["u"] = u

        return constraints, meta
