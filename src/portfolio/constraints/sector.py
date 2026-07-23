"""Sector exposure cap constraints."""

from __future__ import annotations
from typing import List, Dict
import cvxpy as cp
import numpy as np
import pandas as pd


def build_sector_constraints(
    w: cp.Variable,
    tickers: List[str],
    sector_map: pd.Series,
    max_sector_exposure: float = 0.20,
) -> List[cp.Constraint]:
    """Build sector exposure constraints (sum_{i in Sector S} w_i <= max_sector_exposure)."""
    constraints = []
    if sector_map.empty or max_sector_exposure >= 1.0:
        return constraints

    sectors = sector_map.dropna().unique()
    ticker_to_idx = {t: i for i, t in enumerate(tickers)}

    for sector in sectors:
        sector_tickers = sector_map[sector_map == sector].index
        idx_list = [ticker_to_idx[t] for t in sector_tickers if t in ticker_to_idx]
        if idx_list:
            constraints.append(cp.sum(w[idx_list]) <= max_sector_exposure)

    return constraints
