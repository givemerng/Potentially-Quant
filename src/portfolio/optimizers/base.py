"""Base Abstract Class for Portfolio Optimizers."""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import logging
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np


@dataclass
class OptimizationResult:
    """Standardized result object returned by all portfolio optimizers."""

    weights: pd.Series
    status: str
    solver_name: str
    objective_value: Optional[float] = None
    iterations: Optional[int] = None
    turnover: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_successful(self) -> bool:
        return self.status in ("optimal", "optimal_inaccurate")


class BasePortfolioOptimizer(ABC):
    """Abstract base class for all portfolio optimizers."""

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def optimize(
        self,
        expected_returns: pd.Series,
        covariance_df: pd.DataFrame,
        current_weights: Optional[pd.Series] = None,
        sector_map: Optional[pd.Series] = None,
        returns_panel: Optional[pd.DataFrame] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> OptimizationResult:
        """Run portfolio optimization and return optimal weight series + diagnostic metadata."""
        pass
