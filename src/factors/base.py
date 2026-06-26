from __future__ import annotations

import abc
import logging

import pandas as pd
from sqlalchemy.engine import Engine


class BaseAlpha(abc.ABC):
    """Abstract base class for all quantitative alpha factors."""

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    @abc.abstractmethod
    def compute(
        self,
        engine: Engine,
        prices_df: pd.DataFrame,
        metadata_df: pd.DataFrame,
        fundamentals_df: pd.DataFrame,
        date: pd.Timestamp,
        config: dict,
    ) -> pd.Series:
        """Compute the raw factor scores cross-sectionally for a specific date.

        Each factor subclass should implement this calculation. Missing inputs for
        a ticker should result in a NaN/NA value, which will be handled during
        subsequent winsorization, z-score, and neutralization stages.

        Args:
            engine: Active database engine.
            prices_df: DataFrame indexed by (date, ticker) containing pricing.
            metadata_df: DataFrame containing ticker sector and shares outstanding.
            fundamentals_df: DataFrame containing quarterly financial metrics.
            date: Target date to evaluate the factor.
            config: Week 3 configuration dictionary.

        Returns:
            pd.Series: Indexed by ticker, containing the raw factor scores.
        """
        pass
