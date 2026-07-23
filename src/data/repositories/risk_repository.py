"""RiskRepository: Persists risk attribution matrices and covariance metrics."""

from __future__ import annotations
import logging
from typing import Dict, Any, Optional
import pandas as pd
from sqlalchemy.engine import Engine

from src.data.repositories.base import BaseRepository


class RiskRepository(BaseRepository):
    """Repository for storing risk metrics and return attribution tables."""

    def __init__(self, engine: Optional[Engine] = None, logger: Optional[logging.Logger] = None) -> None:
        super().__init__(engine=engine)
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def save_risk_attribution(self, attribution_df: pd.DataFrame, run_name: str) -> int:
        """Save return and risk attribution metrics to database."""
        if attribution_df.empty or self.engine is None:
            return 0

        try:
            attribution_df["run_name"] = run_name
            attribution_df.to_sql("risk_attribution_metrics", con=self.engine, if_exists="append", index=False)
            self.logger.info("Saved %d risk attribution rows for %s", len(attribution_df), run_name)
            return len(attribution_df)
        except Exception as exc:
            self.logger.warning("Failed to save risk attribution to DB: %s", exc)
            return 0
