"""OptimizationRepository: Persists portfolio optimization run metadata and logs."""

from __future__ import annotations
import logging
from typing import Dict, Any, Optional
import pandas as pd
from sqlalchemy import Table, Column, String, Float, DateTime, MetaData
from sqlalchemy.engine import Engine

from src.data.repositories.base import BaseRepository


class OptimizationRepository(BaseRepository):
    """Repository for managing portfolio optimization run metadata persistence."""

    def __init__(self, engine: Optional[Engine] = None, logger: Optional[logging.Logger] = None) -> None:
        super().__init__(engine=engine)
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def save_optimization_logs(self, logs_df: pd.DataFrame, run_name: str) -> int:
        """Save optimization run logs to database."""
        if logs_df.empty or self.engine is None:
            return 0

        # Ensures engine schema compatibility and saves
        try:
            logs_df["run_name"] = run_name
            logs_df.to_sql("optimization_run_logs", con=self.engine, if_exists="append", index=False)
            self.logger.info("Saved %d optimization log entries for run %s", len(logs_df), run_name)
            return len(logs_df)
        except Exception as exc:
            self.logger.warning("Failed to save optimization logs to DB: %s", exc)
            return 0
