"""Abstract base class for all factor combination strategies."""

from __future__ import annotations

import abc
import hashlib
import json
import logging
import uuid
from datetime import date as date_type
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sqlalchemy.engine import Engine

from src.data.db import (
    combination_runs_table,
    composite_scores_table,
    combination_weights_table,
    combination_metrics_table,
    upsert_rows,
)


class BaseComposite(abc.ABC):
    """Common interface for all factor combination methods.

    Subclasses must implement ``fit`` and ``predict``.  The base class provides
    helpers for experiment tracking, weight persistence, and DB writes.
    """

    method_name: str = "base"

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.run_id: str = ""
        self._factor_names: List[str] = []

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def fit(
        self,
        factor_scores: pd.DataFrame,
        forward_returns: pd.Series,
        eval_date: pd.Timestamp,
        config: dict,
    ) -> None:
        """Fit the combination model using *only* historical data up to ``eval_date``.

        Args:
            factor_scores: Wide DataFrame (dates × factors) of neutralized z-scores
                           for a single date's cross-section, or the full historical panel.
            forward_returns: Series of next-period simple returns aligned with the
                             cross-section (ticker-indexed).
            eval_date: The current evaluation date (only data ≤ this date may be used).
            config: Week6Config parameters as a dictionary.
        """

    @abc.abstractmethod
    def predict(
        self,
        factor_scores: pd.DataFrame,
        eval_date: pd.Timestamp,
    ) -> pd.Series:
        """Generate composite scores for the cross-section at ``eval_date``.

        Args:
            factor_scores: Wide DataFrame with one row per ticker, one column
                           per factor (values = neutralized z-scores).
            eval_date: The evaluation date.

        Returns:
            pd.Series indexed by ticker with composite scores.
        """

    # ------------------------------------------------------------------
    # Experiment tracking helpers
    # ------------------------------------------------------------------

    def _generate_run_id(self, config: dict) -> str:
        """Create a unique, reproducible run identifier."""
        config_str = json.dumps(config, sort_keys=True, default=str)
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]
        unique_suffix = uuid.uuid4().hex[:8]
        self.run_id = f"{self.method_name}_{config_hash}_{unique_suffix}"
        return self.run_id

    def _config_hash(self, config: dict) -> str:
        config_str = json.dumps(config, sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def save_run_metadata(
        self,
        engine: Engine,
        config: dict,
        train_start: Optional[date_type] = None,
        train_end: Optional[date_type] = None,
        test_start: Optional[date_type] = None,
        test_end: Optional[date_type] = None,
    ) -> None:
        """Persist an experiment run record to the ``combination_runs`` table."""
        record = {
            "run_id": self.run_id,
            "method": self.method_name,
            "config_hash": self._config_hash(config),
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "created_at": date_type.today(),
        }
        upsert_rows(engine, combination_runs_table, [record])
        self.logger.info("Saved run metadata: run_id=%s method=%s", self.run_id, self.method_name)

    def save_composite_scores(
        self,
        engine: Engine,
        scores: pd.DataFrame,
    ) -> int:
        """Persist composite scores to the ``composite_scores`` table.

        Args:
            engine: Database engine.
            scores: Long DataFrame with columns [date, ticker, composite_score].

        Returns:
            Number of rows upserted.
        """
        records = []
        for _, row in scores.iterrows():
            records.append({
                "date": row["date"] if isinstance(row["date"], date_type) else pd.Timestamp(row["date"]).date(),
                "ticker": row["ticker"],
                "method": self.method_name,
                "run_id": self.run_id,
                "composite_score": float(row["composite_score"]) if pd.notna(row["composite_score"]) else None,
            })
        if records:
            return upsert_rows(engine, composite_scores_table, records)
        return 0

    def save_factor_weights(
        self,
        engine: Engine,
        weights_df: pd.DataFrame,
    ) -> int:
        """Persist rolling factor weights to the ``combination_weights`` table.

        Args:
            engine: Database engine.
            weights_df: Long DataFrame with columns [date, factor_name, weight].

        Returns:
            Number of rows upserted.
        """
        records = []
        for _, row in weights_df.iterrows():
            records.append({
                "date": row["date"] if isinstance(row["date"], date_type) else pd.Timestamp(row["date"]).date(),
                "factor_name": row["factor_name"],
                "method": self.method_name,
                "run_id": self.run_id,
                "weight": float(row["weight"]) if pd.notna(row["weight"]) else None,
            })
        if records:
            return upsert_rows(engine, combination_weights_table, records)
        return 0

    def save_metrics(
        self,
        engine: Engine,
        metrics: Dict[str, float],
        scope: str = "full",
    ) -> int:
        """Persist performance metrics to the ``combination_metrics`` table.

        Args:
            engine: Database engine.
            metrics: Dict mapping metric_name → value.
            scope: Either ``'full'`` or ``'oos'``.

        Returns:
            Number of rows upserted.
        """
        records = [
            {
                "run_id": self.run_id,
                "method": self.method_name,
                "metric_name": name,
                "metric_scope": scope,
                "value": float(val) if pd.notna(val) else None,
            }
            for name, val in metrics.items()
        ]
        if records:
            return upsert_rows(engine, combination_metrics_table, records)
        return 0
