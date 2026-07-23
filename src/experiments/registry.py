"""Experiment Registry for logging experiment outputs and metrics."""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional


class ExperimentRegistry:
    """Manages persistence of experiment runs to disk / database."""

    def __init__(self, registry_dir: str | Path = "data/experiments", logger: Optional[logging.Logger] = None) -> None:
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def save_experiment(self, record: Dict[str, Any], metrics: Dict[str, Any]) -> None:
        """Save experiment record and metrics to JSON file."""
        exp_id = record.get("experiment_id", "exp_unknown")
        payload = {**record, "metrics": metrics}

        file_path = self.registry_dir / f"{exp_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)

        self.logger.info("Registered experiment %s to %s", exp_id, file_path)
