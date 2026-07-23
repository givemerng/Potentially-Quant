"""Experiment Tracker for hashing configurations, datasets, and git metadata."""

from __future__ import annotations
import hashlib
import json
import logging
import subprocess
from typing import Dict, Any, Optional


class ExperimentTracker:
    """Computes git commit hashes, config hashes, and logs experiment runs for 100% reproducibility."""

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def get_git_hash(self) -> str:
        """Get current git HEAD hash."""
        try:
            res = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, check=True
            )
            return res.stdout.strip()
        except Exception:
            return "unknown_git_hash"

    def compute_config_hash(self, config_dict: Dict[str, Any]) -> str:
        """Compute SHA256 hash of configuration parameters."""
        try:
            dumped = json.dumps(config_dict, sort_keys=True, default=str)
            return hashlib.sha256(dumped.encode("utf-8")).hexdigest()[:12]
        except Exception:
            return "unknown_config_hash"

    def compute_dataset_hash(self, dataset_summary_str: str) -> str:
        """Compute SHA256 hash of dataset state."""
        return hashlib.sha256(dataset_summary_str.encode("utf-8")).hexdigest()[:12]

    def create_experiment_record(
        self,
        experiment_name: str,
        config_dict: Dict[str, Any],
        dataset_summary: str = "",
    ) -> Dict[str, Any]:
        """Create complete experiment metadata record."""
        git_hash = self.get_git_hash()
        config_hash = self.compute_config_hash(config_dict)
        dataset_hash = self.compute_dataset_hash(dataset_summary)

        return {
            "experiment_id": f"{experiment_name}_{config_hash}",
            "experiment_name": experiment_name,
            "git_hash": git_hash,
            "config_hash": config_hash,
            "dataset_hash": dataset_hash,
            "config": config_dict,
        }
