from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Week2Config
from src.factors.analysis import Week2AnalysisPipeline


def _make_returns() -> pd.DataFrame:
    dates = pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"])
    rows = [
        {"date": dates[0], "ticker": "AAA", "simple_return": 0.02, "log_return": np.log1p(0.02)},
        {"date": dates[1], "ticker": "AAA", "simple_return": 0.01, "log_return": np.log1p(0.01)},
        {"date": dates[2], "ticker": "AAA", "simple_return": -0.03, "log_return": np.log1p(-0.03)},
        {"date": dates[0], "ticker": "BBB", "simple_return": 0.01, "log_return": np.log1p(0.01)},
        {"date": dates[1], "ticker": "BBB", "simple_return": 0.04, "log_return": np.log1p(0.04)},
        {"date": dates[2], "ticker": "BBB", "simple_return": 0.05, "log_return": np.log1p(0.05)},
    ]
    df = pd.DataFrame(rows)
    return df.set_index(["date", "ticker"]).sort_index()


def _make_membership() -> pd.DataFrame:
    rows = [
        {"date": "2020-01-30", "ticker": "AAA", "in_universe": True},
        {"date": "2020-02-28", "ticker": "AAA", "in_universe": False},
        {"date": "2020-03-30", "ticker": "AAA", "in_universe": True},
        {"date": "2020-01-30", "ticker": "BBB", "in_universe": True},
        {"date": "2020-02-28", "ticker": "BBB", "in_universe": True},
        {"date": "2020-03-30", "ticker": "BBB", "in_universe": True},
    ]
    return pd.DataFrame(rows)


def test_analysis_pipeline_masks_out_of_universe_rows() -> None:
    pipeline = Week2AnalysisPipeline()
    summary = pipeline.run(
        returns_df=_make_returns(),
        membership_df=_make_membership(),
        config=Week2Config(save_artifacts=False),
    )
    assert summary.return_matrix_shape == (3, 2)
    assert summary.trimmed_matrix_non_null == 5
    assert summary.cross_sectional_dates == 3


def test_analysis_pipeline_writes_artifacts(tmp_path: Path) -> None:
    pipeline = Week2AnalysisPipeline()
    config = Week2Config(save_artifacts=True, artifact_dir=str(tmp_path))
    summary = pipeline.run(
        returns_df=_make_returns(),
        membership_df=_make_membership(),
        config=config,
    )
    assert summary.artifacts_written == 6
    assert (tmp_path / "return_matrix_cleaned.csv").exists()
    assert (tmp_path / "return_matrix_trimmed.csv").exists()
    assert (tmp_path / "cross_sectional_stats.csv").exists()
    assert (tmp_path / "annual_stats.csv").exists()
