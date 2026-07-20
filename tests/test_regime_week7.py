from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine

from src.data.db import metadata
from src.regime.detector import MarketRegimeDetector
from src.regime.visualization import (
    plot_spx_with_regimes,
    plot_transition_matrix,
    plot_regime_feature_profiles,
)


@pytest.fixture
def mock_prices_df() -> pd.DataFrame:
    """Generate 600 days of mock prices for 3 tickers."""
    dates = pd.date_range("2020-01-01", periods=600, freq="B")
    data = []
    np.random.seed(42)

    for ticker in ["AAPL", "MSFT", "GOOGL"]:
        init_price = 100.0
        returns = np.random.normal(0.0005, 0.015, len(dates))
        prices = init_price * np.exp(np.cumsum(returns))
        for d, p in zip(dates, prices):
            data.append({"date": d, "ticker": ticker, "close": float(p)})

    return pd.DataFrame(data)


@pytest.fixture
def mock_fred_df(mock_prices_df) -> pd.DataFrame:
    """Generate mock FRED macro series."""
    dates = mock_prices_df["date"].unique()
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "date": dates,
            "VIXCLS": 15.0 + np.random.normal(0, 3, len(dates)),
            "DGS10": 2.5 + np.random.normal(0, 0.2, len(dates)),
            "DGS2": 1.0 + np.random.normal(0, 0.2, len(dates)),
            "BAMLH0A0HYM2": 3.5 + np.random.normal(0, 0.5, len(dates)),
            "GDP": 20000.0 + np.cumsum(np.random.normal(10, 2, len(dates))),
        }
    )
    return df


def test_build_feature_matrix(mock_prices_df, mock_fred_df):
    detector = MarketRegimeDetector()
    features = detector.build_feature_matrix(mock_prices_df, mock_fred_df)

    assert not features.empty
    assert "spx_momentum" in features.columns
    assert "realized_vol" in features.columns
    assert "vix_level" in features.columns
    assert "vix_change" in features.columns
    assert "yield_curve_slope" in features.columns
    assert "credit_spread" in features.columns
    assert "gdp_growth" in features.columns
    assert not features.isnull().any().any()


def test_fit_and_probabilities(mock_prices_df, mock_fred_df):
    detector = MarketRegimeDetector(n_components=4, random_state=42)
    features = detector.build_feature_matrix(mock_prices_df, mock_fred_df)

    state_series, prob_df = detector.fit(features)

    assert len(state_series) == len(features)
    assert prob_df.shape == (len(features), 4)

    # Check soft probabilities sum to 1.0 per row
    row_sums = prob_df.sum(axis=1)
    np.testing.assert_allclose(row_sums.values, 1.0, rtol=1e-4)

    # Check hard states are integers between 0 and 3
    assert set(state_series.unique()).issubset({0, 1, 2, 3})


def test_select_optimal_states(mock_prices_df, mock_fred_df):
    detector = MarketRegimeDetector(random_state=42)
    features = detector.build_feature_matrix(mock_prices_df, mock_fred_df)

    res = detector.select_optimal_states(features, max_states=5)
    assert "bic_scores" in res
    assert "optimal_k" in res
    assert len(res["bic_scores"]) == 4  # k in 2, 3, 4, 5
    assert 2 <= res["optimal_k"] <= 5


def test_economic_labeling(mock_prices_df, mock_fred_df):
    detector = MarketRegimeDetector(n_components=4, random_state=42)
    features = detector.build_feature_matrix(mock_prices_df, mock_fred_df)
    state_series, _ = detector.fit(features)

    labels = detector.regime_labels
    assert len(labels) == 4
    label_values = list(labels.values())
    assert "Bear / High-Vol" in label_values
    assert "Bull Market" in label_values


def test_transition_matrix(mock_prices_df, mock_fred_df):
    detector = MarketRegimeDetector(n_components=4, random_state=42)
    features = detector.build_feature_matrix(mock_prices_df, mock_fred_df)
    detector.fit(features)

    trans_df, dur_series = detector.compute_transition_matrix()

    assert trans_df.shape == (4, 4)
    assert len(dur_series) == 4
    # Row probabilities in transition matrix sum to 1
    row_sums = trans_df.sum(axis=1)
    np.testing.assert_allclose(row_sums.values, 1.0, rtol=1e-4)


def test_save_to_db(mock_prices_df, mock_fred_df):
    engine = create_engine("sqlite:///:memory:")
    metadata.create_all(engine)

    detector = MarketRegimeDetector(n_components=4, random_state=42)
    features = detector.build_feature_matrix(mock_prices_df, mock_fred_df)
    state_series, prob_df = detector.fit(features)

    regime_df = prob_df.copy()
    regime_df["regime_id"] = state_series
    regime_df["regime_label"] = state_series.map(detector.regime_labels)

    count = detector.save_to_db(engine, regime_df)
    assert count == len(regime_df)

    with engine.connect() as conn:
        df_read = pd.read_sql("SELECT * FROM market_regimes", conn)
        assert len(df_read) == len(regime_df)
        assert "regime_label" in df_read.columns
        assert "prob_0" in df_read.columns


def test_visualization_helpers(mock_prices_df, mock_fred_df):
    detector = MarketRegimeDetector(n_components=4, random_state=42)
    features = detector.build_feature_matrix(mock_prices_df, mock_fred_df)
    state_series, prob_df = detector.fit(features)

    regime_df = prob_df.copy()
    regime_df["regime_id"] = state_series

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        fig1 = plot_spx_with_regimes(mock_prices_df, regime_df, detector.regime_labels, save_path=tmp_path / "spx.png")
        assert (tmp_path / "spx.png").exists()

        trans_df, _ = detector.compute_transition_matrix()
        fig2 = plot_transition_matrix(trans_df, save_path=tmp_path / "trans.png")
        assert (tmp_path / "trans.png").exists()

        fig3 = plot_regime_feature_profiles(features, state_series, detector.regime_labels, save_path=tmp_path / "profiles.png")
        assert (tmp_path / "profiles.png").exists()
