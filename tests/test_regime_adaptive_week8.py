from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.combination.regime_adaptive import RegimeAdaptiveComposite
from src.combination.regime_reporting import Week8Reporter
from src.regime.adaptive_weights import AdaptiveWeightGenerator
from src.regime.bayesian_updater import BayesianUpdater
from src.regime.factor_analysis import RegimeFactorAnalyzer


@pytest.fixture
def synthetic_factor_and_return_data():
    """Generates synthetic factor scores, returns, and HMM regimes for testing."""
    dates = pd.date_range(start="2020-01-31", periods=12, freq="ME")
    tickers = ["AAPL", "MSFT", "AMZN", "GOOGL"]
    factors = ["Momentum", "Value", "Quality"]

    factor_records = []
    return_records = []
    regime_records = []

    np.random.seed(42)

    for i, d in enumerate(dates):
        # Regime ID cycles between 0, 1, 2, 3
        reg_id = i % 4
        reg_label = ["Bull", "Bear", "RateShock", "Neutral"][reg_id]

        p0 = 0.7 if reg_id == 0 else 0.1
        p1 = 0.7 if reg_id == 1 else 0.1
        p2 = 0.7 if reg_id == 2 else 0.1
        p3 = 1.0 - (p0 + p1 + p2)

        regime_records.append({
            "date": d,
            "regime_id": reg_id,
            "regime_label": reg_label,
            "prob_0": p0,
            "prob_1": p1,
            "prob_2": p2,
            "prob_3": p3,
        })

        for t in tickers:
            fwd_ret = np.random.normal(0.01, 0.05)
            return_records.append({"date": d, "ticker": t, "simple_return": fwd_ret})

            for f in factors:
                score = np.random.normal(0.0, 1.0)
                factor_records.append({
                    "date": d,
                    "ticker": t,
                    "factor_name": f,
                    "final_score": score,
                })

    factors_df = pd.DataFrame(factor_records).set_index(["date", "ticker"])
    returns_df = pd.DataFrame(return_records).set_index(["date", "ticker"])
    regimes_df = pd.DataFrame(regime_records).set_index("date")

    return factors_df, returns_df, regimes_df


def test_regime_factor_analyzer(synthetic_factor_and_return_data):
    factors_df, returns_df, regimes_df = synthetic_factor_and_return_data
    analyzer = RegimeFactorAnalyzer(ic_lookback_months=6, min_regime_samples=1)

    ic_df = analyzer.compute_daily_cross_sectional_ic(factors_df, returns_df)
    assert not ic_df.empty
    assert len(ic_df.columns) == 3

    summary_matrix, rolling_df = analyzer.compute_regime_conditional_stats(ic_df, regimes_df)
    assert not summary_matrix.empty
    assert not rolling_df.empty
    assert "rolling_icir" in rolling_df.columns


def test_bayesian_updater(synthetic_factor_and_return_data):
    factors_df, returns_df, regimes_df = synthetic_factor_and_return_data
    analyzer = RegimeFactorAnalyzer(ic_lookback_months=6, min_regime_samples=1)
    ic_df = analyzer.compute_daily_cross_sectional_ic(factors_df, returns_df)
    _, rolling_df = analyzer.compute_regime_conditional_stats(ic_df, regimes_df)

    updater = BayesianUpdater(prior_weight=0.3, decay_halflife=6.0)
    posteriors_df = updater.update_posteriors(rolling_df, ic_df)

    assert "posterior_ic" in posteriors_df.columns
    assert "posterior_variance" in posteriors_df.columns
    assert "effective_sample_size" in posteriors_df.columns
    assert len(posteriors_df) == len(rolling_df)


def test_adaptive_weight_generator(synthetic_factor_and_return_data):
    factors_df, returns_df, regimes_df = synthetic_factor_and_return_data
    analyzer = RegimeFactorAnalyzer(ic_lookback_months=6, min_regime_samples=1)
    ic_df = analyzer.compute_daily_cross_sectional_ic(factors_df, returns_df)
    _, rolling_df = analyzer.compute_regime_conditional_stats(ic_df, regimes_df)
    updater = BayesianUpdater(prior_weight=0.3, decay_halflife=6.0)
    posteriors_df = updater.update_posteriors(rolling_df, ic_df)

    generator = AdaptiveWeightGenerator(max_factor_weight=0.5, min_weight_threshold=0.01)
    weights_df, direction_df, metadata_df = generator.generate_weights(posteriors_df, regimes_df)

    assert not weights_df.empty
    assert not direction_df.empty
    # Weights should sum to 1.0 per date
    for d in weights_df.index:
        row_sum = weights_df.loc[d].sum()
        assert pytest.approx(row_sum, 1e-4) == 1.0

    # Directions should be in {-1.0, 1.0}
    for col in direction_df.columns:
        vals = direction_df[col].unique()
        for v in vals:
            assert v in [-1.0, 1.0]


def test_regime_adaptive_composite(synthetic_factor_and_return_data):
    factors_df, returns_df, regimes_df = synthetic_factor_and_return_data
    composite = RegimeAdaptiveComposite()
    config = {
        "ic_lookback_months": 6,
        "decay_halflife": 6.0,
        "prior_weight": 0.3,
        "min_regime_samples": 1,
        "max_factor_weight": 0.5,
        "min_weight_threshold": 0.01,
        "oos_start_date": "2020-06-01",
        "oos_end_date": "2020-12-31",
    }

    comp_scores_df, weights_df, posteriors_df, summary_ic_matrix = composite.fit_predict_full_pipeline(
        factors_df=factors_df,
        returns_df=returns_df.reset_index(),
        regimes_df=regimes_df,
        config=config,
        engine=None,
    )

    assert not comp_scores_df.empty
    assert not weights_df.empty
    assert not posteriors_df.empty
    assert not summary_ic_matrix.empty
    assert "composite_score" in comp_scores_df.columns


def test_week8_reporter(synthetic_factor_and_return_data):
    factors_df, returns_df, regimes_df = synthetic_factor_and_return_data
    composite = RegimeAdaptiveComposite()
    config = {
        "ic_lookback_months": 6,
        "decay_halflife": 6.0,
        "prior_weight": 0.3,
        "min_regime_samples": 1,
        "max_factor_weight": 0.5,
        "min_weight_threshold": 0.01,
    }

    comp_scores_df, weights_df, posteriors_df, summary_ic_matrix = composite.fit_predict_full_pipeline(
        factors_df=factors_df,
        returns_df=returns_df.reset_index(),
        regimes_df=regimes_df,
        config=config,
        engine=None,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        reporter = Week8Reporter(output_dir=tmpdir)
        heatmap_path = reporter.generate_ic_heatmap(summary_ic_matrix)
        weights_path = reporter.generate_weights_plot(weights_df)
        post_path = reporter.generate_posterior_ic_evolution(posteriors_df)
        weights_hm_path = reporter.generate_weights_heatmap(weights_df)

        assert heatmap_path.exists()
        assert weights_path.exists()
        assert post_path.exists()
        assert weights_hm_path.exists()

        comp_df = pd.DataFrame([
            {"method": "equal_weight", "full_sharpe": 0.5, "oos_sharpe": 0.4},
            {"method": "regime_adaptive", "full_sharpe": 0.9, "oos_sharpe": 0.85},
        ])
        csv_p, md_p = reporter.generate_composite_comparison_report(comp_df)
        assert csv_p.exists()
        assert md_p.exists()
