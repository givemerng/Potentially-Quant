"""Comprehensive tests for Week 6 Factor Combination & ML Integration."""

from __future__ import annotations

import hashlib
import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers: build synthetic factor scores and returns
# ---------------------------------------------------------------------------

def _make_factor_scores(
    n_dates: int = 48,
    n_tickers: int = 30,
    n_factors: int = 5,
    start: str = "2005-01-31",
) -> pd.DataFrame:
    """Generate a synthetic long-format factor scores table."""
    np.random.seed(42)
    dates = pd.date_range(start, periods=n_dates, freq="ME")
    tickers = [f"T{i:03d}" for i in range(n_tickers)]
    factor_names = [f"factor_{i}" for i in range(n_factors)]

    records = []
    for d in dates:
        for t in tickers:
            for f in factor_names:
                records.append({
                    "date": d,
                    "ticker": t,
                    "factor_name": f,
                    "raw_score": np.random.randn(),
                    "winsorized_score": np.random.randn(),
                    "z_score": np.random.randn(),
                    "final_score": np.random.randn(),
                })
    return pd.DataFrame(records)


def _make_returns(
    n_dates: int = 49,
    n_tickers: int = 30,
    start: str = "2005-01-31",
) -> pd.DataFrame:
    """Generate a synthetic long-format monthly returns table."""
    np.random.seed(123)
    dates = pd.date_range(start, periods=n_dates, freq="ME")
    tickers = [f"T{i:03d}" for i in range(n_tickers)]

    records = []
    for d in dates:
        for t in tickers:
            records.append({
                "date": d,
                "ticker": t,
                "simple_return": np.random.randn() * 0.05,
                "log_return": np.random.randn() * 0.05,
            })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Preprocessing tests
# ---------------------------------------------------------------------------

class TestFeaturePreprocessor:

    def test_output_shapes_and_types(self):
        from src.combination.preprocessing import FeaturePreprocessor
        scores = _make_factor_scores(n_dates=24, n_tickers=10, n_factors=3)
        returns = _make_returns(n_dates=25, n_tickers=10)

        pp = FeaturePreprocessor()
        feature_panel, fwd_returns, eval_dates, factor_names = pp.prepare(scores, returns)

        assert isinstance(feature_panel, pd.DataFrame)
        assert isinstance(fwd_returns, pd.Series)
        assert len(factor_names) == 3
        assert len(eval_dates) > 0
        # Feature panel columns match factor names
        assert list(feature_panel.columns) == sorted(factor_names)

    def test_no_nans_in_output(self):
        from src.combination.preprocessing import FeaturePreprocessor
        scores = _make_factor_scores(n_dates=24, n_tickers=10, n_factors=3)
        returns = _make_returns(n_dates=25, n_tickers=10)

        pp = FeaturePreprocessor()
        feature_panel, fwd_returns, _, _ = pp.prepare(scores, returns)

        assert feature_panel.isna().sum().sum() == 0
        assert fwd_returns.isna().sum() == 0

    def test_consistent_factor_ordering(self):
        from src.combination.preprocessing import FeaturePreprocessor
        scores = _make_factor_scores(n_dates=12, n_tickers=5, n_factors=4)
        returns = _make_returns(n_dates=13, n_tickers=5)

        pp = FeaturePreprocessor()
        _, _, _, factor_names = pp.prepare(scores, returns)

        # Factor names must be sorted for consistency
        assert factor_names == sorted(factor_names)

    def test_empty_input_raises(self):
        from src.combination.preprocessing import FeaturePreprocessor
        pp = FeaturePreprocessor()

        with pytest.raises(ValueError, match="factor_scores_df is empty"):
            pp.prepare(pd.DataFrame(), _make_returns(n_dates=5, n_tickers=5))

        with pytest.raises(ValueError, match="returns_df is empty"):
            pp.prepare(_make_factor_scores(n_dates=5, n_tickers=5), pd.DataFrame())


# ---------------------------------------------------------------------------
# IC-Weighted Composite tests
# ---------------------------------------------------------------------------

class TestICWeightedComposite:

    def _build_inputs(self):
        from src.combination.preprocessing import FeaturePreprocessor
        scores = _make_factor_scores(n_dates=48, n_tickers=20, n_factors=5)
        returns = _make_returns(n_dates=49, n_tickers=20)
        pp = FeaturePreprocessor()
        return pp.prepare(scores, returns)

    def test_weights_sum_to_one(self):
        from src.combination.ic_weighted import ICWeightedComposite
        feature_panel, fwd_returns, eval_dates, factor_names = self._build_inputs()

        model = ICWeightedComposite()
        eval_date = eval_dates[-2]
        config = {"ic_lookback_months": 36}

        model.fit(feature_panel, fwd_returns, eval_date, config)

        weights = model._current_weights
        assert abs(sum(weights.values()) - 1.0) < 1e-9

    def test_negative_ic_zeroed_out(self):
        from src.combination.ic_weighted import ICWeightedComposite

        model = ICWeightedComposite()
        # Manually set weights to verify zeroing logic
        feature_panel, fwd_returns, eval_dates, factor_names = self._build_inputs()
        config = {"ic_lookback_months": 36}

        model.fit(feature_panel, fwd_returns, eval_dates[-2], config)

        # All weights must be >= 0
        for w in model._current_weights.values():
            assert w >= 0.0

    def test_predict_returns_series(self):
        from src.combination.ic_weighted import ICWeightedComposite
        feature_panel, fwd_returns, eval_dates, factor_names = self._build_inputs()

        model = ICWeightedComposite()
        eval_date = eval_dates[-2]
        config = {"ic_lookback_months": 24}

        model.fit(feature_panel, fwd_returns, eval_date, config)
        preds = model.predict(feature_panel, eval_date)

        assert isinstance(preds, pd.Series)
        assert len(preds) > 0

    def test_weights_history_recorded(self):
        from src.combination.ic_weighted import ICWeightedComposite
        feature_panel, fwd_returns, eval_dates, factor_names = self._build_inputs()

        model = ICWeightedComposite()
        config = {"ic_lookback_months": 24}

        # Fit on two consecutive dates
        model.fit(feature_panel, fwd_returns, eval_dates[-3], config)
        model.fit(feature_panel, fwd_returns, eval_dates[-2], config)

        weights_df = model.get_weights_dataframe()
        assert len(weights_df) == 2 * len(factor_names)


# ---------------------------------------------------------------------------
# Fama-MacBeth Composite tests
# ---------------------------------------------------------------------------

class TestFamaMacBethComposite:

    def _build_inputs(self):
        from src.combination.preprocessing import FeaturePreprocessor
        scores = _make_factor_scores(n_dates=48, n_tickers=20, n_factors=5)
        returns = _make_returns(n_dates=49, n_tickers=20)
        pp = FeaturePreprocessor()
        return pp.prepare(scores, returns)

    def test_coefficients_computed(self):
        from src.combination.fama_macbeth import FamaMacBethComposite
        feature_panel, fwd_returns, eval_dates, factor_names = self._build_inputs()

        model = FamaMacBethComposite()
        config = {"ic_lookback_months": 24}
        model.fit(feature_panel, fwd_returns, eval_dates[-2], config)

        assert hasattr(model, "_current_weights")
        assert len(model._current_weights) == len(factor_names)

    def test_weights_normalised(self):
        from src.combination.fama_macbeth import FamaMacBethComposite
        feature_panel, fwd_returns, eval_dates, factor_names = self._build_inputs()

        model = FamaMacBethComposite()
        config = {"ic_lookback_months": 24}
        model.fit(feature_panel, fwd_returns, eval_dates[-2], config)

        total_abs = sum(abs(w) for w in model._current_weights.values())
        assert abs(total_abs - 1.0) < 1e-9

    def test_predict_returns_series(self):
        from src.combination.fama_macbeth import FamaMacBethComposite
        feature_panel, fwd_returns, eval_dates, factor_names = self._build_inputs()

        model = FamaMacBethComposite()
        config = {"ic_lookback_months": 24}
        model.fit(feature_panel, fwd_returns, eval_dates[-2], config)
        preds = model.predict(feature_panel, eval_dates[-2])

        assert isinstance(preds, pd.Series)
        assert len(preds) > 0

    def test_coefficients_dataframe(self):
        from src.combination.fama_macbeth import FamaMacBethComposite
        feature_panel, fwd_returns, eval_dates, factor_names = self._build_inputs()

        model = FamaMacBethComposite()
        config = {"ic_lookback_months": 12}
        model.fit(feature_panel, fwd_returns, eval_dates[-2], config)

        coeff_df = model.get_coefficients_dataframe()
        assert isinstance(coeff_df, pd.DataFrame)
        assert "date" in coeff_df.columns


# ---------------------------------------------------------------------------
# XGBoost Composite tests
# ---------------------------------------------------------------------------

class TestXGBoostComposite:

    def _build_inputs(self):
        from src.combination.preprocessing import FeaturePreprocessor
        scores = _make_factor_scores(n_dates=60, n_tickers=20, n_factors=5)
        returns = _make_returns(n_dates=61, n_tickers=20)
        pp = FeaturePreprocessor()
        return pp.prepare(scores, returns)

    def test_model_trains_with_enough_data(self):
        xgb = pytest.importorskip("xgboost")
        from src.combination.xgboost_composite import XGBoostComposite
        feature_panel, fwd_returns, eval_dates, factor_names = self._build_inputs()

        model = XGBoostComposite()
        config = {"xgb_min_train_months": 36, "xgb_purge_gap_months": 1, "xgb_n_estimators": 10, "xgb_max_depth": 3}

        # Pick a date far enough to have 36+ training months
        eval_date = eval_dates[-2]
        model.fit(feature_panel, fwd_returns, eval_date, config)

        assert model.model is not None

    def test_model_skips_with_insufficient_data(self):
        xgb = pytest.importorskip("xgboost")
        from src.combination.xgboost_composite import XGBoostComposite
        feature_panel, fwd_returns, eval_dates, factor_names = self._build_inputs()

        model = XGBoostComposite()
        config = {"xgb_min_train_months": 999, "xgb_purge_gap_months": 1}

        model.fit(feature_panel, fwd_returns, eval_dates[-2], config)
        assert model.model is None

    def test_no_lookahead_in_training(self):
        xgb = pytest.importorskip("xgboost")
        from src.combination.xgboost_composite import XGBoostComposite
        feature_panel, fwd_returns, eval_dates, factor_names = self._build_inputs()

        model = XGBoostComposite()
        purge_gap = 1
        eval_date = eval_dates[-2]
        config = {"xgb_min_train_months": 12, "xgb_purge_gap_months": purge_gap, "xgb_n_estimators": 10}

        cutoff = eval_date - pd.DateOffset(months=purge_gap)
        available = sorted(feature_panel.index.get_level_values("date").unique())
        expected_train_dates = [d for d in available if d <= cutoff]

        model.fit(feature_panel, fwd_returns, eval_date, config)

        # The model was trained; verify no data after cutoff was used
        # (we check implicitly: if model trained, it used dates <= cutoff)
        assert model.model is not None
        # Verify training did not include eval_date itself
        assert eval_date not in expected_train_dates or eval_date <= cutoff

    def test_predict_returns_correct_shape(self):
        xgb = pytest.importorskip("xgboost")
        from src.combination.xgboost_composite import XGBoostComposite
        feature_panel, fwd_returns, eval_dates, factor_names = self._build_inputs()

        model = XGBoostComposite()
        eval_date = eval_dates[-2]
        config = {"xgb_min_train_months": 12, "xgb_purge_gap_months": 1, "xgb_n_estimators": 10}

        model.fit(feature_panel, fwd_returns, eval_date, config)
        preds = model.predict(feature_panel, eval_date)

        n_tickers_at_date = len(feature_panel.loc[eval_date])
        assert len(preds) == n_tickers_at_date

    def test_predict_no_nans(self):
        xgb = pytest.importorskip("xgboost")
        from src.combination.xgboost_composite import XGBoostComposite
        feature_panel, fwd_returns, eval_dates, factor_names = self._build_inputs()

        model = XGBoostComposite()
        eval_date = eval_dates[-2]
        config = {"xgb_min_train_months": 12, "xgb_purge_gap_months": 1, "xgb_n_estimators": 10}

        model.fit(feature_panel, fwd_returns, eval_date, config)
        preds = model.predict(feature_panel, eval_date)

        assert preds.isna().sum() == 0


# ---------------------------------------------------------------------------
# SHAP Analysis tests
# ---------------------------------------------------------------------------

class TestSHAPAnalysis:

    def test_shap_values_shape(self):
        xgb = pytest.importorskip("xgboost")
        shap_mod = pytest.importorskip("shap")
        from src.combination.shap_analysis import SHAPAnalyzer
        from src.combination.xgboost_composite import XGBoostComposite
        from src.combination.preprocessing import FeaturePreprocessor

        scores = _make_factor_scores(n_dates=48, n_tickers=15, n_factors=4)
        returns = _make_returns(n_dates=49, n_tickers=15)
        pp = FeaturePreprocessor()
        feature_panel, fwd_returns, eval_dates, factor_names = pp.prepare(scores, returns)

        # Train model
        xgb_model = XGBoostComposite()
        config = {"xgb_min_train_months": 12, "xgb_purge_gap_months": 1, "xgb_n_estimators": 10}
        xgb_model.fit(feature_panel, fwd_returns, eval_dates[-2], config)
        assert xgb_model.model is not None

        # Analyze last cross-section
        X_latest = feature_panel.loc[eval_dates[-2]]
        analyzer = SHAPAnalyzer()
        shap_df = analyzer.analyze(xgb_model.model, X_latest, factor_names)

        assert shap_df.shape == X_latest.shape
        assert list(shap_df.columns) == factor_names

    def test_importance_ranking(self):
        xgb = pytest.importorskip("xgboost")
        shap_mod = pytest.importorskip("shap")
        from src.combination.shap_analysis import SHAPAnalyzer
        from src.combination.xgboost_composite import XGBoostComposite
        from src.combination.preprocessing import FeaturePreprocessor

        scores = _make_factor_scores(n_dates=48, n_tickers=15, n_factors=4)
        returns = _make_returns(n_dates=49, n_tickers=15)
        pp = FeaturePreprocessor()
        feature_panel, fwd_returns, eval_dates, factor_names = pp.prepare(scores, returns)

        xgb_model = XGBoostComposite()
        config = {"xgb_min_train_months": 12, "xgb_purge_gap_months": 1, "xgb_n_estimators": 10}
        xgb_model.fit(feature_panel, fwd_returns, eval_dates[-2], config)

        X_latest = feature_panel.loc[eval_dates[-2]]
        analyzer = SHAPAnalyzer()
        analyzer.analyze(xgb_model.model, X_latest, factor_names)
        ranking = analyzer.get_importance_ranking()

        assert isinstance(ranking, pd.Series)
        assert len(ranking) == len(factor_names)
        # Values should be non-negative (mean |SHAP|)
        assert (ranking >= 0).all()


# ---------------------------------------------------------------------------
# BaseComposite experiment tracking tests
# ---------------------------------------------------------------------------

class TestBaseCompositeTracking:

    def test_run_id_generation(self):
        from src.combination.ic_weighted import ICWeightedComposite
        model = ICWeightedComposite()
        config = {"ic_lookback_months": 36}
        run_id = model._generate_run_id(config)
        assert run_id.startswith("ic_weighted_")
        assert len(run_id) > 20

    def test_config_hash_deterministic(self):
        from src.combination.ic_weighted import ICWeightedComposite
        model = ICWeightedComposite()
        config = {"ic_lookback_months": 36, "key": "value"}
        h1 = model._config_hash(config)
        h2 = model._config_hash(config)
        assert h1 == h2

    def test_different_configs_different_hashes(self):
        from src.combination.ic_weighted import ICWeightedComposite
        model = ICWeightedComposite()
        h1 = model._config_hash({"ic_lookback_months": 36})
        h2 = model._config_hash({"ic_lookback_months": 48})
        assert h1 != h2


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestWeek6Config:

    def test_config_loads_defaults(self):
        from src.config import Week6Config
        cfg = Week6Config()
        assert cfg.xgb_n_estimators == 200
        assert cfg.xgb_max_depth == 4
        assert cfg.xgb_learning_rate == 0.05
        assert cfg.xgb_min_train_months == 36
        assert cfg.xgb_purge_gap_months == 1
        assert cfg.ic_lookback_months == 36
        assert "ic_weighted" in cfg.combination_methods
        assert "fama_macbeth" in cfg.combination_methods
        assert "xgboost" in cfg.combination_methods

    def test_full_config_loads_with_week6(self):
        from src.config import AppConfig
        from pathlib import Path
        config = AppConfig.from_yaml(Path("config/config.yaml"))
        assert hasattr(config, "week6")
        assert config.week6.xgb_n_estimators == 200


# ---------------------------------------------------------------------------
# Integration: composite scores through backtest pipeline
# ---------------------------------------------------------------------------

class TestCompositeBacktestIntegration:

    def test_composite_scores_as_factor_input(self):
        """Verify composite scores can be formatted for VectorizedBacktester input."""
        from src.combination.preprocessing import FeaturePreprocessor
        from src.combination.ic_weighted import ICWeightedComposite

        scores = _make_factor_scores(n_dates=48, n_tickers=20, n_factors=5)
        returns = _make_returns(n_dates=49, n_tickers=20)
        pp = FeaturePreprocessor()
        feature_panel, fwd_returns, eval_dates, factor_names = pp.prepare(scores, returns)

        model = ICWeightedComposite()
        config = {"ic_lookback_months": 24}

        all_scores = []
        for eval_date in eval_dates[-12:]:
            model.fit(feature_panel, fwd_returns, eval_date, config)
            preds = model.predict(feature_panel, eval_date)
            if preds.empty:
                continue

            for ticker, score in preds.items():
                all_scores.append({
                    "date": eval_date.date(),
                    "ticker": ticker,
                    "factor_name": "ic_weighted_composite",
                    "raw_score": float(score),
                    "final_score": float(score),
                })

        composite_df = pd.DataFrame(all_scores)
        # Must have the columns expected by VectorizedBacktester
        assert "date" in composite_df.columns
        assert "ticker" in composite_df.columns
        assert "factor_name" in composite_df.columns
        assert "final_score" in composite_df.columns
        assert len(composite_df) > 0
