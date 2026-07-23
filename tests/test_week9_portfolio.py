"""Unit tests for Week 9 Portfolio Construction, Risk Modeling & Analytics Subsystem."""

import pytest
import numpy as np
import pandas as pd

from src.portfolio.risk_models.covariance import (
    LedoitWolfCovariance,
    OASCovariance,
    SampleCovariance,
    FactorCovariance,
    get_covariance_estimator,
)
from src.portfolio.risk_models.cvar import CVaRCalculator
from src.portfolio.constraints.manager import ConstraintManager
from src.portfolio.optimizers.cvar_optimizer import CVaROptimizer
from src.portfolio.optimizers.mean_variance import MeanVarianceOptimizer
from src.portfolio.optimizers.risk_parity import RiskParityOptimizer
from src.portfolio.execution.solver import OptimizationSolverManager
from src.portfolio.execution.walk_forward import WalkForwardOptimizer
from src.analytics.attribution import PerformanceAttributor
from src.analytics.risk import PortfolioRiskAnalytics
from src.analytics.performance import PortfolioPerformanceMetrics
from src.analytics.reporting.portfolio_report import PortfolioReporter
from src.experiments.tracking import ExperimentTracker
from src.experiments.registry import ExperimentRegistry
from src.services.portfolio_service import PortfolioService


@pytest.fixture
def sample_returns_df():
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    data = np.random.normal(0.0005, 0.015, size=(100, 4))
    return pd.DataFrame(data, index=dates, columns=tickers)


@pytest.fixture
def sample_factor_scores():
    dates = pd.date_range("2020-01-01", periods=5, freq="ME")
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    records = []
    for d in dates:
        for t in tickers:
            records.append({
                "date": d,
                "ticker": t,
                "final_score": np.random.uniform(-1.0, 1.0),
            })
    return pd.DataFrame(records)


@pytest.fixture
def sample_metadata():
    return pd.DataFrame({
        "ticker": ["AAPL", "MSFT", "GOOGL", "AMZN"],
        "sector": ["Tech", "Tech", "Tech", "Consumer"],
    })


def test_covariance_estimators(sample_returns_df):
    for name in ["ledoit_wolf", "oas", "sample", "factor"]:
        estimator = get_covariance_estimator(name)
        cov = estimator.estimate(sample_returns_df)
        assert isinstance(cov, pd.DataFrame)
        assert cov.shape == (4, 4)
        assert np.allclose(cov.values, cov.values.T)  # Symmetric
        assert np.all(np.linalg.eigvals(cov.values) > 0)  # Positive definite


def test_factor_covariance_decomposition(sample_returns_df):
    exposures = pd.DataFrame(
        np.random.normal(size=(4, 2)),
        index=["AAPL", "MSFT", "GOOGL", "AMZN"],
        columns=["factor1", "factor2"],
    )
    estimator = FactorCovariance()
    total_cov, sys_cov, idio_cov = estimator.estimate_decomposed(sample_returns_df, exposures)
    assert total_cov.shape == (4, 4)
    assert sys_cov.shape == (4, 4)
    assert idio_cov.shape == (4, 4)


def test_cvar_calculator(sample_returns_df):
    calc = CVaRCalculator(alpha=0.95)
    cvar = calc.compute_historical_cvar(sample_returns_df["AAPL"])
    assert cvar >= 0.0
    scenario_matrix, tickers = calc.build_scenario_matrix(sample_returns_df)
    assert scenario_matrix.shape == (100, 4)
    assert tickers == ["AAPL", "MSFT", "GOOGL", "AMZN"]


def test_cvar_optimizer(sample_returns_df, sample_metadata):
    expected_returns = pd.Series([0.02, 0.03, 0.015, 0.025], index=["AAPL", "MSFT", "GOOGL", "AMZN"])
    cov_df = SampleCovariance().estimate(sample_returns_df)
    sector_map = sample_metadata.set_index("ticker")["sector"]

    optimizer = CVaROptimizer()
    res = optimizer.optimize(
        expected_returns=expected_returns,
        covariance_df=cov_df,
        sector_map=sector_map,
        returns_panel=sample_returns_df,
        config={"max_position_size": 0.5, "max_sector_exposure": 0.8},
    )

    assert res.is_successful
    assert len(res.weights) == 4
    assert np.isclose(res.weights.sum(), 1.0, atol=1e-3)
    assert (res.weights >= 0).all()


def test_mean_variance_optimizer(sample_returns_df):
    expected_returns = pd.Series([0.02, 0.03, 0.015, 0.025], index=["AAPL", "MSFT", "GOOGL", "AMZN"])
    cov_df = SampleCovariance().estimate(sample_returns_df)

    optimizer = MeanVarianceOptimizer()
    res = optimizer.optimize(
        expected_returns=expected_returns,
        covariance_df=cov_df,
        config={"max_position_size": 0.5},
    )

    assert res.is_successful
    assert len(res.weights) == 4
    assert np.isclose(res.weights.sum(), 1.0, atol=1e-3)


def test_risk_parity_optimizer(sample_returns_df):
    expected_returns = pd.Series([0.02, 0.03, 0.015, 0.025], index=["AAPL", "MSFT", "GOOGL", "AMZN"])
    cov_df = SampleCovariance().estimate(sample_returns_df)

    optimizer = RiskParityOptimizer()
    res = optimizer.optimize(
        expected_returns=expected_returns,
        covariance_df=cov_df,
        config={"max_position_size": 0.5},
    )

    assert res.is_successful
    assert len(res.weights) == 4
    assert np.isclose(res.weights.sum(), 1.0, atol=1e-3)


def test_performance_attributor(sample_returns_df):
    weights = pd.Series([0.25, 0.25, 0.25, 0.25], index=["AAPL", "MSFT", "GOOGL", "AMZN"])
    asset_returns = pd.Series([0.01, 0.02, -0.01, 0.015], index=["AAPL", "MSFT", "GOOGL", "AMZN"])
    exposures = pd.DataFrame(
        np.random.normal(size=(4, 2)),
        index=["AAPL", "MSFT", "GOOGL", "AMZN"],
        columns=["f1", "f2"],
    )

    attributor = PerformanceAttributor()
    attr = attributor.attribute_returns(weights, asset_returns, exposures)
    assert "total_return" in attr
    assert "systematic_return" in attr
    assert "idiosyncratic_return" in attr


def test_portfolio_risk_analytics(sample_returns_df):
    weights = pd.Series([0.25, 0.25, 0.25, 0.25], index=["AAPL", "MSFT", "GOOGL", "AMZN"])
    cov_df = SampleCovariance().estimate(sample_returns_df)

    analytics = PortfolioRiskAnalytics()
    mcr_df = analytics.compute_mcr(weights, cov_df)
    assert mcr_df.shape == (4, 4)
    assert "mcr" in mcr_df.columns
    assert "pcr" in mcr_df.columns


def test_experiment_tracker():
    tracker = ExperimentTracker()
    git_hash = tracker.get_git_hash()
    assert isinstance(git_hash, str)

    rec = tracker.create_experiment_record(
        experiment_name="test_run",
        config_dict={"solver": "CLARABEL"},
    )
    assert rec["experiment_name"] == "test_run"
    assert "config_hash" in rec


def test_portfolio_service_walk_forward(sample_returns_df, sample_factor_scores, sample_metadata):
    # Melt returns df to long format for portfolio service
    rets_reset = sample_returns_df.reset_index().melt(id_vars="index", var_name="ticker", value_name="simple_return")
    rets_reset.rename(columns={"index": "date"}, inplace=True)

    service = PortfolioService()
    weights_df, logs_df, exp_record = service.run_walk_forward_optimization(
        composite_scores_df=sample_factor_scores,
        returns_df=rets_reset,
        metadata_df=sample_metadata,
        config={"covariance_model": "sample", "optimizer_type": "cvar", "max_position_size": 0.5},
    )

    assert not weights_df.empty
    assert not logs_df.empty
    assert "experiment_id" in exp_record
