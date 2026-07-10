from __future__ import annotations

from unittest.mock import MagicMock
import numpy as np
import pandas as pd
import pytest

from src.backtest.portfolio import PortfolioConstructor
from src.backtest.rebalancer import Rebalancer
from src.backtest.transaction_cost import LinearTransactionCostModel
from src.backtest.metrics import PortfolioMetrics
from src.backtest.engine import VectorizedBacktester


def test_portfolio_construction():
    constructor = PortfolioConstructor()
    factor_scores = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=["A", "B", "C", "D", "E"])

    # 1. Equal Weight Long Only
    # k = max(1, ceil(5 * 0.2)) = 1 (top 20% of 5 is 1 asset)
    # With net_exposure=1.0, the top asset (E) gets 1.0 weight
    # Since only 1 asset is selected, the max_position_size of 0.5 is relaxed to 1.0 to keep weight sum = 1.0.
    w_lo = constructor.construct_weights(factor_scores, method="equal_weight_long_only", max_position_size=0.5)
    assert w_lo["E"] == pytest.approx(1.0)
    assert (w_lo.drop("E") == 0.0).all()

    # If we have 10 assets, k = 2
    factor_scores_10 = pd.Series(range(10), index=[f"T{i}" for i in range(10)])
    # top 2 are T9, T8. Target weights are 0.5 each.
    # Max position size 0.3 is relaxed to 0.5 because 2 * 0.3 = 0.6 < 1.0 (infeasible target sum)
    w_lo_10 = constructor.construct_weights(factor_scores_10, method="equal_weight_long_only", max_position_size=0.3)
    assert w_lo_10["T9"] == pytest.approx(0.5)
    assert w_lo_10["T8"] == pytest.approx(0.5)

    # 2. Equal Weight Long Short
    # k = 1. top is E (long), bottom is A (short). Target sum = 1.0 gross, 0.0 net.
    # Long target = 0.5, Short target = -0.5
    w_ls = constructor.construct_weights(factor_scores, method="equal_weight_long_short", max_position_size=0.5, long_only=False, net_exposure=0.0, gross_exposure=1.0)
    assert w_ls["E"] == pytest.approx(0.5)
    assert w_ls["A"] == pytest.approx(-0.5)
    assert (w_ls.loc[["B", "C", "D"]] == 0.0).all()


def test_rebalancer():
    rebalancer = Rebalancer(frequency="monthly")

    # 1. Schedule filtering
    dates = pd.DatetimeIndex([
        "2020-01-15", "2020-01-31",
        "2020-02-15", "2020-02-28",
        "2020-03-10", "2020-03-31"
    ])
    schedule = rebalancer.get_rebalance_schedule(dates)
    assert len(schedule) == 3
    assert schedule[0] == pd.Timestamp("2020-01-31")
    assert schedule[1] == pd.Timestamp("2020-02-28")
    assert schedule[2] == pd.Timestamp("2020-03-31")

    # 2. Turnover calculation
    prev_w = pd.Series([0.4, 0.6], index=["A", "B"])
    target_w = pd.Series([0.5, 0.5], index=["A", "B"])
    rets = pd.Series([0.10, -0.05], index=["A", "B"])  # A up 10%, B down 5%

    # drifted weights:
    # port_ret = 0.4 * 0.10 + 0.6 * -0.05 = 0.04 - 0.03 = 0.01 (1%)
    # w_drift_A = 0.4 * 1.1 / 1.01 = 0.43564
    # w_drift_B = 0.6 * 0.95 / 1.01 = 0.56436
    # turnover = |0.5 - 0.43564| + |0.5 - 0.56436| = 0.06436 * 2 = 0.12871
    turnover, weight_change = rebalancer.calculate_turnover(prev_w, target_w, rets)
    assert turnover == pytest.approx(0.12871287, rel=1e-5)


def test_transaction_cost_model():
    # Commission = 5 bps, Spread = 10 bps
    # Cost = Turnover * (5 + 0.5 * 10) / 10000 = Turnover * 0.001
    model = LinearTransactionCostModel(commission_bps=5.0, bid_ask_spread_bps=10.0)
    cost = model.calculate_cost(pd.Series([0.1, -0.1]), 0.2)
    assert cost == pytest.approx(0.0002)


def test_metrics_calculation():
    # Setup simple net returns: 5% monthly gain for 12 months
    net_rets = pd.Series([0.05] * 12)
    gross_rets = pd.Series([0.06] * 12)
    turnover = pd.Series([0.1] * 12)

    metrics = PortfolioMetrics.compute_all(net_rets, gross_rets, turnover, frequency="monthly")
    
    # Cumulative return = 1.05^12 - 1 = 0.795856
    assert metrics["cumulative_return_net"] == pytest.approx(0.795856, rel=1e-5)
    # CAGR = cumulative return over 1 year = 79.58%
    assert metrics["cagr_net"] == pytest.approx(0.795856, rel=1e-5)
    assert metrics["annualized_volatility"] == pytest.approx(0.0)  # since standard dev of constant returns is 0
    assert metrics["sharpe_ratio"] == 0.0  # std is 0


def test_backtesting_engine():
    # Create simple mock data
    dates = pd.date_range("2020-01-31", periods=5, freq="ME")
    tickers = ["AAPL", "MSFT", "GOOGL"]
    
    factor_records = []
    return_records = []
    universe_records = []
    
    # 5 dates, 3 tickers
    for date in dates:
        for i, ticker in enumerate(tickers):
            factor_records.append({
                "date": date,
                "ticker": ticker,
                "factor_name": "TestFactor",
                "final_score": float(i + 1),  # AAPL: 1.0, MSFT: 2.0, GOOGL: 3.0
            })
            return_records.append({
                "date": date,
                "ticker": ticker,
                "simple_return": 0.02,  # flat 2% return
                "log_return": np.log1p(0.02),
            })
            universe_records.append({
                "date": date,
                "ticker": ticker,
                "in_universe": True,
            })
            
    factor_df = pd.DataFrame(factor_records)
    returns_df = pd.DataFrame(return_records)
    universe_df = pd.DataFrame(universe_records)
    
    backtester = VectorizedBacktester(
        initial_capital=10000.0,
        rebalance_frequency="monthly",
        weighting_method="equal_weight_long_only",
        max_position_size=1.0,
        commission_bps=0.0,
        bid_ask_spread_bps=0.0,
    )
    
    summary = backtester.run(
        factor_scores_df=factor_df,
        returns_df=returns_df,
        universe_membership_df=universe_df,
        backtest_name="Test_Run",
    )
    
    assert summary["backtest_name"] == "Test_Run"
    assert "metrics" in summary
    assert "results" in summary
    assert len(summary["results"]) == 5  # 5 periods
    
    # Verify weights were constructed
    assert len(summary["weights"]) > 0


def test_backtester_edge_cases():
    backtester = VectorizedBacktester()
    
    # 1. Empty dataframes
    summary = backtester.run(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), "Empty_Run")
    assert summary == {}
