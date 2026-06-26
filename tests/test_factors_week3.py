from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.factors.evaluation import FactorEvaluator
from src.factors.library import BookToPrice, GrossProfitability, LowVolatility, PriceMomentum, ThreeMonthMomentum
from src.factors.neutralization import neutralize_factor_scores, winsorize_series, z_score_series


def test_winsorize_series():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 100.0], index=["A", "B", "C", "D", "E"])
    # 20% clipping on both sides limits below 1.8 and above 23.2
    res = winsorize_series(s, limits=(0.2, 0.2))
    assert np.isclose(res["A"], 1.8)
    assert np.isclose(res["E"], 23.2)
    assert np.isclose(res["B"], 2.0)
    assert np.isclose(res["D"], 4.0)
    assert np.isclose(res["C"], 3.0)


def test_z_score_series():
    s = pd.Series([1.0, 2.0, 3.0], index=["A", "B", "C"])
    res = z_score_series(s)
    assert np.isclose(res.mean(), 0.0)
    assert np.isclose(res.std(), 1.0)


def test_z_score_series_constant():
    s = pd.Series([2.0, 2.0, 2.0], index=["A", "B", "C"])
    res = z_score_series(s)
    assert (res == 0.0).all()


def test_neutralize_factor_scores():
    factor = pd.Series([1.0, 2.0, 3.0, 4.0], index=["A", "B", "C", "D"])
    sizes = pd.Series([10.0, 20.0, 15.0, 30.0], index=["A", "B", "C", "D"])
    sectors = pd.Series(["Tech", "Tech", "Fin", "Fin"], index=["A", "B", "C", "D"])
    res = neutralize_factor_scores(factor, sizes, sectors, neutralize_size=True, neutralize_sector=True)
    assert len(res) == 4
    assert not res.isna().any()


def test_price_momentum():
    dates = pd.date_range("2020-01-01", periods=15, freq="ME")
    idx = pd.MultiIndex.from_product([dates, ["AAPL", "MSFT"]], names=["date", "ticker"])
    prices = pd.DataFrame(
        {"adj_close": [float(10.0 + i) for i in range(30)]}, index=idx
    )

    mom_factor = PriceMomentum()
    res = mom_factor.compute(
        engine=MagicMock(),
        prices_df=prices,
        metadata_df=pd.DataFrame(),
        fundamentals_df=pd.DataFrame(),
        date=pd.Timestamp("2020-12-31"),
        config={"momentum_lookback_months": 12},
    )
    assert len(res) == 2
    assert "AAPL" in res.index
    assert "MSFT" in res.index


def test_three_month_momentum():
    dates = pd.date_range("2020-01-01", periods=5, freq="ME")
    idx = pd.MultiIndex.from_product([dates, ["AAPL", "MSFT"]], names=["date", "ticker"])
    prices = pd.DataFrame(
        {"adj_close": [float(10.0 + i) for i in range(10)]}, index=idx
    )

    mom_factor = ThreeMonthMomentum()
    res = mom_factor.compute(
        engine=MagicMock(),
        prices_df=prices,
        metadata_df=pd.DataFrame(),
        fundamentals_df=pd.DataFrame(),
        date=pd.Timestamp("2020-04-30"),
        config={"short_momentum_lookback_months": 3},
    )
    assert len(res) == 2


def test_book_to_price():
    dates = pd.date_range("2020-01-01", periods=3, freq="ME")
    idx = pd.MultiIndex.from_product([dates, ["AAPL", "MSFT"]], names=["date", "ticker"])
    prices = pd.DataFrame(
        {"adj_close": [100.0, 150.0, 105.0, 155.0, 110.0, 160.0]}, index=idx
    )

    metadata = pd.DataFrame([
        {"ticker": "AAPL", "sector": "Tech", "shares_outstanding": 1000.0},
        {"ticker": "MSFT", "sector": "Tech", "shares_outstanding": 2000.0}
    ])

    fundamentals = pd.DataFrame([
        {"date": pd.Timestamp("2020-01-31"), "ticker": "AAPL", "book_value": 50000.0, "gross_profit": 5000.0, "total_assets": 100000.0, "eps": 2.5},
        {"date": pd.Timestamp("2020-01-31"), "ticker": "MSFT", "book_value": 80000.0, "gross_profit": 8000.0, "total_assets": 150000.0, "eps": 3.0}
    ])

    bp_factor = BookToPrice()
    res = bp_factor.compute(
        engine=MagicMock(),
        prices_df=prices,
        metadata_df=metadata,
        fundamentals_df=fundamentals,
        date=pd.Timestamp("2020-01-31"),
        config={},
    )
    assert len(res) == 2
    # AAPL B/P = 50000 / (100 * 1000) = 0.5
    assert np.isclose(res["AAPL"], 0.5)


def test_gross_profitability():
    dates = pd.date_range("2020-01-01", periods=3, freq="ME")
    idx = pd.MultiIndex.from_product([dates, ["AAPL", "MSFT"]], names=["date", "ticker"])
    prices = pd.DataFrame(
        {"adj_close": [100.0, 150.0, 105.0, 155.0, 110.0, 160.0]}, index=idx
    )

    fundamentals = pd.DataFrame([
        {"date": pd.Timestamp("2020-01-31"), "ticker": "AAPL", "book_value": 50000.0, "gross_profit": 5000.0, "total_assets": 100000.0, "eps": 2.5},
        {"date": pd.Timestamp("2020-01-31"), "ticker": "MSFT", "book_value": 80000.0, "gross_profit": 9000.0, "total_assets": 150000.0, "eps": 3.0}
    ])

    gp_factor = GrossProfitability()
    res = gp_factor.compute(
        engine=MagicMock(),
        prices_df=prices,
        metadata_df=pd.DataFrame(),
        fundamentals_df=fundamentals,
        date=pd.Timestamp("2020-01-31"),
        config={},
    )
    assert len(res) == 2
    # AAPL GP = 5000 / 100000 = 0.05
    assert np.isclose(res["AAPL"], 0.05)


def test_low_volatility():
    # 20 trading days of prices
    dates = pd.date_range("2020-01-01", periods=10)
    idx = pd.MultiIndex.from_product([dates, ["AAPL"]], names=["date", "ticker"])
    prices = pd.DataFrame(
        {"adj_close": [10.0, 10.1, 10.2, 10.1, 10.3, 10.4, 10.3, 10.5, 10.6, 10.5]}, index=idx
    )

    vol_factor = LowVolatility()
    res = vol_factor.compute(
        engine=MagicMock(),
        prices_df=prices,
        metadata_df=pd.DataFrame(),
        fundamentals_df=pd.DataFrame(),
        date=pd.Timestamp("2020-01-31"),  # month end of January 2020
        config={"volatility_lookback_days": 5},
    )
    assert len(res) == 1
    assert not pd.isna(res["AAPL"])


def test_factor_evaluator():
    factor_scores = pd.DataFrame([
        {"date": pd.Timestamp("2020-01-31"), "ticker": "A", "raw_score": 1.0, "final_score": 1.1},
        {"date": pd.Timestamp("2020-01-31"), "ticker": "B", "raw_score": 2.0, "final_score": 1.9},
        {"date": pd.Timestamp("2020-01-31"), "ticker": "C", "raw_score": 3.0, "final_score": 3.1},
        {"date": pd.Timestamp("2020-01-31"), "ticker": "D", "raw_score": 4.0, "final_score": 4.1},
        {"date": pd.Timestamp("2020-01-31"), "ticker": "E", "raw_score": 5.0, "final_score": 4.9},
        {"date": pd.Timestamp("2020-02-29"), "ticker": "A", "raw_score": 1.1, "final_score": 1.2},
        {"date": pd.Timestamp("2020-02-29"), "ticker": "B", "raw_score": 1.8, "final_score": 1.7},
        {"date": pd.Timestamp("2020-02-29"), "ticker": "C", "raw_score": 2.9, "final_score": 3.0},
        {"date": pd.Timestamp("2020-02-29"), "ticker": "D", "raw_score": 3.9, "final_score": 4.0},
        {"date": pd.Timestamp("2020-02-29"), "ticker": "E", "raw_score": 4.8, "final_score": 4.7},
        {"date": pd.Timestamp("2020-03-31"), "ticker": "A", "raw_score": 1.2, "final_score": 1.3},
        {"date": pd.Timestamp("2020-03-31"), "ticker": "B", "raw_score": 1.7, "final_score": 1.6},
        {"date": pd.Timestamp("2020-03-31"), "ticker": "C", "raw_score": 2.8, "final_score": 2.9},
        {"date": pd.Timestamp("2020-03-31"), "ticker": "D", "raw_score": 3.8, "final_score": 3.9},
        {"date": pd.Timestamp("2020-03-31"), "ticker": "E", "raw_score": 4.7, "final_score": 4.6},
    ])

    returns = pd.DataFrame([
        {"date": pd.Timestamp("2020-01-31"), "ticker": "A", "simple_return": 0.05},
        {"date": pd.Timestamp("2020-01-31"), "ticker": "B", "simple_return": -0.02},
        {"date": pd.Timestamp("2020-01-31"), "ticker": "C", "simple_return": 0.10},
        {"date": pd.Timestamp("2020-01-31"), "ticker": "D", "simple_return": 0.01},
        {"date": pd.Timestamp("2020-01-31"), "ticker": "E", "simple_return": 0.03},
        {"date": pd.Timestamp("2020-02-29"), "ticker": "A", "simple_return": 0.02},
        {"date": pd.Timestamp("2020-02-29"), "ticker": "B", "simple_return": 0.03},
        {"date": pd.Timestamp("2020-02-29"), "ticker": "C", "simple_return": -0.01},
        {"date": pd.Timestamp("2020-02-29"), "ticker": "D", "simple_return": 0.05},
        {"date": pd.Timestamp("2020-02-29"), "ticker": "E", "simple_return": 0.04},
        {"date": pd.Timestamp("2020-03-31"), "ticker": "A", "simple_return": 0.01},
        {"date": pd.Timestamp("2020-03-31"), "ticker": "B", "simple_return": 0.02},
        {"date": pd.Timestamp("2020-03-31"), "ticker": "C", "simple_return": 0.04},
        {"date": pd.Timestamp("2020-03-31"), "ticker": "D", "simple_return": 0.06},
        {"date": pd.Timestamp("2020-03-31"), "ticker": "E", "simple_return": 0.05},
        {"date": pd.Timestamp("2020-04-30"), "ticker": "A", "simple_return": 0.03},
        {"date": pd.Timestamp("2020-04-30"), "ticker": "B", "simple_return": 0.01},
        {"date": pd.Timestamp("2020-04-30"), "ticker": "C", "simple_return": 0.05},
        {"date": pd.Timestamp("2020-04-30"), "ticker": "D", "simple_return": 0.02},
        {"date": pd.Timestamp("2020-04-30"), "ticker": "E", "simple_return": 0.04},
    ])

    evaluator = FactorEvaluator()
    import src.factors.evaluation as ev
    orig_upsert = ev.upsert_rows
    ev.upsert_rows = MagicMock(return_value=1)

    try:
        res = evaluator.evaluate_factor(
            engine=MagicMock(),
            factor_name="TestFactor",
            factor_scores_df=factor_scores,
            returns_df=returns,
        )
        assert "raw_ic_mean" in res
        assert "final_ic_mean" in res
        assert "raw_icir" in res
        assert "final_sharpe_ls" in res
        assert "raw_half_life" in res
    finally:
        ev.upsert_rows = orig_upsert
