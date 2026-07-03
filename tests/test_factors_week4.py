from __future__ import annotations

from unittest.mock import MagicMock
import numpy as np
import pandas as pd
import pytest

from src.factors.library import (
    EVtoEBITDA, FCFYield, DebtToEquityChange, Accruals, OneMonthReversal,
    MarketBeta, IdiosyncraticVolatility, EarningsSurpriseSUE, EarningsRevision,
    InsiderBuyingRatio, YieldCurveSlopeBeta, CreditSpreadBeta, PMIMomentumBeta,
    FactorLibrary
)


def test_ev_to_ebitda():
    dates = pd.date_range("2020-01-01", periods=3, freq="ME")
    idx = pd.MultiIndex.from_product([dates, ["AAPL"]], names=["date", "ticker"])
    prices = pd.DataFrame({"adj_close": [100.0, 105.0, 110.0]}, index=idx)
    metadata = pd.DataFrame([{"ticker": "AAPL", "sector": "Tech", "shares_outstanding": 1000.0}])
    fundamentals = pd.DataFrame([
        {"date": pd.Timestamp("2020-01-31"), "ticker": "AAPL", "ebitda": 5000.0, "total_debt": 20000.0}
    ])
    factor = EVtoEBITDA()
    res = factor.compute(
        engine=MagicMock(),
        prices_df=prices,
        metadata_df=metadata,
        fundamentals_df=fundamentals,
        date=pd.Timestamp("2020-01-31"),
        config={},
    )
    # EV = 100 * 1000 + 20000 = 120000
    # EBITDA / EV = 5000 / 120000 = 0.0416666
    assert len(res) == 1
    assert np.isclose(res["AAPL"], 5000.0 / 120000.0)


def test_fcf_yield():
    dates = pd.date_range("2020-01-01", periods=3, freq="ME")
    idx = pd.MultiIndex.from_product([dates, ["AAPL"]], names=["date", "ticker"])
    prices = pd.DataFrame({"adj_close": [100.0, 105.0, 110.0]}, index=idx)
    metadata = pd.DataFrame([{"ticker": "AAPL", "sector": "Tech", "shares_outstanding": 1000.0}])
    fundamentals = pd.DataFrame([
        {"date": pd.Timestamp("2020-01-31"), "ticker": "AAPL", "operating_cash_flow": 12000.0, "capital_expenditures": 4000.0}
    ])
    factor = FCFYield()
    res = factor.compute(
        engine=MagicMock(),
        prices_df=prices,
        metadata_df=metadata,
        fundamentals_df=fundamentals,
        date=pd.Timestamp("2020-01-31"),
        config={},
    )
    # FCF = 12000 - 4000 = 8000
    # MC = 100 * 1000 = 100000
    # FCF / MC = 8000 / 100000 = 0.08
    assert len(res) == 1
    assert np.isclose(res["AAPL"], 0.08)


def test_debt_to_equity_change():
    dates = pd.date_range("2020-01-01", periods=3, freq="ME")
    idx = pd.MultiIndex.from_product([dates, ["AAPL"]], names=["date", "ticker"])
    prices = pd.DataFrame({"adj_close": [100.0, 105.0, 110.0]}, index=idx)
    fundamentals = pd.DataFrame([
        {"date": pd.Timestamp("2019-12-31"), "ticker": "AAPL", "total_debt": 10000.0, "book_value": 50000.0},
        {"date": pd.Timestamp("2020-01-31"), "ticker": "AAPL", "total_debt": 12000.0, "book_value": 50000.0}
    ])
    factor = DebtToEquityChange()
    res = factor.compute(
        engine=MagicMock(),
        prices_df=prices,
        metadata_df=pd.DataFrame(),
        fundamentals_df=fundamentals,
        date=pd.Timestamp("2020-01-31"),
        config={},
    )
    # DE_t0 = 10000 / 50000 = 0.2
    # DE_t1 = 12000 / 50000 = 0.24
    # Diff = 0.24 - 0.2 = 0.04
    # Factor output is -1 * Diff = -0.04
    assert len(res) == 1
    assert np.isclose(res["AAPL"], -0.04)


def test_accruals():
    dates = pd.date_range("2020-01-01", periods=3, freq="ME")
    idx = pd.MultiIndex.from_product([dates, ["AAPL"]], names=["date", "ticker"])
    prices = pd.DataFrame({"adj_close": [100.0, 105.0, 110.0]}, index=idx)
    fundamentals = pd.DataFrame([
        {"date": pd.Timestamp("2020-01-31"), "ticker": "AAPL", "net_income": 8000.0, "operating_cash_flow": 10000.0, "total_assets": 100000.0}
    ])
    factor = Accruals()
    res = factor.compute(
        engine=MagicMock(),
        prices_df=prices,
        metadata_df=pd.DataFrame(),
        fundamentals_df=fundamentals,
        date=pd.Timestamp("2020-01-31"),
        config={},
    )
    # Accruals = (8000 - 10000) / 100000 = -0.02
    # Factor output = -1 * Accruals = 0.02
    assert len(res) == 1
    assert np.isclose(res["AAPL"], 0.02)


def test_one_month_reversal():
    dates = pd.date_range("2020-01-01", periods=3, freq="ME")
    idx = pd.MultiIndex.from_product([dates, ["AAPL"]], names=["date", "ticker"])
    prices = pd.DataFrame({"adj_close": [100.0, 105.0, 110.0]}, index=idx)
    factor = OneMonthReversal()
    res = factor.compute(
        engine=MagicMock(),
        prices_df=prices,
        metadata_df=pd.DataFrame(),
        fundamentals_df=pd.DataFrame(),
        date=pd.Timestamp("2020-03-31"),
        config={},
    )
    # 1-month return from Feb to Mar = 110.0 / 105.0 - 1 = 0.047619
    # Output = -1 * 0.047619 = -0.047619
    assert len(res) == 1
    assert np.isclose(res["AAPL"], -1 * (110.0 / 105.0 - 1))


def test_market_beta_and_idiosyncratic_vol():
    dates = pd.date_range("2020-01-01", periods=10)
    idx = pd.MultiIndex.from_product([dates, ["AAPL"]], names=["date", "ticker"])
    prices = pd.DataFrame({"adj_close": [10.0, 10.1, 10.2, 10.3, 10.2, 10.4, 10.5, 10.6, 10.5, 10.7]}, index=idx)

    # Mock engine that returns Fama-French market factors
    engine = MagicMock()
    mock_sql = pd.DataFrame({
        "date": dates,
        "mkt_rf": [0.1, 0.2, -0.1, 0.3, -0.2, 0.1, 0.2, -0.1, 0.3, -0.2],
        "rf": [0.01] * 10
    })
    
    orig_read_sql = pd.read_sql
    pd.read_sql = MagicMock(return_value=mock_sql)

    try:
        beta_factor = MarketBeta()
        res_beta = beta_factor.compute(
            engine=engine,
            prices_df=prices,
            metadata_df=pd.DataFrame(),
            fundamentals_df=pd.DataFrame(),
            date=pd.Timestamp("2020-01-10"),
            config={},
        )
        assert len(res_beta) == 1
        assert not pd.isna(res_beta["AAPL"])

        vol_factor = IdiosyncraticVolatility()
        res_vol = vol_factor.compute(
            engine=engine,
            prices_df=prices,
            metadata_df=pd.DataFrame(),
            fundamentals_df=pd.DataFrame(),
            date=pd.Timestamp("2020-01-10"),
            config={},
        )
        assert len(res_vol) == 1
        assert not pd.isna(res_vol["AAPL"])
    finally:
        pd.read_sql = orig_read_sql


def test_earnings_factors():
    dates = pd.date_range("2020-01-01", periods=3, freq="ME")
    idx = pd.MultiIndex.from_product([dates, ["AAPL"]], names=["date", "ticker"])
    prices = pd.DataFrame({"adj_close": [100.0, 105.0, 110.0]}, index=idx)

    mock_calendar = pd.DataFrame([
        {"ticker": "AAPL", "date": pd.Timestamp("2020-01-15"), "surprise_pct": 5.0, "eps_estimate": 1.20},
        {"ticker": "AAPL", "date": pd.Timestamp("2020-02-15"), "surprise_pct": 8.0, "eps_estimate": 1.30}
    ])

    orig_read_sql = pd.read_sql
    pd.read_sql = MagicMock(return_value=mock_calendar)

    try:
        sue_factor = EarningsSurpriseSUE()
        res_sue = sue_factor.compute(
            engine=MagicMock(),
            prices_df=prices,
            metadata_df=pd.DataFrame(),
            fundamentals_df=pd.DataFrame(),
            date=pd.Timestamp("2020-02-29"),
            config={},
        )
        assert len(res_sue) == 1
        assert np.isclose(res_sue["AAPL"], 8.0)

        rev_factor = EarningsRevision()
        res_rev = rev_factor.compute(
            engine=MagicMock(),
            prices_df=prices,
            metadata_df=pd.DataFrame(),
            fundamentals_df=pd.DataFrame(),
            date=pd.Timestamp("2020-02-29"),
            config={},
        )
        assert len(res_rev) == 1
        assert np.isclose(res_rev["AAPL"], 0.10)
    finally:
        pd.read_sql = orig_read_sql


def test_insider_buying_ratio():
    dates = pd.date_range("2020-01-01", periods=3, freq="ME")
    idx = pd.MultiIndex.from_product([dates, ["AAPL"]], names=["date", "ticker"])
    prices = pd.DataFrame({"adj_close": [100.0, 105.0, 110.0]}, index=idx)

    # Insider transactions within 60 days of 2020-02-29 (Feb 29)
    mock_insider = pd.DataFrame([
        {"ticker": "AAPL", "date": pd.Timestamp("2020-01-10"), "shares": 1000.0, "text": "Purchase at price 100.00"},
        {"ticker": "AAPL", "date": pd.Timestamp("2020-02-10"), "shares": 500.0, "text": "Sale at price 105.00"}
    ])

    orig_read_sql = pd.read_sql
    pd.read_sql = MagicMock(return_value=mock_insider)

    try:
        factor = InsiderBuyingRatio()
        res = factor.compute(
            engine=MagicMock(),
            prices_df=prices,
            metadata_df=pd.DataFrame(),
            fundamentals_df=pd.DataFrame(),
            date=pd.Timestamp("2020-02-29"),
            config={},
        )
        assert len(res) == 1
        # Buys = 1000, Sells = 500
        # Ratio = (1000 - 500) / (1000 + 500) = 500 / 1500 = 0.333333
        assert np.isclose(res["AAPL"], 1.0 / 3.0)
    finally:
        pd.read_sql = orig_read_sql


def test_macro_betas():
    dates = pd.date_range("2020-01-01", periods=10)
    idx = pd.MultiIndex.from_product([dates, ["AAPL"]], names=["date", "ticker"])
    prices = pd.DataFrame({"adj_close": [10.0, 10.1, 10.2, 10.3, 10.2, 10.4, 10.5, 10.6, 10.5, 10.7]}, index=idx)

    mock_macro = pd.DataFrame([
        {"date": d, "series_name": "DGS10", "value": 1.5 + 0.1 * i} for i, d in enumerate(dates)
    ] + [
        {"date": d, "series_name": "DGS2", "value": 0.5} for d in dates
    ] + [
        {"date": d, "series_name": "BAMLH0A0HYM2", "value": 3.0 - 0.05 * i} for i, d in enumerate(dates)
    ] + [
        {"date": d, "series_name": "NAPM", "value": 50.0 + 0.5 * i + 0.1 * (i % 3)} for i, d in enumerate(dates)
    ])

    def mock_read_sql_fn(sql, con, *args, **kwargs):
        sql_str = str(sql).lower()
        if "dgs10" in sql_str or "dgs2" in sql_str:
            return mock_macro[mock_macro["series_name"].isin(["DGS10", "DGS2"])]
        elif "bamlh0a0hym2" in sql_str:
            return mock_macro[mock_macro["series_name"] == "BAMLH0A0HYM2"]
        elif "napm" in sql_str:
            return mock_macro[mock_macro["series_name"] == "NAPM"]
        return mock_macro

    orig_read_sql = pd.read_sql
    pd.read_sql = mock_read_sql_fn

    try:
        factor_curve = YieldCurveSlopeBeta()
        res_curve = factor_curve.compute(
            engine=MagicMock(),
            prices_df=prices,
            metadata_df=pd.DataFrame(),
            fundamentals_df=pd.DataFrame(),
            date=pd.Timestamp("2020-01-10"),
            config={},
        )
        assert len(res_curve) == 1
        assert not pd.isna(res_curve["AAPL"])

        factor_spread = CreditSpreadBeta()
        res_spread = factor_spread.compute(
            engine=MagicMock(),
            prices_df=prices,
            metadata_df=pd.DataFrame(),
            fundamentals_df=pd.DataFrame(),
            date=pd.Timestamp("2020-01-10"),
            config={},
        )
        assert len(res_spread) == 1
        assert not pd.isna(res_spread["AAPL"])

        factor_pmi = PMIMomentumBeta()
        res_pmi = factor_pmi.compute(
            engine=MagicMock(),
            prices_df=prices,
            metadata_df=pd.DataFrame(),
            fundamentals_df=pd.DataFrame(),
            date=pd.Timestamp("2020-01-10"),
            config={},
        )
        assert len(res_pmi) == 1
        assert not pd.isna(res_pmi["AAPL"])
    finally:
        pd.read_sql = orig_read_sql
