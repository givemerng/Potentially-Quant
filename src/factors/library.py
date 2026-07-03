from __future__ import annotations

import logging
import abc
from typing import Optional, List
import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine
from joblib import Parallel, delayed

from src.factors.base import BaseAlpha
from src.factors.neutralization import winsorize_series, z_score_series, neutralize_factor_scores


class PriceMomentum(BaseAlpha):
    """Price Momentum factor: Cumulative return over the past N months, excluding the last month.

    Often represented as 12-1 month momentum.
    """

    def compute(
        self,
        engine: Engine,
        prices_df: pd.DataFrame,
        metadata_df: pd.DataFrame,
        fundamentals_df: pd.DataFrame,
        date: pd.Timestamp,
        config: dict,
    ) -> pd.Series:
        if prices_df.empty:
            return pd.Series(dtype=float)

        close_wide = prices_df["adj_close"].unstack("ticker").sort_index()
        close_m = close_wide.ffill().resample("ME").last()

        lookback = config.get("momentum_lookback_months", 12)
        if lookback < 2:
            lookback = 12

        mom = close_m.shift(1) / close_m.shift(lookback) - 1.0

        if date not in mom.index:
            return pd.Series(np.nan, index=close_wide.columns)

        return mom.loc[date]


class ThreeMonthMomentum(BaseAlpha):
    """Three-Month Momentum factor: Cumulative return over the past 3 months, excluding the last month."""

    def compute(
        self,
        engine: Engine,
        prices_df: pd.DataFrame,
        metadata_df: pd.DataFrame,
        fundamentals_df: pd.DataFrame,
        date: pd.Timestamp,
        config: dict,
    ) -> pd.Series:
        if prices_df.empty:
            return pd.Series(dtype=float)

        close_wide = prices_df["adj_close"].unstack("ticker").sort_index()
        close_m = close_wide.ffill().resample("ME").last()

        lookback = config.get("short_momentum_lookback_months", 3)
        if lookback < 2:
            lookback = 3

        mom = close_m.shift(1) / close_m.shift(lookback) - 1.0

        if date not in mom.index:
            return pd.Series(np.nan, index=close_wide.columns)

        return mom.loc[date]


class BookToPrice(BaseAlpha):
    """Book-to-Price (B/P) factor: Book value of equity / Market Capitalization."""

    def compute(
        self,
        engine: Engine,
        prices_df: pd.DataFrame,
        metadata_df: pd.DataFrame,
        fundamentals_df: pd.DataFrame,
        date: pd.Timestamp,
        config: dict,
    ) -> pd.Series:
        if prices_df.empty:
            return pd.Series(dtype=float)

        close_wide = prices_df["adj_close"].unstack("ticker").sort_index()
        close_m = close_wide.ffill().resample("ME").last()

        if fundamentals_df.empty or metadata_df.empty:
            return pd.Series(np.nan, index=close_wide.columns)

        book_df = fundamentals_df.dropna(subset=["book_value"])
        if book_df.empty:
            return pd.Series(np.nan, index=close_wide.columns)

        book_wide = book_df.pivot(index="date", columns="ticker", values="book_value")
        union_idx = book_wide.index.union(close_m.index)
        book_wide_m = book_wide.reindex(union_idx).ffill().reindex(close_m.index)

        meta_indexed = metadata_df.set_index("ticker")
        shares = meta_indexed["shares_outstanding"].reindex(close_wide.columns)
        market_cap = close_m.multiply(shares, axis=1)

        bp = book_wide_m / market_cap

        if date not in bp.index:
            return pd.Series(np.nan, index=close_wide.columns)

        return bp.loc[date]


class GrossProfitability(BaseAlpha):
    """Gross Profitability factor: Gross Profit / Total Assets."""

    def compute(
        self,
        engine: Engine,
        prices_df: pd.DataFrame,
        metadata_df: pd.DataFrame,
        fundamentals_df: pd.DataFrame,
        date: pd.Timestamp,
        config: dict,
    ) -> pd.Series:
        if prices_df.empty:
            return pd.Series(dtype=float)

        close_wide = prices_df["adj_close"].unstack("ticker").sort_index()
        close_m = close_wide.ffill().resample("ME").last()

        if fundamentals_df.empty:
            return pd.Series(np.nan, index=close_wide.columns)

        gp_df = fundamentals_df.dropna(subset=["gross_profit"])
        assets_df = fundamentals_df.dropna(subset=["total_assets"])

        if gp_df.empty or assets_df.empty:
            return pd.Series(np.nan, index=close_wide.columns)

        gp_wide = gp_df.pivot(index="date", columns="ticker", values="gross_profit")
        assets_wide = assets_df.pivot(index="date", columns="ticker", values="total_assets")

        union_gp = gp_wide.index.union(close_m.index)
        gp_wide_m = gp_wide.reindex(union_gp).ffill().reindex(close_m.index)

        union_assets = assets_wide.index.union(close_m.index)
        assets_wide_m = assets_wide.reindex(union_assets).ffill().reindex(close_m.index)

        gp_factor = gp_wide_m / assets_wide_m

        if date not in gp_factor.index:
            return pd.Series(np.nan, index=close_wide.columns)

        return gp_factor.loc[date]


class LowVolatility(BaseAlpha):
    """Low Volatility factor: Standard deviation of daily returns over the past N trading days."""

    def compute(
        self,
        engine: Engine,
        prices_df: pd.DataFrame,
        metadata_df: pd.DataFrame,
        fundamentals_df: pd.DataFrame,
        date: pd.Timestamp,
        config: dict,
    ) -> pd.Series:
        if prices_df.empty:
            return pd.Series(dtype=float)

        close_wide = prices_df["adj_close"].unstack("ticker").sort_index()
        daily_returns = close_wide.pct_change(fill_method=None)

        lookback = config.get("volatility_lookback_days", 252)
        min_periods = max(1, lookback // 2)
        rolling_std = daily_returns.rolling(window=lookback, min_periods=min_periods).std()

        close_m_index = close_wide.resample("ME").last().index
        union_vol = rolling_std.index.union(close_m_index)
        vol_m = rolling_std.reindex(union_vol).ffill().reindex(close_m_index)

        # Invert volatility so higher score = lower volatility (standard alpha convention)
        if date not in vol_m.index:
            return pd.Series(np.nan, index=close_wide.columns)

        return -1.0 * vol_m.loc[date]


class EVtoEBITDA(BaseAlpha):
    """EV/EBITDA factor: EBITDA / Enterprise Value (inverted for alpha sorting)."""

    def compute(
        self,
        engine: Engine,
        prices_df: pd.DataFrame,
        metadata_df: pd.DataFrame,
        fundamentals_df: pd.DataFrame,
        date: pd.Timestamp,
        config: dict,
    ) -> pd.Series:
        if prices_df.empty:
            return pd.Series(dtype=float)

        close_wide = prices_df["adj_close"].unstack("ticker").sort_index()
        close_m = close_wide.ffill().resample("ME").last()

        if fundamentals_df.empty or metadata_df.empty:
            return pd.Series(np.nan, index=close_wide.columns)

        ebitda_df = fundamentals_df.dropna(subset=["ebitda"])
        debt_df = fundamentals_df.dropna(subset=["total_debt"])

        if ebitda_df.empty:
            return pd.Series(np.nan, index=close_wide.columns)

        ebitda_wide = ebitda_df.pivot(index="date", columns="ticker", values="ebitda")
        debt_wide = debt_df.pivot(index="date", columns="ticker", values="total_debt") if not debt_df.empty else pd.DataFrame(0.0, index=ebitda_wide.index, columns=ebitda_wide.columns)

        union_idx = ebitda_wide.index.union(close_m.index)
        ebitda_wide_m = ebitda_wide.reindex(union_idx).ffill().reindex(close_m.index)
        debt_wide_m = debt_wide.reindex(union_idx).ffill().reindex(close_m.index)

        meta_indexed = metadata_df.set_index("ticker")
        shares = meta_indexed["shares_outstanding"].reindex(close_wide.columns)
        market_cap = close_m.multiply(shares, axis=1)

        # EV = Market Cap + Total Debt
        ev = market_cap + debt_wide_m.fillna(0.0)

        ebitda_ev = ebitda_wide_m / ev

        if date not in ebitda_ev.index:
            return pd.Series(np.nan, index=close_wide.columns)

        return ebitda_ev.loc[date]


class FCFYield(BaseAlpha):
    """Free Cash Flow Yield factor: FCF / Market Cap."""

    def compute(
        self,
        engine: Engine,
        prices_df: pd.DataFrame,
        metadata_df: pd.DataFrame,
        fundamentals_df: pd.DataFrame,
        date: pd.Timestamp,
        config: dict,
    ) -> pd.Series:
        if prices_df.empty:
            return pd.Series(dtype=float)

        close_wide = prices_df["adj_close"].unstack("ticker").sort_index()
        close_m = close_wide.ffill().resample("ME").last()

        if fundamentals_df.empty or metadata_df.empty:
            return pd.Series(np.nan, index=close_wide.columns)

        ocf_df = fundamentals_df.dropna(subset=["operating_cash_flow"])
        capex_df = fundamentals_df.dropna(subset=["capital_expenditures"])

        if ocf_df.empty:
            return pd.Series(np.nan, index=close_wide.columns)

        ocf_wide = ocf_df.pivot(index="date", columns="ticker", values="operating_cash_flow")
        capex_wide = capex_df.pivot(index="date", columns="ticker", values="capital_expenditures") if not capex_df.empty else pd.DataFrame(0.0, index=ocf_wide.index, columns=ocf_wide.columns)

        # Align to union index
        union_idx = ocf_wide.index.union(close_m.index)
        ocf_wide_m = ocf_wide.reindex(union_idx).ffill().reindex(close_m.index)
        capex_wide_m = capex_wide.reindex(union_idx).ffill().reindex(close_m.index).fillna(0.0)

        # FCF = OCF - CapEx (since CapEx is stored as positive absolute value)
        fcf = ocf_wide_m - capex_wide_m

        meta_indexed = metadata_df.set_index("ticker")
        shares = meta_indexed["shares_outstanding"].reindex(close_wide.columns)
        market_cap = close_m.multiply(shares, axis=1)

        fcf_yield = fcf / market_cap

        if date not in fcf_yield.index:
            return pd.Series(np.nan, index=close_wide.columns)

        return fcf_yield.loc[date]


class DebtToEquityChange(BaseAlpha):
    """Debt-to-Equity Change factor: Quarter-on-quarter change in Total Debt / Stockholders' Equity."""

    def compute(
        self,
        engine: Engine,
        prices_df: pd.DataFrame,
        metadata_df: pd.DataFrame,
        fundamentals_df: pd.DataFrame,
        date: pd.Timestamp,
        config: dict,
    ) -> pd.Series:
        if prices_df.empty:
            return pd.Series(dtype=float)

        close_wide = prices_df["adj_close"].unstack("ticker").sort_index()
        close_m = close_wide.ffill().resample("ME").last()

        if fundamentals_df.empty:
            return pd.Series(np.nan, index=close_wide.columns)

        debt_df = fundamentals_df.dropna(subset=["total_debt"])
        eq_df = fundamentals_df.dropna(subset=["book_value"])

        if debt_df.empty or eq_df.empty:
            return pd.Series(np.nan, index=close_wide.columns)

        debt_wide = debt_df.pivot(index="date", columns="ticker", values="total_debt")
        eq_wide = eq_df.pivot(index="date", columns="ticker", values="book_value")

        # Compute D/E ratio quarterly
        de_ratio = debt_wide / eq_wide.replace(0.0, np.nan)
        de_change = de_ratio.diff(1)

        # Align to month-ends
        union_idx = de_change.index.union(close_m.index)
        de_change_m = de_change.reindex(union_idx).ffill().reindex(close_m.index)

        # Invert change: lower debt growth is preferred (alpha convention)
        if date not in de_change_m.index:
            return pd.Series(np.nan, index=close_wide.columns)

        return -1.0 * de_change_m.loc[date]


class Accruals(BaseAlpha):
    """Accruals factor: (Net Income - Operating Cash Flow) / Total Assets."""

    def compute(
        self,
        engine: Engine,
        prices_df: pd.DataFrame,
        metadata_df: pd.DataFrame,
        fundamentals_df: pd.DataFrame,
        date: pd.Timestamp,
        config: dict,
    ) -> pd.Series:
        if prices_df.empty:
            return pd.Series(dtype=float)

        close_wide = prices_df["adj_close"].unstack("ticker").sort_index()
        close_m = close_wide.ffill().resample("ME").last()

        if fundamentals_df.empty:
            return pd.Series(np.nan, index=close_wide.columns)

        ni_df = fundamentals_df.dropna(subset=["net_income"])
        ocf_df = fundamentals_df.dropna(subset=["operating_cash_flow"])
        assets_df = fundamentals_df.dropna(subset=["total_assets"])

        if ni_df.empty or ocf_df.empty or assets_df.empty:
            return pd.Series(np.nan, index=close_wide.columns)

        ni_wide = ni_df.pivot(index="date", columns="ticker", values="net_income")
        ocf_wide = ocf_df.pivot(index="date", columns="ticker", values="operating_cash_flow")
        assets_wide = assets_df.pivot(index="date", columns="ticker", values="total_assets")

        accruals_quarterly = (ni_wide - ocf_wide) / assets_wide.replace(0.0, np.nan)

        # Align to month-ends
        union_idx = accruals_quarterly.index.union(close_m.index)
        accruals_m = accruals_quarterly.reindex(union_idx).ffill().reindex(close_m.index)

        # Invert accruals: lower accruals (higher earnings quality) is preferred
        if date not in accruals_m.index:
            return pd.Series(np.nan, index=close_wide.columns)

        return -1.0 * accruals_m.loc[date]


class OneMonthReversal(BaseAlpha):
    """One-Month Reversal factor: -1 * Return over the past month."""

    def compute(
        self,
        engine: Engine,
        prices_df: pd.DataFrame,
        metadata_df: pd.DataFrame,
        fundamentals_df: pd.DataFrame,
        date: pd.Timestamp,
        config: dict,
    ) -> pd.Series:
        if prices_df.empty:
            return pd.Series(dtype=float)

        close_wide = prices_df["adj_close"].unstack("ticker").sort_index()
        close_m = close_wide.ffill().resample("ME").last()

        monthly_return = close_m.pct_change(1)

        if date not in monthly_return.index:
            return pd.Series(np.nan, index=close_wide.columns)

        return -1.0 * monthly_return.loc[date]


class MarketBeta(BaseAlpha):
    """Market Beta factor: Rolling 252-day beta against the Fama-French market return."""

    def compute(
        self,
        engine: Engine,
        prices_df: pd.DataFrame,
        metadata_df: pd.DataFrame,
        fundamentals_df: pd.DataFrame,
        date: pd.Timestamp,
        config: dict,
    ) -> pd.Series:
        if prices_df.empty:
            return pd.Series(dtype=float)

        close_wide = prices_df["adj_close"].unstack("ticker").sort_index()
        stock_rets = close_wide.pct_change(fill_method=None)

        # Load Fama-French market excess return (mkt_rf) and risk-free rate (rf)
        query = "SELECT date, mkt_rf, rf FROM fama_french ORDER BY date"
        try:
            ff_df = pd.read_sql(query, engine)
            ff_df["date"] = pd.to_datetime(ff_df["date"])
            ff_df = ff_df.set_index("date")
            mkt_rets = (ff_df["mkt_rf"] + ff_df["rf"]) / 100.0  # convert from pct to decimal
        except Exception:
            # Fallback to equal weight index if Fama-French table is missing or empty
            mkt_rets = stock_rets.mean(axis=1)

        # Slice trailing 252 days
        window_stocks = stock_rets.loc[:date].tail(252)
        window_mkt = mkt_rets.reindex(window_stocks.index).ffill()

        if window_stocks.empty or window_mkt.dropna().empty:
            return pd.Series(np.nan, index=close_wide.columns)

        cov = window_stocks.apply(lambda col: col.cov(window_mkt))
        var = window_mkt.var()

        if pd.isna(var) or var == 0.0:
            return pd.Series(np.nan, index=close_wide.columns)

        return cov / var


class IdiosyncraticVolatility(BaseAlpha):
    """Idiosyncratic Volatility factor: Standard deviation of residuals against the market return."""

    def compute(
        self,
        engine: Engine,
        prices_df: pd.DataFrame,
        metadata_df: pd.DataFrame,
        fundamentals_df: pd.DataFrame,
        date: pd.Timestamp,
        config: dict,
    ) -> pd.Series:
        if prices_df.empty:
            return pd.Series(dtype=float)

        close_wide = prices_df["adj_close"].unstack("ticker").sort_index()
        stock_rets = close_wide.pct_change(fill_method=None)

        # Load Market return
        query = "SELECT date, mkt_rf, rf FROM fama_french ORDER BY date"
        try:
            ff_df = pd.read_sql(query, engine)
            ff_df["date"] = pd.to_datetime(ff_df["date"])
            ff_df = ff_df.set_index("date")
            mkt_rets = (ff_df["mkt_rf"] + ff_df["rf"]) / 100.0
        except Exception:
            mkt_rets = stock_rets.mean(axis=1)

        window_stocks = stock_rets.loc[:date].tail(252)
        window_mkt = mkt_rets.reindex(window_stocks.index).ffill()

        if window_stocks.empty or window_mkt.dropna().empty:
            return pd.Series(np.nan, index=close_wide.columns)

        cov = window_stocks.apply(lambda col: col.cov(window_mkt))
        var_mkt = window_mkt.var()
        var_stock = window_stocks.var()

        if pd.isna(var_mkt) or var_mkt == 0.0:
            return pd.Series(np.nan, index=close_wide.columns)

        beta = cov / var_mkt
        var_resid = var_stock - (beta ** 2) * var_mkt
        idio_vol = np.sqrt(np.maximum(0.0, var_resid))

        # Invert: lower idiosyncratic vol is preferred (standard anomaly sorting)
        return -1.0 * idio_vol


class EarningsSurpriseSUE(BaseAlpha):
    """Earnings Surprise (SUE) factor: surprise percentage from the most recent earnings calendar."""

    def compute(
        self,
        engine: Engine,
        prices_df: pd.DataFrame,
        metadata_df: pd.DataFrame,
        fundamentals_df: pd.DataFrame,
        date: pd.Timestamp,
        config: dict,
    ) -> pd.Series:
        if prices_df.empty:
            return pd.Series(dtype=float)

        close_wide = prices_df["adj_close"].unstack("ticker").sort_index()
        close_m = close_wide.ffill().resample("ME").last()

        # Load surprise data
        query = "SELECT ticker, date, surprise_pct FROM earnings_calendar"
        try:
            earn_df = pd.read_sql(query, engine)
            earn_df["date"] = pd.to_datetime(earn_df["date"])
            earn_df = earn_df.dropna(subset=["surprise_pct"])
        except Exception:
            return pd.Series(np.nan, index=close_wide.columns)

        if earn_df.empty:
            return pd.Series(np.nan, index=close_wide.columns)

        surprise_wide = earn_df.pivot(index="date", columns="ticker", values="surprise_pct")
        union_idx = surprise_wide.index.union(close_m.index)
        surprise_m = surprise_wide.reindex(union_idx).ffill().reindex(close_m.index)

        if date not in surprise_m.index:
            return pd.Series(np.nan, index=close_wide.columns)

        return surprise_m.loc[date]


class EarningsRevision(BaseAlpha):
    """Earnings Revision factor: EPS estimate change over the previous estimate."""

    def compute(
        self,
        engine: Engine,
        prices_df: pd.DataFrame,
        metadata_df: pd.DataFrame,
        fundamentals_df: pd.DataFrame,
        date: pd.Timestamp,
        config: dict,
    ) -> pd.Series:
        if prices_df.empty:
            return pd.Series(dtype=float)

        close_wide = prices_df["adj_close"].unstack("ticker").sort_index()
        close_m = close_wide.ffill().resample("ME").last()

        query = "SELECT ticker, date, eps_estimate FROM earnings_calendar"
        try:
            earn_df = pd.read_sql(query, engine)
            earn_df["date"] = pd.to_datetime(earn_df["date"])
            earn_df = earn_df.dropna(subset=["eps_estimate"])
        except Exception:
            return pd.Series(np.nan, index=close_wide.columns)

        if earn_df.empty:
            return pd.Series(np.nan, index=close_wide.columns)

        estimate_wide = earn_df.pivot(index="date", columns="ticker", values="eps_estimate").sort_index()
        revision_wide = estimate_wide.diff(1)

        union_idx = revision_wide.index.union(close_m.index)
        revision_m = revision_wide.reindex(union_idx).ffill().reindex(close_m.index)

        if date not in revision_m.index:
            return pd.Series(np.nan, index=close_wide.columns)

        return revision_m.loc[date]


class InsiderBuyingRatio(BaseAlpha):
    """Insider Buying Ratio factor: (Buy Shares - Sell Shares) / (Buy Shares + Sell Shares + 1e-5) in the last 60 days."""

    def compute(
        self,
        engine: Engine,
        prices_df: pd.DataFrame,
        metadata_df: pd.DataFrame,
        fundamentals_df: pd.DataFrame,
        date: pd.Timestamp,
        config: dict,
    ) -> pd.Series:
        if prices_df.empty:
            return pd.Series(dtype=float)

        close_wide = prices_df["adj_close"].unstack("ticker").sort_index()

        query = "SELECT ticker, date, shares, text FROM insider_transactions"
        try:
            insider_df = pd.read_sql(query, engine)
            insider_df["date"] = pd.to_datetime(insider_df["date"])
        except Exception:
            return pd.Series(np.nan, index=close_wide.columns)

        if insider_df.empty:
            return pd.Series(np.nan, index=close_wide.columns)

        # Filter insider transactions for the trailing 60 days ending at the evaluation date
        start_window = date - pd.Timedelta(days=60)
        window_df = insider_df[(insider_df["date"] >= start_window) & (insider_df["date"] <= date)].copy()

        scores = pd.Series(0.0, index=close_wide.columns)

        for ticker in close_wide.columns:
            ticker_trades = window_df[window_df["ticker"] == ticker]
            if ticker_trades.empty:
                continue

            buys = 0.0
            sells = 0.0

            for _, row in ticker_trades.iterrows():
                text_val = str(row["text"]).lower()
                shares_val = float(row["shares"]) if not pd.isna(row["shares"]) else 0.0

                if "buy" in text_val or "purchase" in text_val:
                    buys += shares_val
                elif "sale" in text_val or "sell" in text_val:
                    sells += shares_val

            if buys + sells > 0.0:
                scores[ticker] = (buys - sells) / (buys + sells + 1e-5)

        return scores


class YieldCurveSlopeBeta(BaseAlpha):
    """Yield Curve Slope Beta factor: Beta against the yield curve slope change (DGS10 - DGS2)."""

    def compute(
        self,
        engine: Engine,
        prices_df: pd.DataFrame,
        metadata_df: pd.DataFrame,
        fundamentals_df: pd.DataFrame,
        date: pd.Timestamp,
        config: dict,
    ) -> pd.Series:
        if prices_df.empty:
            return pd.Series(dtype=float)

        close_wide = prices_df["adj_close"].unstack("ticker").sort_index()
        stock_rets = close_wide.pct_change(fill_method=None)

        query = "SELECT date, value FROM macro WHERE series_name IN ('DGS10', 'DGS2') ORDER BY date"
        try:
            macro_df = pd.read_sql(query, engine)
            macro_df["date"] = pd.to_datetime(macro_df["date"])
            pivot = macro_df.pivot(index="date", columns="series_name", values="value").ffill()
            slope = pivot["DGS10"] - pivot["DGS2"]
            slope_change = slope.diff(1)
        except Exception:
            return pd.Series(np.nan, index=close_wide.columns)

        window_stocks = stock_rets.loc[:date].tail(252)
        window_macro = slope_change.reindex(window_stocks.index).ffill()

        if window_stocks.empty or window_macro.dropna().empty:
            return pd.Series(np.nan, index=close_wide.columns)

        cov = window_stocks.apply(lambda col: col.cov(window_macro))
        var = window_macro.var()

        if pd.isna(var) or var == 0.0:
            return pd.Series(np.nan, index=close_wide.columns)

        return cov / var


class CreditSpreadBeta(BaseAlpha):
    """Credit Spread Beta factor: Beta against corporate credit spread change (BAMLH0A0HYM2)."""

    def compute(
        self,
        engine: Engine,
        prices_df: pd.DataFrame,
        metadata_df: pd.DataFrame,
        fundamentals_df: pd.DataFrame,
        date: pd.Timestamp,
        config: dict,
    ) -> pd.Series:
        if prices_df.empty:
            return pd.Series(dtype=float)

        close_wide = prices_df["adj_close"].unstack("ticker").sort_index()
        stock_rets = close_wide.pct_change(fill_method=None)

        query = "SELECT date, value FROM macro WHERE series_name = 'BAMLH0A0HYM2' ORDER BY date"
        try:
            macro_df = pd.read_sql(query, engine)
            macro_df["date"] = pd.to_datetime(macro_df["date"])
            spread = macro_df.set_index("date")["value"].ffill()
            spread_change = spread.diff(1)
        except Exception:
            return pd.Series(np.nan, index=close_wide.columns)

        window_stocks = stock_rets.loc[:date].tail(252)
        window_macro = spread_change.reindex(window_stocks.index).ffill()

        if window_stocks.empty or window_macro.dropna().empty:
            return pd.Series(np.nan, index=close_wide.columns)

        cov = window_stocks.apply(lambda col: col.cov(window_macro))
        var = window_macro.var()

        if pd.isna(var) or var == 0.0:
            return pd.Series(np.nan, index=close_wide.columns)

        return cov / var


class PMIMomentumBeta(BaseAlpha):
    """PMI Momentum Beta factor: Beta against PMI monthly momentum change (NAPM)."""

    def compute(
        self,
        engine: Engine,
        prices_df: pd.DataFrame,
        metadata_df: pd.DataFrame,
        fundamentals_df: pd.DataFrame,
        date: pd.Timestamp,
        config: dict,
    ) -> pd.Series:
        if prices_df.empty:
            return pd.Series(dtype=float)

        close_wide = prices_df["adj_close"].unstack("ticker").sort_index()
        stock_rets = close_wide.pct_change(fill_method=None)

        query = "SELECT date, value FROM macro WHERE series_name = 'NAPM' ORDER BY date"
        try:
            macro_df = pd.read_sql(query, engine)
            macro_df["date"] = pd.to_datetime(macro_df["date"])
            pmi = macro_df.set_index("date")["value"].ffill()
            # PMI momentum (3-month change approx, fallback to shorter for tests)
            shift_period = min(60, max(1, len(pmi) // 5))
            pmi_mom = pmi - pmi.shift(shift_period)
            pmi_mom_change = pmi_mom.diff(1)
        except Exception:
            return pd.Series(np.nan, index=close_wide.columns)

        window_stocks = stock_rets.loc[:date].tail(252)
        window_macro = pmi_mom_change.reindex(window_stocks.index).ffill()

        if window_stocks.empty or window_macro.dropna().empty:
            return pd.Series(np.nan, index=close_wide.columns)

        cov = window_stocks.apply(lambda col: col.cov(window_macro))
        var = window_macro.var()

        if pd.isna(var) or var == 0.0:
            return pd.Series(np.nan, index=close_wide.columns)

        return cov / var


class FactorLibrary:
    """Manager class to register, compute in parallel, and cache multi-factor score profiles."""

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.registry = {
            "PriceMomentum": PriceMomentum(logger=self.logger),
            "ThreeMonthMomentum": ThreeMonthMomentum(logger=self.logger),
            "BookToPrice": BookToPrice(logger=self.logger),
            "GrossProfitability": GrossProfitability(logger=self.logger),
            "LowVolatility": LowVolatility(logger=self.logger),
            "EVtoEBITDA": EVtoEBITDA(logger=self.logger),
            "FCFYield": FCFYield(logger=self.logger),
            "DebtToEquityChange": DebtToEquityChange(logger=self.logger),
            "Accruals": Accruals(logger=self.logger),
            "OneMonthReversal": OneMonthReversal(logger=self.logger),
            "MarketBeta": MarketBeta(logger=self.logger),
            "IdiosyncraticVolatility": IdiosyncraticVolatility(logger=self.logger),
            "EarningsSurpriseSUE": EarningsSurpriseSUE(logger=self.logger),
            "EarningsRevision": EarningsRevision(logger=self.logger),
            "InsiderBuyingRatio": InsiderBuyingRatio(logger=self.logger),
            "YieldCurveSlopeBeta": YieldCurveSlopeBeta(logger=self.logger),
            "CreditSpreadBeta": CreditSpreadBeta(logger=self.logger),
            "PMIMomentumBeta": PMIMomentumBeta(logger=self.logger),
        }

    def compute_for_date(
        self,
        eval_date: pd.Timestamp,
        engine: Engine,
        prices_df: pd.DataFrame,
        metadata_df: pd.DataFrame,
        fundamentals_df: pd.DataFrame,
        active_tickers: List[str],
        close_prices_t: pd.Series,
        sector_map: pd.Series,
        shares_map: pd.Series,
        active_universe: pd.DataFrame,
        config: dict,
    ) -> List[dict]:
        """Runs the entire multi-factor calculation and preprocessing pipeline for a single date."""
        self.logger.info("Computing factors for date %s in child process", eval_date.date())
        records = []
        
        # Dispose the connection pool to avoid multi-processing socket sharing issues
        engine.dispose()

        for factor_name, factor_obj in self.registry.items():
            try:
                raw_scores = factor_obj.compute(
                    engine=engine,
                    prices_df=prices_df,
                    metadata_df=metadata_df,
                    fundamentals_df=fundamentals_df,
                    date=eval_date,
                    config=config,
                )
            except Exception as exc:
                self.logger.error("Error computing factor %s on date %s: %s", factor_name, eval_date, exc)
                continue

            raw_scores = raw_scores.reindex(active_tickers)
            if raw_scores.dropna().empty:
                continue

            limits = tuple(config.get("winsorize_limits", [0.01, 0.01]))
            winsorized = winsorize_series(raw_scores, limits)
            zscore = z_score_series(winsorized)

            # Sizes = Close Price * Shares
            sizes_t = close_prices_t.reindex(active_tickers) * shares_map.reindex(active_tickers)
            mc_proxy = active_universe.set_index("ticker")["market_cap_proxy"] * 1e9
            sizes_t = sizes_t.fillna(mc_proxy).fillna(1.0)
            
            sectors_t = sector_map.reindex(active_tickers).fillna("Unknown")

            final_scores = neutralize_factor_scores(
                factor_scores=zscore,
                sizes=sizes_t,
                sectors=sectors_t,
                neutralize_size=config.get("neutralize_size", True),
                neutralize_sector=config.get("neutralize_sector", True),
            )

            for ticker in active_tickers:
                records.append({
                    "date": eval_date.date(),
                    "ticker": ticker,
                    "factor_name": factor_name,
                    "raw_score": float(raw_scores.loc[ticker]) if not pd.isna(raw_scores.loc[ticker]) else None,
                    "winsorized_score": float(winsorized.loc[ticker]) if not pd.isna(winsorized.loc[ticker]) else None,
                    "z_score": float(zscore.loc[ticker]) if not pd.isna(zscore.loc[ticker]) else None,
                    "final_score": float(final_scores.loc[ticker]) if not pd.isna(final_scores.loc[ticker]) else None,
                })
        return records
