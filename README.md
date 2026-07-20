# Regime-Adaptive Multi-Factor Alpha Engine

## What This Project Does

### The Core Idea

In quantitative investing, **factors** (signals like momentum, value, quality) are used to rank and select stocks. The problem is that not all factors work all the time:

- During a **bull market**, momentum signals work great — stocks that have been rising tend to keep rising.
- During a **crash** (2008, COVID), momentum gets destroyed, but "cheap" value stocks and low-volatility stocks tend to hold up better.

Most quant systems ignore this and treat every signal equally all the time. This project says: **detect what kind of market we're in, and adjust which signals we trust accordingly.**

### How It Works

1. **Get the Data** — Download historical stock prices (S&P 500, 2000–present) from Yahoo Finance and macroeconomic data from FRED. Store everything in PostgreSQL.
2. **Score Every Stock Each Month** — Compute 18 different factor scores (momentum, value, quality, sentiment, macro exposure, etc.) for every stock, then clean them by removing size and sector biases.
3. **Detect the Market Regime** *(Weeks 7–8)* — Use a Hidden Markov Model (HMM) to classify the market into states (bull, bear, high-vol, rate-shock) and dynamically adjust factor weights based on which factors historically perform best in the current regime.
4. **Build a Portfolio & Simulate Trading** — Convert factor scores into portfolio weights, simulate trading with realistic transaction costs (commissions, spreads, turnover), and measure performance (Sharpe, Sortino, Max Drawdown, Calmar).

### The Research Question

> **Can a system that detects the market regime and adapts its factor weights beat a system that treats all factors equally?**

The target is a **net Sharpe ratio > 0.8** on 2010–2024 out-of-sample data, proving that regime-adaptive weighting outperforms static equal-weight composites even after realistic trading costs.

### Tech Stack

Python · Pandas · NumPy · scikit-learn · hmmlearn · XGBoost · CVXPY · statsmodels · FastAPI · PostgreSQL · Redis · Streamlit · Docker

---

This repository covers **Weeks 1 to 7** of the 12-week implementation plan.

The project ingests historical equity OHLCV data from Yahoo Finance, macro series from FRED, stores both in PostgreSQL with idempotent upserts, computes daily and monthly returns, builds a point-in-time universe membership table, and runs a parallel config-driven factor calculation, neutralization, evaluation, vectorized backtesting pipeline, and market regime detection model.

## Implemented Scope

- **Week 1**: Config-driven ingestion pipeline, PostgreSQL schema and upserts, centralized logging, and reproducible entrypoint.
- **Week 2**: Daily and monthly return calculation, point-in-time universe membership filtering, return matrix construction/masking, and cross-sectional / annual distribution diagnostics.
- **Week 3**: Abstract base alpha class, 5 base factors (Price Momentum, 3-Month Momentum, Book-to-Price, Gross Profitability, Low Volatility), winsorization/Z-score normalization, statsmodels OLS size/sector neutralization, and factor evaluation harness (time-series IC, ICIR, Quintile Sharpe, autocorrelation decay half-life).
- **Week 4**: Expanded factor library to 18 active signals (sentiment, technical, macro-linked, revisions, insider transactions) computed in parallel with joblib over historical month-ends, cached in PostgreSQL, and evaluated for rank IC/ICIR, Sharpe, and signal decay.
- **Week 5**: Vectorized portfolio backtesting engine supporting commissions and bid-ask spread models, equal-weight long-only and long-short target weight construction, position limit clipping/redistribution, performance metrics calculations (Sharpe, Sortino, Max Drawdown, Calmar, win rate, turnover), database persistence, and reporting layouts (monthly/annual grids, equity curves, drawdown charts).
- **Week 6**: Factor combination and ML integration with three composite strategies (IC-weighted, Fama-MacBeth OLS, XGBoost walk-forward), `BaseComposite` ABC interface, shared feature preprocessing pipeline, experiment tracking with `combination_runs` table, rolling factor weight persistence, SHAP feature importance analysis, factor correlation diagnostics, and dual metric reporting (full + OOS 2010–2024).
- **Week 7**: Market regime detection using Gaussian Hidden Markov Models (HMM) fitted on a 7-feature macro/market panel (VIX level/change, yield curve slope, credit spread, SPX 12M momentum, 21D realized volatility, GDP YoY growth), automated BIC state selection, economic regime labeling (Bull Market, Bear / High-Vol, Rate Shock / Stagnation, Neutral / Transition), state transition dynamics, database persistence (`market_regimes` table), and visualization routines (SPX regime shading, transition matrix heatmaps, feature profiles).

## Repository Layout

- `config/config.yaml`: Universe, date range, FRED series, downloader settings, and Week 2/3/5/6/7 pipeline parameters
- `src/config.py`: Pydantic-backed config models (including Week 7 configs)
- `src/data/downloader.py`: Yahoo Finance downloader with parallel fetching, metadata, and fundamentals downloader
- `src/data/fred_client.py`: FRED downloader with retries and forward-filled normalization
- `src/data/fama_french_client.py`: Client to download and parse Fama-French 5-factor returns
- `src/data/db.py`: Database schema (including fundamentals, metadata, factors, metrics, composite, and market regimes tables) and PostgreSQL upsert helpers
- `src/data/ingestion.py`: Week 1 and 3 data ingestion pipeline orchestrator
- `src/factors/base.py`: Abstract `BaseAlpha` factor class
- `src/factors/library.py`: Multi-factor library containing the alpha implementations
- `src/factors/neutralization.py`: Factor preprocessing pipeline (Winsorize, Z-score, statsmodels OLS size/sector neutralization)
- `src/factors/evaluation.py`: Factor performance evaluation harness
- `src/factors/returns.py`: Daily and monthly return computation plus validation
- `src/factors/universe.py`: Point-in-time universe construction
- `src/factors/return_matrix.py`: Return matrix helpers for masking and distribution stats
- `src/factors/analysis.py`: Week 2 return-matrix analysis and artifact generation
- `src/backtest/`: Week 5 vectorized portfolio backtesting engine, rebalancer, metrics, transaction costs, and reporting modules
- `src/combination/`: Week 6 factor combination algorithms (IC-weighted, Fama-MacBeth, XGBoost), feature preprocessor, SHAP analyzer, and reporting modules
- `src/regime/detector.py`: MarketRegimeDetector class for GaussianHMM fitting, BIC selection, economic labeling, transition matrices, and DB persistence
- `src/regime/visualization.py`: Regime visualization routines (SPX regime overlay, transition matrix heatmaps, feature profile bar charts)
- `src/regime/__init__.py`: Regime detection module exports
- `src/main.py`: Orchestrates the entire quant pipeline (Weeks 1 to 7)
- `notebooks/week3_factor_evaluation.ipynb`: Diagnostic notebook for raw vs. final factor IC, rolling stats, quintile returns, and decay curves
- `tests/`: Focused unit tests covering all components, including test_factors_week3.py, test_factors_week4.py, test_backtester.py, test_combination_week6.py, and test_regime_week7.py

## Database Tables

- `prices`: OHLCV and adjusted close by `date, ticker`
- `macro`: Macro time series values by `date, series_name`
- `returns`: Daily and monthly returns by `date, ticker, freq`
- `universe_membership`: Point-in-time universe flags and liquidity metrics by `date, ticker`
- `ticker_metadata`: Sector, industry, and shares outstanding by `ticker`
- `fundamentals`: Balance sheet and income statement items (Book Value, Gross Profit, Total Assets, EPS) by `date, ticker`
- `factors`: Raw, winsorized, z-score, and neutralized scores by `date, ticker, factor_name`
- `factor_metrics`: Time-series and summary performance metrics (IC, ICIR, Sharpe, decay) by `date, factor_name, stage, metric_name`
- `fama_french`: Daily Fama-French 5-factor returns by `date`
- `combination_runs`: Experiment tracking with `run_id`, method, config hash, train/test windows, and timestamp
- `composite_scores`: Composite signal scores by `date, ticker, method` linked to `run_id`
- `combination_weights`: Rolling factor weights by `date, factor_name, method` linked to `run_id`
- `combination_metrics`: Performance metrics by `run_id, method, metric_name` with `full` and `oos` scopes
- `market_regimes`: Daily hard regime IDs, economic labels, and soft posterior probabilities (`prob_0` to `prob_3`) by `date`

## Run

1. Create a Python 3.10+ virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Update `.env` with PostgreSQL credentials and your `FRED_API_KEY`.
4. Adjust `config/config.yaml` for the target universe and date range.
5. Run the pipeline:

```bash
python -m src.main
```

This will:
- create or rebuild the tracked database tables
- download and store price data
- download and store macro data
- compute daily and monthly returns
- compute point-in-time universe membership
- build the Week 2 return matrix
- generate cross-sectional stats and annual fat-tail diagnostics
- write processed artifacts under `data/processed/week2`
- compute 18 alpha factors in parallel using joblib over month-end dates
- winsorize, Z-score standardize, and OLS neutralize factor scores by sector and size
- evaluate rank IC/ICIR, quintile Sharpe ratios, and signal decay half-lives
- write evaluation results to database tables (`factors`, `factor_metrics`) and CSV files under `data/processed/week3`
- run vectorized backtests for all factors, enforcing position limits and exposures
- apply linear transaction costs (commissions and spreads) and rebalancing turnover tracking
- save weights, trades, results, and metrics to database tables (`portfolio_weights`, `trades`, `backtest_results`, `backtest_metrics`)
- output backtest reports (CSVs, markdown summary matrices, performance plots) under `data/processed/week5/`
- prepare shared feature matrix from all 18 neutralized factor scores
- run three composite strategies: IC-weighted, Fama-MacBeth OLS, and XGBoost walk-forward
- backtest each composite through the Week 5 engine and compare full vs. OOS (2010–2024) performance
- run SHAP feature importance analysis on the XGBoost model
- generate factor correlation matrix, composite comparison tables, and SHAP plots under `data/processed/week6/`
- construct 7-feature macro/market panel, run BIC selection over 2 to 6 components, fit 4-state GaussianHMM, auto-assign economic regime labels, calculate transition dynamics, save regime assignments to `market_regimes` DB table, and output SPX regime overlay plots, transition matrix heatmaps, and feature profile charts under `data/processed/week7/`

## Week 2 Artifacts

The Week 2 analysis writes:

- `data/processed/week2/return_matrix_cleaned.csv`
- `data/processed/week2/return_matrix_trimmed.csv`
- `data/processed/week2/cross_sectional_stats.csv`
- `data/processed/week2/annual_stats.csv`
- `data/processed/week2/latest_cross_section_hist.png`
- `data/processed/week2/annual_tail_diagnostics.png`

## Rebuild Mode

Set `rebuild_on_run: true` in `config/config.yaml` to drop and recreate the tracked tables before ingestion.

## Not Yet Implemented

- Regime-adaptive weighted composite model (Week 8)
- CVaR portfolio risk optimization (Week 9)
- API, dashboard, and deployment layers (Weeks 10-12)

## Reproducibility Notes

- The pipeline is config-driven and rerunnable.
- Database writes use PostgreSQL upserts to avoid duplicate inserts.
- Stock and macro ingestion fail independently so one broken source does not terminate the entire run.
- Universe construction is point-in-time and avoids future data leakage by using only information available on or before each date.
