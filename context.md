# Context

## Project

`Regime-Adaptive Multi-Factor Alpha Engine`

This repository follows a 12-week quant research build plan. The current codebase has completed the main Week 1 foundation work, the core Week 2 return and universe workflow, and the Week 3 factor engineering foundation.

## Objective of Current Build

The current implementation establishes the research data backbone and the alpha factor framework:

- ingest historical equity OHLCV data and macroeconomic time series
- ingest stock metadata (sector, industry, shares) and quarterly financials (Book Value, Gross Profit, Assets, EPS)
- persist all data points in PostgreSQL with idempotent upserts
- compute point-in-time universe, returns, and wide return matrices
- calculate raw and neutralized factor scores (winsorized, z-scored, and size/sector neutralized)
- evaluate factor performance (Spearman IC, Quintile Sharpe, signal decay) and save time-series and aggregate metrics

## What Has Been Completed

### Week 1

- monorepo-style project structure created
- YAML config system implemented
- centralized logger implemented
- Yahoo Finance downloader implemented
- FRED downloader implemented
- PostgreSQL schema and upsert workflow implemented
- ingestion pipeline implemented
- single command entrypoint implemented

### Week 2

- daily log returns implemented
- monthly return series implemented
- return persistence table added
- point-in-time universe membership implemented
- return matrix abstraction implemented
- cross-sectional per-date stats implemented
- annual skewness and kurtosis diagnostics implemented
- processed CSV and PNG artifact generation implemented
- test coverage expanded for Week 2 components

### Week 3

- Pydantic config extended for factor lookbacks, winsorization, and neutralization
- SQLAlchemy schema extended with `ticker_metadata`, `fundamentals`, `factors`, `factor_metrics`, and `fama_french` tables
- parallel Yahoo metadata and fundamentals fetcher (and Tuck Fama-French daily returns client) added
- abstract `BaseAlpha` factor class and processing pipeline (Winsorize ➔ Z-score ➔ Statsmodels OLS size/sector neutralization) created
- 5 base factors implemented: Price Momentum (12-1M), 3-Month Momentum, Book-to-Price, Gross Profitability, Low Volatility (1Y)
- factor evaluation harness (`FactorEvaluator`) created for raw vs. final IC, ICIR, Quintile Sharpe, and decay half-life
- interactive diagnostic Jupyter notebook `week3_factor_evaluation.ipynb` added
- test coverage expanded with `test_factors_week3.py` (all tests passing)

## What Is Not Done Yet

### Gaps that still exist operationally

- no runtime-verified end-to-end DB execution has been completed in this session (requires active local PostgreSQL)
- no CI workflow exists yet

### Week 4 and beyond

- SUE (Earnings Momentum) integration and factor library expansion (Week 4)
- backtesting engine (Week 5)
- combination models and Fama-MacBeth OLS / XGBoost combinations (Week 6)
- HMM regime detection (Week 7)
- regime-adaptive weighting (Week 8)
- CVaR portfolio risk optimization (Week 9)
- dashboard and deployment (Weeks 10-12)

## Current Execution Entry Point

- `python -m src.main`

This currently runs:

1. Week 1 ingestion (OHLCV prices, macro series, stock metadata, quarterly financials, Fama-French factors)
2. Week 2 return calculation (daily log, monthly simple/log returns)
3. Week 2 universe construction (PIT filtering)
4. Week 2 return-matrix analysis (diagnostics, cross-sectional stats)
5. Week 3 factor calculation (5 base factors, winsorization, z-score)
6. Week 3 statsmodels OLS size and sector neutralization
7. Week 3 factor performance evaluation (Spearman IC, Quintile Sharpe, autocorrelation decay half-life)

## Current Database Model

- `prices`
- `macro`
- `returns`
- `universe_membership`
- `ticker_metadata`
- `fundamentals`
- `factors`
- `factor_metrics`
- `fama_french`

## Current Processed Outputs

Under `data/processed/week2`:
- cleaned return matrix
- trimmed return matrix
- cross-sectional stats
- annual tail diagnostics data
- histogram image
- annual diagnostics image

Under `data/processed/week3`:
- factor scores (all intermediate stages)
- factor evaluation summary metrics

Under `notebooks/`:
- `week3_factor_evaluation.ipynb`

## Important Implementation Notes

- the system is config-driven
- DB writes are idempotent via upserts
- universe construction and factor calculations are point-in-time (no future leaks)
- stock and macro ingestion fail independently
- Week 3 neutralization applies a winsorize ➔ z-score ➔ statsmodels OLS size/sector residuals sequence
- factor evaluation harness runs separately on raw and final factor scores for comparatives
