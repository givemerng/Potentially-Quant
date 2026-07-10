# Context

## Project

`Regime-Adaptive Multi-Factor Alpha Engine`

This repository follows a 12-week quant research build plan. The current codebase has completed the main Week 1 foundation work, the core Week 2 return and universe workflow, the Week 3 factor engineering foundation, the Week 4 factor library expansion, the Week 5 vectorized portfolio backtesting engine, and the Week 6 factor combination & ML integration.

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

### Week 4

- Expanded factor library to 18 active alphas (EV/EBITDA, FCF Yield, Debt-to-Equity Change, Accruals, 1M Reversal, 3M Momentum, Market Beta, Idiosyncratic Volatility, SUE, Earnings Revision, Insider Buying, macro betas)
- Parallelized factor calculations at month-ends using joblib Parallel/delayed
- Integrated full factor preprocessing pipeline (Winsorize -> Z-score -> OLS Size & Sector Neutralization)
- Automated Rank IC/ICIR, Sharpe, and decay computations for raw vs. neutralized factors, storing results in database
- Expanded test coverage with `test_factors_week4.py` (all tests passing)

### Week 5

- Created Pydantic model `Week5Config` for config parsing and validation
- Implemented `PortfolioConstructor` supporting equal-weight long-only and long-short target weight calculations
- Built position size limit constraints utilizing iterative redistribution logic
- Created `Rebalancer` to determine rebalancing schedules, handle drifted weights, and compute rebalance turnovers
- Created `LinearTransactionCostModel` supporting fixed commission rates and bid-ask spread models
- Built vectorized performance metrics evaluator (`PortfolioMetrics`) calculating CAGR, Sharpe, Sortino, Drawdown, Calmar, and turnovers
- Created `VectorizedBacktester` simulating weights, trade histories, gross/net returns, and values
- Added `Reporter` to save CSVs, markdown performance grids, and equity/drawdown curve plots
- Extended test coverage with `test_backtester.py` (all tests passing)

### Week 6

- Standardized `FeaturePreprocessor` pipeline for cross-sectional factor-return alignment and median NaN imputation
- Implemented abstract `BaseComposite` class supporting fit/predict models, run_id generation, and database updates
- Implemented three composite models: `ICWeightedComposite`, `FamaMacBethComposite`, and `XGBoostComposite` (strict walk-forward expanding window)
- Added rolling weights tracking for IC and Fama-MacBeth strategies
- Built `SHAPAnalyzer` for local and global model explanation (beeswarm and summary plots)
- Added `Week6Reporter` for side-by-side performance table (Sharpe, max DD) and factor correlation matrices
- Extended database schema with `combination_runs`, `composite_scores`, `combination_weights`, and `combination_metrics` tables
- Built comprehensive test suite in `tests/test_combination_week6.py` (all tests passing)


## What Is Not Done Yet

### Gaps that still exist operationally

- no runtime-verified end-to-end DB execution has been completed in this session (requires active local PostgreSQL)
- no CI workflow exists yet

### Week 7 and beyond

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
5. Week 4 parallel factor calculation (18 alphas, winsorization, z-score)
6. Week 4 statsmodels OLS size and sector neutralization
7. Week 4 factor performance evaluation (Spearman rank IC, ICIR, Quintile Sharpe, signal decay half-life)
8. Week 5 vectorized portfolio backtest simulation
9. Week 5 transaction cost estimation, exposure scaling, and weight rebalancing
10. Week 5 performance metrics generation (Sharpe, Sortino, Drawdown, Calmar) and reporting (markdown, plots)
11. Week 6 shared feature preprocessing and OOS split logic
12. Week 6 composite generation (IC-weighted, Fama-MacBeth, XGBoost) and walk-forward prediction
13. Week 6 backtesting of composite signals and dual metrics generation (full + OOS 2010–2024)
14. Week 6 model explaining (SHAP plots, features ranking) and metrics reporting (CSV, MD)


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
- `portfolio_weights`
- `trades`
- `backtest_results`
- `backtest_metrics`
- `combination_runs`
- `composite_scores`
- `combination_weights`
- `combination_metrics`


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

Under `data/processed/week6`:
- method comparison grid (`composite_comparison.csv` and `.md`)
- rolling factor weights CSVs (`factor_weights_{ic_weighted, fama_macbeth}.csv`)
- fama macbeth coefficients CSV
- SHAP values CSV and importance plots (`shap_summary.png`, `shap_beeswarm.png`)
- factor rank correlation matrix plot
- serialized XGBoost model (`xgb_model.json`)
- backtest reports and plots for all 3 composite methods

Under `notebooks/`:
- `week3_factor_evaluation.ipynb`


## Important Implementation Notes

- the system is config-driven
- DB writes are idempotent via upserts
- universe construction and factor calculations are point-in-time (no future leaks)
- stock and macro ingestion fail independently
- Week 3 neutralization applies a winsorize ➔ z-score ➔ statsmodels OLS size/sector residuals sequence
- factor evaluation harness runs separately on raw and final factor scores for comparatives
