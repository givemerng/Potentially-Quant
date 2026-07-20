# Context

## Project

`Regime-Adaptive Multi-Factor Alpha Engine`

This repository follows a 12-week quant research build plan. The codebase implements Weeks 1-8 scope refactored into an enterprise **4-Tier Architecture** (`Domain Models` → `Repositories` → `Services` → `Dataset Builders`) backed by **Neon PostgreSQL** via `psycopg` (v3).

## Objective of Current Build

The implementation establishes the research data backbone, repository abstractions, service layer, and alpha factor framework:

- ingest historical equity OHLCV data and macroeconomic time series
- ingest stock metadata (sector, industry, shares) and quarterly financials (Book Value, Gross Profit, Assets, EPS)
- persist all data points in Neon PostgreSQL with idempotent, batched upserts (`psycopg` v3)
- abstract database interactions into repository interfaces (`PricesRepository`, `MetadataRepository`, `FundamentalsRepository`, `FactorsRepository`, `RegimesRepository`, `PortfolioRepository`) and generic `BaseRepository`
- provide domain service layer (`MarketDataService`, `FactorService`, `RegimeService`, `PortfolioService`)
- assemble ML datasets via specialized builders (`MLDatasetBuilder`, `HMMDatasetBuilder`, `FactorDatasetBuilder`, `BacktestDatasetBuilder`)
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

### Week 8

- Created `Week8Config` Pydantic model and updated `config.yaml` with parameters (`ic_lookback_months`, `decay_halflife`, `prior_weight`, `max_factor_weight`, `min_weight_threshold`, `oos_start_date`, `oos_end_date`)
- Extended SQLAlchemy schema with `regime_factor_ic`, `regime_factor_weights`, and `adaptive_runs` tables in `src/data/db.py`
- Implemented `RegimeFactorAnalyzer` in `src/regime/factor_analysis.py` for rolling Spearman IC and rolling ICIR by regime
- Implemented `BayesianUpdater` in `src/regime/bayesian_updater.py` for Gaussian prior shrinkage, exponential decay, posterior IC variance, and $N_{\text{eff}}$
- Implemented `AdaptiveWeightGenerator` in `src/regime/adaptive_weights.py` for dynamic HMM soft state blending, directional sign flips ($\text{sign}(IC)$), weight clipping, thresholding, and sum-to-1 normalization
- Implemented `RegimeAdaptiveComposite` in `src/combination/regime_adaptive.py` inheriting from `BaseComposite`
- Implemented `Week8Reporter` in `src/combination/regime_reporting.py` for heatmaps, dynamic weight drift, posterior IC evolution plots, and Markdown teardowns
- Integrated Stage 8 pipeline step into `src/main.py`
- Created comprehensive unit test suite in `tests/test_regime_adaptive_week8.py` (5/5 tests passing)


## What Is Not Done Yet

### Gaps that still exist operationally

- no runtime-verified end-to-end DB execution has been completed in this session (requires active local PostgreSQL)
- no CI workflow exists yet

### Week 9 and beyond

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
15. Week 7 HMM market regime detection, state selection, economic labeling, and transition matrix calculation
16. Week 8 regime-conditional factor IC & rolling ICIR analysis, Bayesian updating, direction-aware adaptive factor weighting, dynamic `RegimeAdaptiveComposite` score computation, and full vs OOS backtest performance teardown


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
- `market_regimes`
- `regime_factor_ic`
- `regime_factor_weights`
- `adaptive_runs`


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

Under `data/processed/week7`:
- SPX regimes overlay plot (`spx_regimes.png`)
- transition matrix heatmap (`transition_matrix.png`)
- feature profiles chart (`feature_profiles.png`)

Under `data/processed/week8`:
- factor x regime IC heatmap (`factor_regime_ic_heatmap.png`)
- dynamic factor weights trajectory plot (`regime_adaptive_weights.png`)
- posterior IC evolution plot (`posterior_ic_evolution.png`)
- dynamic weight allocation heatmap (`dynamic_weight_allocation_heatmap.png`)
- strategy performance comparison tables (`week8_composite_comparison.csv` and `.md`)


Under `notebooks/`:
- `week3_factor_evaluation.ipynb`


## Important Implementation Notes

- the system is config-driven
- DB writes are idempotent via upserts
- universe construction and factor calculations are point-in-time (no future leaks)
- stock and macro ingestion fail independently
- Week 3 neutralization applies a winsorize ➔ z-score ➔ statsmodels OLS size/sector residuals sequence
- factor evaluation harness runs separately on raw and final factor scores for comparatives
