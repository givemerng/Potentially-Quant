# Memory

## Current Progress Snapshot

The repository is at:

- Week 1: implemented
- Week 2: core workflow implemented
- Week 3: factor engine implemented
- Week 4: factor library expanded & parallelized
- Week 5: vectorized backtesting engine implemented
- Week 6: factor combination & ML integration implemented
- Week 7+: not started


## Week 1 Done

- project scaffold created
- `.gitignore` added
- config system added
- logger added
- Yahoo downloader added
- FRED downloader added
- PostgreSQL schema and upsert helpers added
- ingestion pipeline added
- `README.md` updated to reflect current scope

## Week 2 Done

- `returns` table added
- `universe_membership` table added
- `ReturnCalculator` added
- `UniverseFilter` added
- `ReturnMatrix` added
- `Week2AnalysisPipeline` added
- Week 2 artifact generation added
- tests added for returns, return matrix, universe, and Week 2 analysis

## Week 3 Done

- Extended DB schemas with `ticker_metadata`, `fundamentals`, `factors`, `factor_metrics`, and `fama_french` tables
- Extended `YahooFinanceDownloader` to pull sector/industry classifications, shares outstanding, and quarterly financial statement items in parallel
- Implemented Tuck public library client to download daily Fama-French 5-factor returns
- Created abstract `BaseAlpha` class inside `src/factors/base.py`
- Created factor preprocessing pipeline (Winsorize ➔ Z-score ➔ Statsmodels OLS size/sector neutralization) in `src/factors/neutralization.py`
- Implemented 5 base factors inside `src/factors/library.py` (Price Momentum 12-1M, 3-Month Momentum, B/P, Gross Profitability, Low Volatility)
- Implemented `FactorEvaluator` inside `src/factors/evaluation.py` to calculate raw vs. neutralized time-series IC, ICIR, Quintile Sharpe, and decay half-life
- Added interactive diagnostic notebook `week3_factor_evaluation.ipynb`
- Added comprehensive unit tests in `tests/test_factors_week3.py` (all tests passing)

## Week 4 Done

- Expanded the factor library (`src/factors/library.py`) to 18 active signals, covering value (EV/EBITDA, FCF Yield), quality/accruals (Debt/Equity change, Accruals), technicals (1M Reversal, 3M Momentum, Market Beta, Idiosyncratic Volatility), macro exposures (Yield Curve, Credit Spread, PMI Momentum), and alternative/earnings factors (Earnings Surprise SUE, Earnings Revision, Insider Buying).
- Implemented parallelized factor scoring over historical month-ends in `src/main.py` using joblib.
- Built test suites in `tests/test_factors_week4.py` (all tests passing).
- Fixed the missing joblib import in `src/main.py`.

## Week 5 Done

- Created `Week5Config` and added configurable parameters to `config.yaml`
- Created `PortfolioConstructor` supporting equal-weight long-only and long-short target weight construction and iterative position size limit redistribution
- Built `Rebalancer` calculating drifted portfolio weights and rebalancing turnovers
- Created linear transaction cost model (`LinearTransactionCostModel`) using commissions and spread models
- Built vectorized metrics calculation (`PortfolioMetrics`) for returns, CAGR, volatility, Sharpe, Sortino, max drawdown, and Calmar ratios
- Created `VectorizedBacktester` simulating weights, trades, net/gross returns, value, and drawdowns, and writing results to database tables (`portfolio_weights`, `trades`, `backtest_results`, `backtest_metrics`)
- Created `Reporter` generating report CSVs, tables, summaries, and equity/drawdown curve plots under `data/processed/week5/`
- Added comprehensive unit tests in `tests/test_backtester.py` (all tests passing)

## Week 6 Done

- Integrated `BaseComposite` ABC interface for factor combination algorithms
- Built standardized `FeaturePreprocessor` pipeline for date alignment, order sorting, and median NaN imputation
- Implemented `ICWeightedComposite` using rolling trailing Spearman rank correlations (negative ICs zeroed, normalized)
- Implemented `FamaMacBethComposite` using cross-sectional OLS regressions averaged over time
- Implemented `XGBoostComposite` using strict walk-forward expanding window regressor (configurable purge gap)
- Added `SHAPAnalyzer` for explainable AI importance diagnostics (summary & beeswarm plots, exported values)
- Created `Week6Reporter` generating method comparisons, factor correlations, and rolling weights plots/CSVs
- Added experiment tracking database tables: `combination_runs`, `composite_scores`, `combination_weights`, and `combination_metrics`
- Built full diagnostic test suites in `tests/test_combination_week6.py` (all tests passing)


## Current Files That Matter Most

- `src/main.py`
- `src/config.py`
- `src/data/db.py`
- `src/data/ingestion.py`
- `src/data/downloader.py`
- `src/data/fama_french_client.py`
- `src/factors/base.py`
- `src/factors/library.py`
- `src/factors/neutralization.py`
- `src/factors/evaluation.py`
- `src/factors/returns.py`
- `src/factors/universe.py`
- `src/factors/return_matrix.py`
- `src/factors/analysis.py`
- config/config.yaml
- notebooks/week3_factor_evaluation.ipynb
- src/backtest/engine.py
- src/backtest/portfolio.py
- src/backtest/rebalancer.py
- src/backtest/transaction_cost.py
- src/backtest/metrics.py
- src/backtest/reporting.py
- tests/test_factors_week3.py
- tests/test_factors_week4.py
- tests/test_backtester.py
- src/combination/base.py
- src/combination/preprocessing.py
- src/combination/ic_weighted.py
- src/combination/fama_macbeth.py
- src/combination/xgboost_composite.py
- src/combination/shap_analysis.py
- src/combination/reporting.py
- tests/test_combination_week6.py


## Known Constraints

- local environment has a binary compatibility issue in installed packages around `NumPy` / `pandas` / `pyarrow`
- compile checks pass
- full live runtime validation has not been completed in this session against a clean env
- plotting dependencies are imported lazily inside analysis functions

## Decisions Taken

- removed `SciPy` from the Week 2 analytics path to reduce binary fragility
- kept Week 2 diagnostics file-based rather than notebook-only
- preserved point-in-time universe logic as a strict design rule
- used PostgreSQL upserts instead of append-only inserts
- added `statsmodels` dependency to support OLS size and sector neutralization
- rejected synthetic data generation for missing fundamentals; opted to store `NULL` values and skip affected tickers
- postponed Earnings Momentum (SUE) to Week 4, replacing it with a 3-Month Momentum factor for Week 3 stability
- persisted intermediate factor values (`winsorized_score`, `z_score`) alongside raw and neutralized final scores for auditability
- evaluated raw and neutralized factor series separately to calculate the marginal predictive power of the signal
- externalized all lookbacks, clipping limits, and neutralization toggles to `config.yaml`

### If continuing the roadmap

- start Week 7 market regime detection with HMM
- build a regime feature matrix (VIX level, VIX change, yield curve slope (10Y-2Y), credit spread, realized volatility, GDP YoY)
- fit Hidden Markov Model (`hmmlearn.HMM.GaussianHMM`) with 4 states
- use Bayesian Information Criterion (BIC) to select the optimal number of states
- visualize SPX price chart with regime probabilities overlaid, and verify economic alignments


### Before deeper feature work

- set up a local PostgreSQL database server to run end-to-end main execution runs
- run the full test suite (`python -m pytest`) to ensure local edits remain regression-free
- clean runtime build artifacts (`__pycache__`) before commits
