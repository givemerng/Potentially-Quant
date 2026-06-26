# Memory

## Current Progress Snapshot

The repository is at:

- Week 1: implemented
- Week 2: core workflow implemented
- Week 3: factor engine implemented
- Week 4+: not started

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
- `config/config.yaml`
- `notebooks/week3_factor_evaluation.ipynb`
- `tests/test_factors_week3.py`

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

- start Week 4 factor expansion
- implement Earnings Momentum (SUE)
- implement macro-exposure signals, EV/EBITDA, Accruals, and insider metrics
- implement parallel registry class (`FactorLibrary`) using joblib

### Before deeper feature work

- set up a local PostgreSQL database server to run end-to-end main execution runs
- run the full test suite (`python -m pytest`) to ensure local edits remain regression-free
- clean runtime build artifacts (`__pycache__`) before commits
