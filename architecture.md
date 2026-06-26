# Architecture

## Current State

The repository currently implements the full Week 1 data foundation and the core Week 2 research plumbing for the `Regime-Adaptive Multi-Factor Alpha Engine`.

The system is structured as a config-driven Python pipeline with PostgreSQL as the system of record.

## High-Level Flow

1. `src/main.py`
2. `src/data/ingestion.py`
3. `src/factors/returns.py`
4. `src/factors/universe.py`
5. `src/factors/analysis.py`

Operationally, the flow is:

- load runtime settings from `config/config.yaml`
- initialize logging via `src/utils/logger.py`
- create or rebuild database tables
- download Yahoo Finance OHLCV data
- download FRED macro series
- upsert raw datasets into PostgreSQL
- compute daily and monthly returns
- compute point-in-time universe membership
- build and trim the Week 2 return matrix
- generate cross-sectional statistics and annual tail diagnostics
- write processed Week 2 artifacts to `data/processed/week2`

## Module Responsibilities

### `src/config.py`

- Defines `AppConfig`
- Defines download, logging, universe, and Week 2 config models
- Loads YAML config with Pydantic validation

### `src/data/downloader.py`

- Yahoo Finance downloader
- Batched multi-ticker fetch
- Retry handling
- Progress logging
- Returns a clean `(date, ticker)` indexed price frame

### `src/data/fred_client.py`

- FRED downloader
- Retry handling
- Date normalization
- Forward-fill for macro series gaps

### `src/data/db.py`

- SQLAlchemy table definitions
- PostgreSQL engine creation
- Schema creation and rebuild helpers
- Generic upsert logic

Implemented tables:

- `prices`
- `macro`
- `returns`
- `universe_membership`

### `src/data/ingestion.py`

- Orchestrates Week 1 ingestion
- Handles stock and macro branches independently
- Summarizes rows written and failure state

### `src/factors/returns.py`

- Computes daily log returns
- Computes monthly simple and log returns
- Runs a lightweight benchmark sanity check
- Stores return series into `returns`

### `src/factors/universe.py`

- Computes point-in-time universe membership
- Uses only historical information available at each date
- Applies:
- market-cap proxy filter
- dollar-volume percentile filter
- IPO age exclusion
- Stores results into `universe_membership`

### `src/factors/return_matrix.py`

- Wraps a wide return matrix
- Supports gap handling
- Supports universe masking
- Computes cross-sectional stats by date
- Computes annual skewness and kurtosis diagnostics

### `src/factors/analysis.py`

- Builds the Week 2 research outputs from stored returns and universe membership
- Aligns universe membership to return dates
- Produces cleaned and trimmed return matrices
- Produces CSV and PNG artifacts for exploratory analysis

### `src/data/fama_french_client.py`

- Handles downloading and parsing of daily Fama-French 5-factor returns

### `src/factors/base.py`

- Defines abstract `BaseAlpha` factor class

### `src/factors/library.py`

- Multi-factor library containing the 5 alpha implementations (Price Momentum, 3-Month Momentum, Book-to-Price, Gross Profitability, Low Volatility)
- Features robust handling of missing inputs per ticker

### `src/factors/neutralization.py`

- Implements factor preprocessing pipeline: Winsorize ➔ Z-score ➔ Statsmodels OLS Size and Sector Neutralization

### `src/factors/evaluation.py`

- Evaluates raw and finalized factor scores separately
- Computes time-series IC, aggregate ICIR, quintile simple returns, annualized Sharpe ratios, and decay half-lives
- Persists all results to `factor_metrics` table

## Week-by-Week Architecture Status

### Week 1

Implemented:

- config system
- logging
- Yahoo price ingestion
- FRED macro ingestion
- PostgreSQL schema and upserts
- single entrypoint orchestration

Not implemented from the broader Week 1 production ideal:

- environment lock tooling such as Poetry or conda lock
- CI and pre-commit automation
- schema diagram generation
- operational metadata tables for job auditability

### Week 2

Implemented:

- daily return calculation
- monthly return calculation
- return persistence
- point-in-time universe construction
- return matrix abstraction
- cross-sectional diagnostics
- saved Week 2 analysis artifacts

Not implemented yet:

- richer benchmark validation against an external reference index workflow
- automated plotting notebook refresh
- formal data quality reporting layer

### Week 3

Implemented:

- abstract base alpha factor class
- 5 base factors library
- winsorization and Z-score standardization helpers
- statsmodels OLS size and sector neutralization pipeline
- factor performance evaluation harness (IC, ICIR, Sharpe, half-life)
- factor score intermediate stages and time-series metrics persistence

### Week 4+

Not started in architecture terms:

- Week 4+ factor library expansion (SUE, alternative data)
- backtesting engine (Week 5)
- XGBoost combo model (Week 6)
- regime detection (Week 7)
- adaptive weighting (Week 8)
- CVaR risk optimization (Week 9)
- deployment surfaces (Weeks 10-12)

## Artifact Outputs

Current Week 2 analysis writes:

- `data/processed/week2/return_matrix_cleaned.csv`
- `data/processed/week2/return_matrix_trimmed.csv`
- `data/processed/week2/cross_sectional_stats.csv`
- `data/processed/week2/annual_stats.csv`
- `data/processed/week2/latest_cross_section_hist.png`
- `data/processed/week2/annual_tail_diagnostics.png`

Current Week 3 factor engine writes:

- `data/processed/week3/factor_scores.csv`
- `data/processed/week3/factor_evaluation_summary.csv`

