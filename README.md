# Regime-Adaptive Multi-Factor Alpha Engine

This repository currently covers the full **Week 1 foundation** and the core **Week 2 return and universe workflow** from the 12-week implementation plan.

The project ingests historical equity OHLCV data from Yahoo Finance, macro series from FRED, stores both in PostgreSQL with idempotent upserts, computes daily and monthly returns, builds a point-in-time universe membership table, and generates Week 2 cross-sectional return matrix artifacts.

## Implemented Scope

- Week 1: config-driven ingestion pipeline
- Week 1: PostgreSQL schema creation and upserts
- Week 1: centralized logging and reproducible entrypoint
- Week 2: daily and monthly return calculation
- Week 2: point-in-time universe membership filtering
- Week 2: return matrix construction and masking by universe
- Week 2: cross-sectional distribution statistics and annual tail diagnostics

## Repository Layout

- `config/config.yaml`: universe, date range, FRED series, downloader settings, Week 2 settings
- `src/config.py`: Pydantic-backed config models
- `src/data/downloader.py`: Yahoo Finance downloader with batching, retries, and progress bars
- `src/data/fred_client.py`: FRED downloader with retries and forward-filled normalization
- `src/data/db.py`: database schema and PostgreSQL upsert helpers
- `src/data/ingestion.py`: Week 1 ingestion pipeline
- `src/factors/returns.py`: daily and monthly return computation plus validation
- `src/factors/universe.py`: point-in-time universe construction
- `src/factors/return_matrix.py`: return matrix helpers for masking and distribution stats
- `src/factors/analysis.py`: Week 2 return-matrix analysis and artifact generation
- `src/main.py`: orchestrates Week 1 ingestion and Week 2 processing
- `tests/`: focused unit tests for config, download normalization, returns, return matrix, universe logic, and Week 2 analysis

## Database Tables

- `prices`: OHLCV and adjusted close by `date, ticker`
- `macro`: macro time series values by `date, series_name`
- `returns`: daily and monthly returns by `date, ticker, freq`
- `universe_membership`: point-in-time universe flags and liquidity metrics by `date, ticker`

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

- Week 3 factor classes and factor library
- IC / ICIR evaluation harness
- factor neutralization
- backtesting engine
- regime detection
- adaptive composite model
- API, dashboard, and deployment layers

## Reproducibility Notes

- The pipeline is config-driven and rerunnable.
- Database writes use PostgreSQL upserts to avoid duplicate inserts.
- Stock and macro ingestion fail independently so one broken source does not terminate the entire run.
- Universe construction is point-in-time and avoids future data leakage by using only information available on or before each date.
