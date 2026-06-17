# Regime-Adaptive Multi-Factor Alpha Engine

Week 1 foundation for a reproducible quant research data pipeline. The project downloads historical equity OHLCV data from Yahoo Finance, macro series from FRED, and stores both in PostgreSQL using idempotent upserts.

## Features

- Reproducible config-driven ingestion with `config/config.yaml`
- Optional one-command schema rebuild with `rebuild_on_run: true`
- Parallel Yahoo Finance downloader with retries and progress bars
- FRED macro downloader with forward-fill normalization
- PostgreSQL schema creation and upserts via SQLAlchemy
- Centralized logging to console and file
- Single entrypoint to rebuild the dataset from scratch

## Project Layout

- `config/config.yaml`: universe, date range, FRED series, downloader settings
- `src/config.py`: pydantic-backed config loader
- `src/data/downloader.py`: Yahoo Finance downloader
- `src/data/fred_client.py`: FRED downloader
- `src/data/db.py`: database schema and upsert helpers
- `src/data/ingestion.py`: pipeline orchestration
- `src/main.py`: command-line entrypoint
- `src/utils/logger.py`: logger setup
- `tests/`: focused unit tests for config and download normalization

## Setup

1. Create a Python 3.10+ virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Update `.env` with PostgreSQL credentials and your `FRED_API_KEY`.
4. Adjust `config/config.yaml` for your target universe and dates.

## Run

```bash
python -m src.main
```

This command creates the schema if needed, downloads data, upserts into PostgreSQL, and prints a row-count summary.

Set `rebuild_on_run: true` in `config/config.yaml` when you want the pipeline to drop and recreate the target tables before ingestion.

## PostgreSQL Tables

`prices`

- `date`, `ticker` composite primary key
- `open`, `high`, `low`, `close`, `adj_close`, `volume`

`macro`

- `date`, `series_name` composite primary key
- `value`

## Reproducibility Notes

- The pipeline is config-driven and idempotent.
- Inserts use PostgreSQL upsert logic to avoid duplicates.
- Download retries reduce transient API failures.
- The pipeline isolates stock and macro ingestion so one failure does not terminate the other branch.
