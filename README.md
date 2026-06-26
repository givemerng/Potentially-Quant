# Regime-Adaptive Multi-Factor Alpha Engine

This repository currently covers the **Week 1 foundation**, **Week 2 returns and universe workflow**, and the **Week 3 factor engineering foundation** from the 12-week implementation plan.

The project ingests historical equity OHLCV data from Yahoo Finance, macro series from FRED, stores both in PostgreSQL with idempotent upserts, computes daily and monthly returns, builds a point-in-time universe membership table, and runs a modular, config-driven factor calculation, neutralization, and evaluation pipeline.

## Implemented Scope

- **Week 1**: Config-driven ingestion pipeline, PostgreSQL schema and upserts, centralized logging, and reproducible entrypoint.
- **Week 2**: Daily and monthly return calculation, point-in-time universe membership filtering, return matrix construction/masking, and cross-sectional / annual distribution diagnostics.
- **Week 3**: Abstract base alpha class, 5 base factors (Price Momentum, 3-Month Momentum, Book-to-Price, Gross Profitability, Low Volatility), winsorization/Z-score normalization, statsmodels OLS size/sector neutralization, and factor evaluation harness (time-series IC, ICIR, Quintile Sharpe, autocorrelation decay half-life).

## Repository Layout

- `config/config.yaml`: Universe, date range, FRED series, downloader settings, and Week 2/3 pipeline parameters
- `src/config.py`: Pydantic-backed config models (including Week 3 configs)
- `src/data/downloader.py`: Yahoo Finance downloader with parallel fetching, metadata, and fundamentals downloader
- `src/data/fred_client.py`: FRED downloader with retries and forward-filled normalization
- `src/data/fama_french_client.py`: Client to download and parse Fama-French 5-factor returns
- `src/data/db.py`: Database schema (including fundamentals, metadata, factors, metrics tables) and PostgreSQL upsert helpers
- `src/data/ingestion.py`: Week 1 and 3 data ingestion pipeline orchestrator
- `src/factors/base.py`: Abstract `BaseAlpha` factor class
- `src/factors/library.py`: Multi-factor library containing the 5 alpha implementations
- `src/factors/neutralization.py`: Factor preprocessing pipeline (Winsorize, Z-score, statsmodels OLS size/sector neutralization)
- `src/factors/evaluation.py`: Factor performance evaluation harness
- `src/factors/returns.py`: Daily and monthly return computation plus validation
- `src/factors/universe.py`: Point-in-time universe construction
- `src/factors/return_matrix.py`: Return matrix helpers for masking and distribution stats
- `src/factors/analysis.py`: Week 2 return-matrix analysis and artifact generation
- `src/main.py`: Orchestrates the entire Week 1, 2, and 3 quant pipeline
- `notebooks/week3_factor_evaluation.ipynb`: Diagnostic notebook for raw vs. final factor IC, rolling stats, quintile returns, and decay curves
- `tests/`: Focused unit tests covering all components, including test_factors_week3.py

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

- Week 4+ factor expansion (SUE, alternative sentiment, macro-linked factors, Edgar data)
- Backtesting engine (Week 5)
- XGBoost combination model & ML integration (Week 6)
- Hidden Markov Model (HMM) regime detection (Week 7)
- Regime-adaptive weighted composite model (Week 8)
- CVaR portfolio optimization (Week 9)
- API, dashboard, and deployment layers (Weeks 10-12)

## Reproducibility Notes

- The pipeline is config-driven and rerunnable.
- Database writes use PostgreSQL upserts to avoid duplicate inserts.
- Stock and macro ingestion fail independently so one broken source does not terminate the entire run.
- Universe construction is point-in-time and avoids future data leakage by using only information available on or before each date.
