# Context

## Project

`Regime-Adaptive Multi-Factor Alpha Engine`

This repository follows a 12-week quant research build plan. The current codebase has completed the main Week 1 foundation work and the core Week 2 return and universe workflow.

## Objective of Current Build

The current implementation establishes the research data backbone needed before factor research begins:

- ingest historical equity OHLCV data
- ingest macro series
- persist both in PostgreSQL
- compute return series
- compute point-in-time universe membership
- generate the first return-matrix diagnostics for cross-sectional research

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

## What Is Not Done Yet

### Week 2 gaps that still exist operationally

- no runtime-verified end-to-end DB execution was completed in this session
- environment has local binary package issues around installed `pandas` dependencies
- no CI workflow exists yet

### Week 3 and beyond

- factor library
- IC / ICIR analytics
- factor neutralization
- backtester
- regime detection
- adaptive portfolio logic
- API / dashboard / deployment

## Current Execution Entry Point

- `python -m src.main`

This currently runs:

1. Week 1 ingestion
2. Week 2 return calculation
3. Week 2 universe construction
4. Week 2 return-matrix analysis

## Current Database Model

- `prices`
- `macro`
- `returns`
- `universe_membership`

## Current Processed Outputs

Under `data/processed/week2`:

- cleaned return matrix
- trimmed return matrix
- cross-sectional stats
- annual tail diagnostics data
- histogram image
- annual diagnostics image

## Important Implementation Notes

- the system is config-driven
- DB writes are idempotent via upserts
- universe construction is point-in-time
- stock and macro ingestion fail independently
- Week 2 analysis aligns universe membership to return dates before masking
