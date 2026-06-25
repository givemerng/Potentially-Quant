# Memory

## Current Progress Snapshot

The repository is at:

- Week 1: implemented
- Week 2: core workflow implemented
- Week 3+: not started

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

## Current Files That Matter Most

- `src/main.py`
- `src/config.py`
- `src/data/db.py`
- `src/data/ingestion.py`
- `src/factors/returns.py`
- `src/factors/universe.py`
- `src/factors/return_matrix.py`
- `src/factors/analysis.py`
- `config/config.yaml`

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

## Next Logical Work

### If continuing the roadmap

- start Week 3 factor scaffolding
- add `BaseAlpha`
- implement the first five factors
- add factor storage schema
- add IC / ICIR evaluation harness

### Before deeper feature work

- create a clean Python 3.10+ virtualenv
- reinstall dependencies from `requirements.txt`
- run tests with `pytest`
- run `python -m src.main` against a live PostgreSQL instance
- clean committed runtime artifacts like `__pycache__` from git if still tracked
