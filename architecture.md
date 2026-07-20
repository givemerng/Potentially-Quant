# Architecture

## Current State

The repository implements the full Weeks 1-8 scope of the `Regime-Adaptive Multi-Factor Alpha Engine` structured as an enterprise **4-Tier Architecture** (`Domain Models` → `Repositories` → `Services` → `Dataset Builders`) backed by **Neon PostgreSQL** via `psycopg` (v3).

## High-Level 4-Tier Architecture

1. **Layer 1: Domain Models (`src/domain/`)**
   - Typed domain contracts (`TickerMetadata`, `CombinationRun`, `BacktestResult`, `RegimeWeight`, `PipelineSummary`) decoupling data persistence from application domain logic.

2. **Layer 2: Repository Layer (`src/data/repositories/`)**
   - Abstract interfaces (`src/data/repositories/interfaces.py`).
   - Generic `BaseRepository` (`src/data/repositories/base.py`) providing SQLAlchemy Core parameter execution (`fetch_dataframe`, `bulk_upsert`) and date type normalization.
   - Concrete repositories (`PricesRepository`, `MetadataRepository`, `FundamentalsRepository`, `FactorsRepository`, `RegimesRepository`, `PortfolioRepository`) with in-memory caching for metadata, macro, and Fama-French reference data.

3. **Layer 3: Application Service Layer (`src/services/`)**
   - Domain services (`MarketDataService`, `FactorService`, `RegimeService`, `PortfolioService`) orchestrating multi-repository operations, transactions, and caching.

4. **Layer 4: Specialized Dataset Builders (`src/data/dataset_builders/`)**
   - Configuration-driven dataset builders (`MLDatasetBuilder`, `HMMDatasetBuilder`, `FactorDatasetBuilder`, `BacktestDatasetBuilder`) reading exclusively from Services and Repositories to assemble ML feature matrices $X$ and forward return labels $y$.

5. **Pipeline Entrypoint (`src/main.py`)**
   - Orchestrates the full pipeline using Services and Dataset Builders.

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

### `src/backtest/`

- **`portfolio.py`**: Converts factor scores into portfolio weights, supports equal-weight long-only and long-short target construction, and handles position limit clipping and redistribution.
- **`rebalancer.py`**: Filters dates by rebalancing frequency, calculates drifted portfolio weights, and computes rebalance turnover.
- **`transaction_cost.py`**: Models linear transaction costs (fixed commissions and bid-ask spreads).
- **`metrics.py`**: Evaluates CAGR, annualized volatility, Sharpe, Sortino, Max Drawdown, Calmar, win rate, and average/annualized turnover.
- **`reporting.py`**: Generates CSV datasets, monthly/annual returns matrices, MD summaries, and equity/drawdown curve plots.
- **`engine.py`**: Orchestrates the vectorized simulation loop, aligning data matrices, running historical time-steps, and saving outcomes to the database.

### `src/combination/`

- **`base.py`**: Defines abstract `BaseComposite` class representing a generic factor combination strategy. It handles metadata persistence, rolling weights logging, score persistence, and metric tracking in a structured way.
- **`preprocessing.py`**: Implements standardized feature preparation pipeline aligning factor score matrices with forward-return vectors, handling gaps and missing data, and enforcing consistent column shapes.
- **`ic_weighted.py`**: Implements IC-weighted factor combination based on rolling historical Spearman rank correlation.
- **`fama_macbeth.py`**: Implements Fama-MacBeth two-pass regression combination where rolling cross-sectional betas act as weights.
- **`xgboost_composite.py`**: Implements walk-forward expanding-window XGBoost regressor composite, retrained at each step to prevent lookahead leakage.
- **`shap_analysis.py`**: Computes SHAP values and generates beeswarm/summary importance plots for XGBoost model diagnostics.
- **`reporting.py`**: Week 6 reporter saving method comparisons, rolling weights, and factor correlation matrices.


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

### Week 4

Implemented:

- Expanded factor library to 18 active signals (sentiment, technical, macro-linked, earnings momentum SUE, revision, insider trades)
- Parallel compute pipeline using joblib over month-end evaluation dates
- Neutralization (Winsorize -> Z-score -> OLS Size & Sector residuals)
- Factor performance evaluation (rank IC, ICIR, quintile Sharpe, signal decay half-life) and DB caching

### Week 5

Implemented:

- Vectorized simulation engine (`VectorizedBacktester`) that matches dates, checks universes, and computes portfolio states
- Rebalance schedule filtering and drift-adjusted turnover calculation
- Linear transaction cost model (fixed commission and spread)
- Position limit constraints and iterative redistribution logic
- Independent performance metrics sheet (Sharpe, Sortino, Drawdown, Calmar)
- DB writers for weights, trades, results, and metrics
- Automatic report generation (CSVs, tables, equity curves, drawdown plots) under `data/processed/week5/`

### Week 6

Implemented:
- Abstract `BaseComposite` ABC interface for factor combination methods.
- Shared `FeaturePreprocessor` pipeline ensuring data parity across models.
- Three combination strategies: IC-Weighted, Fama-MacBeth OLS, and walk-forward XGBoost.
- Rolling factor weights and Fama-MacBeth coefficients persistence.
- Complete OOS (2010-2024) performance evaluation against the full-history baseline.
- SHAP feature analysis interface + factor correlation diagnostics.

### Week 7

Implemented:
- 7-feature macro/market panel construction (VIX level/change, yield curve slope, credit spread, SPX 12M momentum, 21D realized vol, GDP YoY growth)
- Gaussian HMM training (`hmmlearn.hmm.GaussianHMM`) with 4 states and soft posterior state probabilities (`predict_proba`)
- BIC model selection evaluating 2 to 6 components
- Automatic economic regime labeling (Bull Market, Bear / High-Vol, Rate Shock / Stagnation, Neutral / Transition)
- State transition matrix $P_{ij}$ and expected state duration ($1 / (1 - P_{ii})$) calculation
- `market_regimes` table in PostgreSQL with date, regime_id, regime_label, and prob_0 to prob_3 soft probabilities
- Diagnostic plotting routines for SPX regime overlay shading, transition matrix heatmaps, and feature profile bar charts
- Comprehensive unit test suite in `tests/test_regime_week7.py`

### Week 8

Implemented:
- `RegimeFactorAnalyzer` computing rolling Spearman rank IC and rolling ICIR ($\frac{\mu_{IC}}{\sigma_{IC}}$) per factor per regime.
- `BayesianUpdater` performing Gaussian prior shrinkage ($\mu_{\text{post}} = \alpha \mu_{\text{prior}} + (1-\alpha) \mu_{\text{sample}}$) with exponential decay weighting, outputting posterior IC, variance, and effective sample size ($N_{\text{eff}}$).
- `AdaptiveWeightGenerator` blending HMM soft state probabilities $P(\text{Regime}_t = k)$, preserving directionality via $\text{sign}(\text{IC})$ score multiplier, and enforcing weight clipping limits, minimum thresholds, and sum-to-1 normalization.
- `RegimeAdaptiveComposite` inheriting from `BaseComposite`, orchestrating the modular pipeline, generating alpha scores, and persisting experiment metadata to PostgreSQL tables (`regime_factor_ic`, `regime_factor_weights`, `adaptive_runs`).
- `Week8Reporter` rendering factor x regime IC heatmaps, dynamic factor weight trajectories, posterior IC evolution curves, weight allocation heatmaps, and Markdown performance comparison teardown matrices.
- Comprehensive unit test suite in `tests/test_regime_adaptive_week8.py` (5/5 passed).

### Week 9+

Not started in architecture terms:

- CVaR risk optimization (Week 9)
- API, dashboard, and deployment layers (Weeks 10-12)


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

Current Week 6 combination engine writes:

- `data/processed/week6/composite_comparison.csv`
- `data/processed/week6/composite_comparison.md`
- `data/processed/week6/factor_correlation_matrix.png`
- `data/processed/week6/factor_weights_ic_weighted.csv`
- `data/processed/week6/factor_weights_fama_macbeth.csv`
- `data/processed/week6/fama_macbeth_coefficients.csv`
- `data/processed/week6/shap_values.csv`
- `data/processed/week6/shap_summary.png`
- `data/processed/week6/shap_beeswarm.png`
- `data/processed/week6/xgb_model.json`
- `data/processed/week6/{ic_weighted, fama_macbeth, xgboost}/*` backtest reports and plots

Current Week 7 regime engine writes:

- `data/processed/week7/spx_regimes.png`
- `data/processed/week7/transition_matrix.png`
- `data/processed/week7/feature_profiles.png`

Current Week 8 adaptive engine writes:

- `data/processed/week8/factor_regime_ic_heatmap.png`
- `data/processed/week8/regime_adaptive_weights.png`
- `data/processed/week8/posterior_ic_evolution.png`
- `data/processed/week8/dynamic_weight_allocation_heatmap.png`
- `data/processed/week8/week8_composite_comparison.csv`
- `data/processed/week8/week8_composite_comparison.md`



