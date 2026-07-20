from __future__ import annotations

import logging
from datetime import date as date_type
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy.engine import Engine

from src.combination.base import BaseComposite
from src.data.db import (
    adaptive_runs_table,
    combination_weights_table,
    composite_scores_table,
    regime_factor_ic_table,
    regime_factor_weights_table,
    upsert_rows,
)
from src.regime.adaptive_weights import AdaptiveWeightGenerator
from src.regime.bayesian_updater import BayesianUpdater
from src.regime.factor_analysis import RegimeFactorAnalyzer

logger = logging.getLogger(__name__)


class RegimeAdaptiveComposite(BaseComposite):
    """Regime-Adaptive Factor Combination Strategy using HMM soft state probabilities and Bayesian IC updates."""

    method_name: str = "regime_adaptive"

    def __init__(self, logger: logging.Logger | None = None) -> None:
        super().__init__(logger=logger)
        self.analyzer: Optional[RegimeFactorAnalyzer] = None
        self.updater: Optional[BayesianUpdater] = None
        self.generator: Optional[AdaptiveWeightGenerator] = None
        self._fitted_weights: Dict[pd.Timestamp, Dict[str, float]] = {}
        self._fitted_directions: Dict[pd.Timestamp, Dict[str, float]] = {}

    def fit(
        self,
        factor_scores: pd.DataFrame,
        forward_returns: pd.Series,
        eval_date: pd.Timestamp,
        config: dict,
    ) -> None:
        """Fit stub to satisfy BaseComposite abstract interface.

        Full fitting is orchestrated via ``fit_predict_full_pipeline``.
        """
        pass

    def predict(
        self,
        factor_scores: pd.DataFrame,
        eval_date: pd.Timestamp,
    ) -> pd.Series:
        """Predict stub returning composite scores for a given cross-section date."""
        if eval_date not in self._fitted_weights:
            # Fallback equal weights if date not fitted
            factors = factor_scores.columns
            return factor_scores.mean(axis=1)

        weights = self._fitted_weights[eval_date]
        directions = self._fitted_directions[eval_date]

        composite = pd.Series(0.0, index=factor_scores.index)
        for factor, w in weights.items():
            if factor in factor_scores.columns:
                direction = directions.get(factor, 1.0)
                composite += w * (direction * factor_scores[factor].fillna(0.0))
        return composite

    def fit_predict_full_pipeline(
        self,
        factors_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        regimes_df: pd.DataFrame,
        config: dict,
        engine: Optional[Engine] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Runs the modular Week 8 regime-adaptive factor pipeline.

        Args:
            factors_df: DataFrame of neutralized factor scores (MultiIndex date, ticker or long format).
            returns_df: DataFrame of returns.
            regimes_df: DataFrame of market regimes with probabilities.
            config: Week 8 config dictionary.
            engine: Optional SQLAlchemy DB Engine for persistence.

        Returns:
            Tuple of (composite_scores_df, dynamic_weights_df, posteriors_df, summary_ic_matrix)
        """
        self._generate_run_id(config)
        self.logger.info("Initializing RegimeAdaptiveComposite run_id=%s", self.run_id)

        # 1. Instantiate modules with config parameters
        ic_lookback = config.get("ic_lookback_months", 36)
        decay_hl = config.get("decay_halflife", 12.0)
        prior_w = config.get("prior_weight", 0.3)
        min_samples = config.get("min_regime_samples", 5)
        max_f_weight = config.get("max_factor_weight", 0.25)
        min_w_thresh = config.get("min_weight_threshold", 0.02)

        self.analyzer = RegimeFactorAnalyzer(ic_lookback_months=ic_lookback, min_regime_samples=min_samples)
        self.updater = BayesianUpdater(prior_weight=prior_w, decay_halflife=decay_hl)
        self.generator = AdaptiveWeightGenerator(max_factor_weight=max_f_weight, min_weight_threshold=min_w_thresh)

        # 2. Step 1: Compute cross-sectional IC and regime-conditional stats
        self.logger.info("Computing daily cross-sectional ICs...")
        ic_df = self.analyzer.compute_daily_cross_sectional_ic(factors_df, returns_df)

        self.logger.info("Computing regime-conditional ICs and rolling ICIR...")
        summary_ic_matrix, rolling_df = self.analyzer.compute_regime_conditional_stats(ic_df, regimes_df)

        # 3. Step 2: Bayesian Posterior Update
        self.logger.info("Computing Bayesian posterior IC estimates...")
        posteriors_df = self.updater.update_posteriors(rolling_df, ic_df)

        # 4. Step 3: Adaptive Weight Generation
        self.logger.info("Generating dynamic factor weights with direction sign flip...")
        weights_df, direction_df, metadata_weights_df = self.generator.generate_weights(posteriors_df, regimes_df)

        # Store internal fitted weights
        for d in weights_df.index:
            self._fitted_weights[d] = weights_df.loc[d].to_dict()
            self._fitted_directions[d] = direction_df.loc[d].to_dict()

        # 5. Step 4: Compute Composite Alpha Scores across all dates
        self.logger.info("Computing composite alpha scores...")
        if isinstance(factors_df.index, pd.MultiIndex):
            factors_flat = factors_df.reset_index()
        else:
            factors_flat = factors_df.copy()

        pivoted = factors_flat.pivot(index=["date", "ticker"], columns="factor_name", values="final_score")
        dates = sorted(pivoted.index.get_level_values("date").unique())

        composite_records = []
        for d in dates:
            if d not in weights_df.index:
                continue
            day_factors = pivoted.xs(d, level="date")
            scores_series = self.predict(day_factors, d)

            for ticker, val in scores_series.items():
                composite_records.append({
                    "date": d,
                    "ticker": ticker,
                    "composite_score": float(val) if not np.isnan(val) else 0.0,
                    "method": self.method_name,
                    "run_id": self.run_id,
                })

        composite_scores_df = pd.DataFrame(composite_records)

        # 6. Step 5: Database Persistence (if engine provided)
        if engine is not None:
            self.logger.info("Persisting Week 8 metadata and metrics to PostgreSQL...")
            # Save run metadata
            self.save_run_metadata(engine, config)

            # Save adaptive_runs metadata
            adaptive_run_record = [{
                "run_id": self.run_id,
                "hmm_model_version": "v1_gaussian_hmm",
                "ic_lookback_months": ic_lookback,
                "bayesian_prior_weight": prior_w,
                "decay_halflife": decay_hl,
                "oos_start": pd.Timestamp(config.get("oos_start_date", "2010-01-01")).date(),
                "oos_end": pd.Timestamp(config.get("oos_end_date", "2024-12-31")).date(),
                "performance_metrics": "{}",
                "created_at": date_type.today(),
            }]
            upsert_rows(engine, adaptive_runs_table, adaptive_run_record)

            # Save composite scores
            if not composite_scores_df.empty:
                self.save_composite_scores(engine, composite_scores_df)

            # Save combination weights
            comb_weight_records = []
            for _, row in metadata_weights_df.iterrows():
                comb_weight_records.append({
                    "date": pd.Timestamp(row["date"]).date(),
                    "factor_name": row["factor_name"],
                    "method": self.method_name,
                    "run_id": self.run_id,
                    "weight": float(row["dynamic_weight"]),
                })
            if comb_weight_records:
                upsert_rows(engine, combination_weights_table, comb_weight_records)

            # Save regime_factor_ic
            ic_records = []
            for _, row in posteriors_df.iterrows():
                ic_records.append({
                    "date": pd.Timestamp(row["date"]).date(),
                    "factor_name": row["factor_name"],
                    "regime_id": int(row["regime_id"]),
                    "regime_label": row["regime_label"],
                    "sample_ic": float(row["sample_ic"]) if pd.notna(row["sample_ic"]) else None,
                    "rolling_icir": float(row["rolling_icir"]) if pd.notna(row["rolling_icir"]) else None,
                    "posterior_ic": float(row["posterior_ic"]) if pd.notna(row["posterior_ic"]) else None,
                    "posterior_variance": float(row["posterior_variance"]) if pd.notna(row["posterior_variance"]) else None,
                    "effective_sample_size": float(row["effective_sample_size"]) if pd.notna(row["effective_sample_size"]) else None,
                })
            if ic_records:
                upsert_rows(engine, regime_factor_ic_table, ic_records)

            # Save regime_factor_weights
            reg_weight_records = []
            for _, row in metadata_weights_df.iterrows():
                reg_weight_records.append({
                    "date": pd.Timestamp(row["date"]).date(),
                    "factor_name": row["factor_name"],
                    "run_id": self.run_id,
                    "regime_probability": float(row["regime_probability"]),
                    "dynamic_weight": float(row["dynamic_weight"]),
                    "posterior_ic": float(row["posterior_ic"]),
                })
            if reg_weight_records:
                upsert_rows(engine, regime_factor_weights_table, reg_weight_records)

        return composite_scores_df, weights_df, posteriors_df, summary_ic_matrix
