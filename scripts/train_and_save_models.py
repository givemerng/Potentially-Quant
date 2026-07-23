"""Standalone Training & Model Serialization Script.

Trains all models on historical Neon PostgreSQL database contents and serializes fitted model artifacts
(XGBoost Regressor, Gaussian HMM Regime Detector, Bayesian Adaptive Factor Model, Risk Covariance)
to `data/models/` for offline inference, testing, and production API serving.
"""

from __future__ import annotations
import os
import sys
import logging
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

# Add repository root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import AppConfig
from src.services import MarketDataService, FactorService, RegimeService, PortfolioService
from src.combination.preprocessing import FeaturePreprocessor
from src.combination.xgboost_composite import XGBoostComposite
from src.regime.detector import MarketRegimeDetector
from src.regime.bayesian_updater import BayesianUpdater
from src.portfolio.risk_models.covariance import LedoitWolfCovariance
from src.utils.logger import setup_logger


def train_and_save_all_models():
    logger = setup_logger("train_save_models", level="INFO")
    logger.info("Initializing Model Training & Serialization Pipeline...")

    config_path = Path("config/config.yaml")
    config = AppConfig.from_yaml(config_path)

    output_dir = Path("data/models")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Instantiate services
    market_data_service = MarketDataService()
    factor_service = FactorService()
    regime_service = RegimeService()
    portfolio_service = PortfolioService()

    # =========================================================================
    # 1. Load Data from Neon PostgreSQL
    # =========================================================================
    logger.info("1/5 Loading historical dataset from Neon PostgreSQL...")
    prices = market_data_service.get_price_history(start_date=config.start_date, end_date=config.end_date)
    monthly_returns = market_data_service.get_returns_matrix(freq="monthly")
    stored_membership = market_data_service.get_universe_membership()
    macro_series = factor_service.get_macro_series()
    metadata_df = market_data_service.get_ticker_metadata()

    if prices.empty or monthly_returns.empty:
        logger.error("Database returns/prices are empty. Ensure ingestion pipeline has run.")
        return

    # Load factor scores from DB
    from src.data.db import get_engine
    from sqlalchemy import text
    with get_engine().connect() as conn:
        factor_scores_df = pd.read_sql(text("SELECT date, ticker, factor_name, final_score FROM factors"), con=conn)

    if factor_scores_df.empty:
        logger.error("No factor scores found in database.")
        return

    # =========================================================================
    # 2. Fit & Serialize XGBoost Composite Model
    # =========================================================================
    logger.info("2/5 Training XGBoost Composite Alpha Model...")
    preprocessor = FeaturePreprocessor(logger=logger)
    monthly_returns_reset = monthly_returns.reset_index()
    monthly_returns_reset["date"] = pd.to_datetime(monthly_returns_reset["date"])

    feature_panel, fwd_returns, eval_dates, factor_names = preprocessor.prepare(
        factor_scores_df=factor_scores_df,
        returns_df=monthly_returns_reset,
        score_column="final_score",
    )

    xgb_composite = XGBoostComposite(logger=logger)
    week6_cfg = config.week6.model_dump()
    latest_eval_date = eval_dates[-2] if len(eval_dates) >= 2 else eval_dates[-1]

    xgb_composite.fit(feature_panel, fwd_returns, latest_eval_date, week6_cfg)
    fitted_xgb_model = xgb_composite.get_model()

    xgb_pickle_path = output_dir / "xgboost_model.pkl"
    xgb_json_path = output_dir / "xgboost_model.json"
    joblib.dump(fitted_xgb_model, xgb_pickle_path)
    xgb_composite.save_model(str(xgb_json_path))
    logger.info("Saved XGBoost model to %s and %s", xgb_pickle_path, xgb_json_path)

    # =========================================================================
    # 3. Fit & Serialize Gaussian HMM Market Regime Detector
    # =========================================================================
    logger.info("3/5 Fitting Gaussian Hidden Markov Model (HMM) Regime Classifier...")
    if not macro_series.empty:
        macro_pivoted = macro_series.pivot(index="date", columns="series_name", values="value")
        macro_pivoted.index = pd.to_datetime(macro_pivoted.index)
    else:
        macro_pivoted = None

    regime_detector = MarketRegimeDetector(
        n_components=config.week7.n_components,
        covariance_type=config.week7.covariance_type,
        n_iter=config.week7.n_iter,
        random_state=config.week7.random_state,
    )
    regime_features = regime_detector.build_feature_matrix(prices.reset_index(), macro_pivoted)
    state_series, prob_df = regime_detector.fit(regime_features)

    hmm_pickle_path = output_dir / "hmm_regime_detector.pkl"
    joblib.dump({
        "hmm_model": regime_detector.model,
        "scaler": regime_detector.scaler,
        "regime_labels": regime_detector.regime_labels,
        "feature_names": list(regime_features.columns),
    }, hmm_pickle_path)
    logger.info("Saved Gaussian HMM model to %s", hmm_pickle_path)

    # =========================================================================
    # 4. Fit & Serialize Bayesian Regime-Adaptive Factor Weights
    # =========================================================================
    logger.info("4/5 Fitting Bayesian Regime-Adaptive Factor Model...")
    regimes_df = regime_service.get_market_regimes()
    if not regimes_df.empty:
        regimes_df["date"] = pd.to_datetime(regimes_df["date"])
        regimes_indexed = regimes_df.set_index("date")

        updater = BayesianUpdater(
            decay_halflife=config.week8.decay_halflife,
            prior_weight=config.week8.prior_weight,
        )

        bayesian_pickle_path = output_dir / "bayesian_regime_model.pkl"
        joblib.dump({
            "updater": updater,
            "regime_labels": regime_detector.regime_labels,
            "factor_names": factor_names,
        }, bayesian_pickle_path)
        logger.info("Saved Bayesian Regime Model to %s", bayesian_pickle_path)

    # =========================================================================
    # 5. Fit & Serialize Ledoit-Wolf Risk & Covariance Model
    # =========================================================================
    logger.info("5/5 Fitting Ledoit-Wolf Risk Covariance Estimator...")
    pivoted_rets = monthly_returns.reset_index().pivot(index="date", columns="ticker", values="simple_return")
    lw_estimator = LedoitWolfCovariance(logger=logger)
    cov_matrix_df = lw_estimator.estimate(pivoted_rets)

    risk_pickle_path = output_dir / "risk_covariance_model.pkl"
    joblib.dump({
        "covariance_matrix": cov_matrix_df,
        "tickers": list(cov_matrix_df.columns),
        "estimator_type": "ledoit_wolf",
    }, risk_pickle_path)
    logger.info("Saved Risk Covariance Model to %s", risk_pickle_path)

    # =========================================================================
    # Master Pipeline Bundle Serialization
    # =========================================================================
    master_bundle_path = output_dir / "master_quant_pipeline.pkl"
    master_bundle = {
        "xgboost_model": fitted_xgb_model,
        "hmm_regime_detector": regime_detector.model,
        "scaler": regime_detector.scaler,
        "regime_labels": regime_detector.regime_labels,
        "feature_names": factor_names,
        "covariance_matrix": cov_matrix_df,
        "stock_universe": config.stock_universe,
        "version": "1.0.0",
        "timestamp": pd.Timestamp.now().isoformat(),
    }
    joblib.dump(master_bundle, master_bundle_path)
    logger.info("SUCCESS: Master Quant Pipeline model bundle serialized to %s", master_bundle_path)

    print("\n" + "=" * 60)
    print("      QUANT MODEL TRAINING & SERIALIZATION COMPLETE")
    print("=" * 60)
    print(f"• XGBoost Model:           {xgb_pickle_path}")
    print(f"• Gaussian HMM Classifier: {hmm_pickle_path}")
    print(f"• Bayesian Regime Model:   {output_dir / 'bayesian_regime_model.pkl'}")
    print(f"• Risk Covariance Model:   {risk_pickle_path}")
    print(f"• Master Pipeline Bundle:  {master_bundle_path}")
    print("=" * 60)


if __name__ == "__main__":
    train_and_save_all_models()
