from pathlib import Path
import numpy as np
import pandas as pd
from sqlalchemy import text
from joblib import Parallel, delayed

from src.config import AppConfig
from src.data.db import get_engine, factors_table, factor_metrics_table, upsert_rows
from src.data.ingestion import DataIngestionPipeline
from src.factors.analysis import Week2AnalysisPipeline
from src.factors.returns import ReturnCalculator
from src.factors.universe import UniverseFilter
from src.factors.neutralization import winsorize_series, z_score_series, neutralize_factor_scores
from src.factors.library import FactorLibrary
from src.factors.evaluation import FactorEvaluator
from src.backtest.engine import VectorizedBacktester
from src.backtest.reporting import Reporter
from src.utils.logger import setup_logger


def _load_metadata_from_db(engine) -> pd.DataFrame:
    query = text("SELECT ticker, sector, industry, shares_outstanding FROM ticker_metadata")
    with engine.connect() as conn:
        return pd.read_sql(query, conn)


def _load_fundamentals_from_db(engine) -> pd.DataFrame:
    query = text("SELECT date, ticker, book_value, gross_profit, total_assets, eps, ebitda, total_debt, operating_cash_flow, capital_expenditures, net_income FROM fundamentals")
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df



def _load_prices_from_db(engine, config: AppConfig) -> pd.DataFrame:
    query = text(
        """
        SELECT date, ticker, open, high, low, close, adj_close, volume
        FROM prices
        WHERE date >= :start AND date <= :end
        ORDER BY date, ticker
        """
    )
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"start": config.start_date, "end": config.end_date})

    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"])
    return df.set_index(["date", "ticker"]).sort_index()


def _load_returns_from_db(engine, freq: str) -> pd.DataFrame:
    query = text(
        """
        SELECT date, ticker, log_return, simple_return
        FROM returns
        WHERE freq = :freq
        ORDER BY date, ticker
        """
    )
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"freq": freq})

    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"])
    return df.set_index(["date", "ticker"]).sort_index()


def _load_universe_membership_from_db(engine) -> pd.DataFrame:
    query = text(
        """
        SELECT date, ticker, in_universe, market_cap_proxy, avg_dollar_vol, days_since_first_price
        FROM universe_membership
        ORDER BY date, ticker
        """
    )
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)

    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"])
    return df


def main() -> None:
    config = AppConfig.from_yaml(Path("config/config.yaml"))
    logger = setup_logger("quant_pipeline", level=config.logging.level, log_file=config.logging.file)
    engine = get_engine()

    logger.info("Starting data ingestion pipeline")
    pipeline = DataIngestionPipeline(config=config, logger=logger)
    ingestion_summary = pipeline.run()

    print("\n=== Week 1 - Data Ingestion Summary ===")
    print(f"Tickers:              {ingestion_summary.ticker_count}")
    print(f"Date range:           {ingestion_summary.start_date} -> {ingestion_summary.end_date}")
    print(f"Price rows upserted:  {ingestion_summary.price_rows_upserted}")
    print(f"Macro rows upserted:  {ingestion_summary.macro_rows_upserted}")
    print(f"Total rows upserted:  {ingestion_summary.total_rows_upserted}")
    print(f"Status:               {'FAILED_WITH_ERRORS' if ingestion_summary.failed else 'SUCCESS'}")

    if ingestion_summary.failed:
        logger.error("Ingestion pipeline failed; skipping Week 2 stages.")
        return

    logger.info("Loading prices from DB for Week 2 computations")
    prices = _load_prices_from_db(engine, config)
    if prices.empty:
        logger.error("No prices found in DB; cannot proceed with Week 2 stages.")
        return

    logger.info(
        "Loaded %d price rows for %d tickers",
        len(prices),
        prices.index.get_level_values("ticker").nunique(),
    )

    logger.info("Starting return calculation")
    return_calculator = ReturnCalculator(logger=logger)
    return_summary = return_calculator.run(engine=engine, prices=prices)

    print("\n=== Week 2a - Return Calculation Summary ===")
    print(f"Daily return rows stored:    {return_summary.daily_rows_stored}")
    print(f"Monthly return rows stored:  {return_summary.monthly_rows_stored}")
    print(f"Validation passed:           {return_summary.validation_passed}")
    print(f"Validation message:          {return_summary.validation_message}")

    logger.info("Starting universe membership computation")
    universe_filter = UniverseFilter(logger=logger)
    universe_summary = universe_filter.run(engine=engine, prices=prices, config=config.universe_filter)

    print("\n=== Week 2b - Universe Construction Summary ===")
    print(f"Total (date, ticker) pairs evaluated: {universe_summary.total_ticker_dates}")
    print(f"In-universe pairs:                    {universe_summary.in_universe_ticker_dates}")
    print(f"Inclusion rate:                       {universe_summary.inclusion_rate:.1%}")
    print(f"Universe membership rows stored:      {universe_summary.rows_stored}")

    logger.info("Starting return-matrix analysis")
    stored_returns = _load_returns_from_db(engine=engine, freq=config.week2.return_frequency)
    stored_membership = _load_universe_membership_from_db(engine)
    analysis_pipeline = Week2AnalysisPipeline(logger=logger)
    analysis_summary = analysis_pipeline.run(
        returns_df=stored_returns,
        membership_df=stored_membership,
        config=config.week2,
    )

    print("\n=== Week 2c - Return Matrix & Distribution Analysis Summary ===")
    print(f"Return matrix shape:                  {analysis_summary.return_matrix_shape}")
    print(f"Non-null trimmed observations:        {analysis_summary.trimmed_matrix_non_null}")
    print(f"Cross-sectional dates analyzed:       {analysis_summary.cross_sectional_dates}")
    print(f"Annual stats rows:                    {analysis_summary.annual_stats_rows}")
    print(f"Artifacts written:                    {analysis_summary.artifacts_written}")

    logger.info("Starting Week 4 - Parallel Factor Calculation and Preprocessing Pipeline")
    metadata_df = _load_metadata_from_db(engine)
    fundamentals_df = _load_fundamentals_from_db(engine)
    library = FactorLibrary(logger=logger)

    # Gather evaluation dates (month-ends from stored returns)
    monthly_returns = _load_returns_from_db(engine=engine, freq="monthly")
    if monthly_returns.empty:
        logger.error("No monthly returns found in DB; cannot compute factors.")
        return

    eval_dates = sorted(monthly_returns.index.get_level_values("date").unique())
    logger.info("Computing factors over %d month-end dates", len(eval_dates))

    prices_sorted = prices.copy().sort_index()

    # Maps for neutralization lookups
    sector_map = metadata_df.set_index("ticker")["sector"]
    shares_map = metadata_df.set_index("ticker")["shares_outstanding"]

    # Build tasks for Parallel execution
    tasks = []
    for eval_date in eval_dates:
        active_universe = stored_membership[
            (stored_membership["date"] == eval_date.date()) & (stored_membership["in_universe"] == True)
        ]
        if active_universe.empty:
            continue

        active_tickers = active_universe["ticker"].tolist()

        try:
            close_prices_t = prices_sorted.xs(eval_date, level="date")["adj_close"]
        except KeyError:
            close_prices_t = pd.Series(dtype=float)

        tasks.append(
            delayed(library.compute_for_date)(
                eval_date=eval_date,
                engine=engine,
                prices_df=prices_sorted,
                metadata_df=metadata_df,
                fundamentals_df=fundamentals_df,
                active_tickers=active_tickers,
                close_prices_t=close_prices_t,
                sector_map=sector_map,
                shares_map=shares_map,
                active_universe=active_universe,
                config=config.week3.model_dump(),
            )
        )

    # Run calculations in parallel using joblib
    all_results = Parallel(n_jobs=-1, backend="multiprocessing")(tasks)
    
    # Flatten list of lists
    all_factor_scores = [record for sublist in all_results for record in sublist]

    if not all_factor_scores:
        logger.error("No factor scores were calculated.")
        return

    # Upsert factor scores
    factor_scores_df = pd.DataFrame(all_factor_scores)
    rows_written = upsert_rows(engine, factors_table, all_factor_scores)
    logger.info("Successfully upserted %d factor scores to DB", rows_written)

    # Evaluate factors
    evaluator = FactorEvaluator(logger=logger)
    eval_results = []

    for factor_name in library.registry.keys():
        factor_subset = factor_scores_df[factor_scores_df["factor_name"] == factor_name]
        monthly_returns_reset = monthly_returns.reset_index()
        monthly_returns_reset["date"] = pd.to_datetime(monthly_returns_reset["date"])

        summary = evaluator.evaluate_factor(
            engine=engine,
            factor_name=factor_name,
            factor_scores_df=factor_subset,
            returns_df=monthly_returns_reset,
        )
        if summary:
            summary["factor_name"] = factor_name
            eval_results.append(summary)

    # Write CSV artifacts
    if config.week3.save_artifacts:
        artifact_dir = Path(config.week3.artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        factor_scores_df.to_csv(artifact_dir / "factor_scores.csv", index=False)

        if eval_results:
            eval_df = pd.DataFrame(eval_results)
            eval_df.to_csv(artifact_dir / "factor_evaluation_summary.csv", index=False)

            print("\n=== Week 4 - Factor Evaluation Summary ===")
            print(eval_df.to_string(index=False))

        logger.info("Saved Week 4 artifacts under %s", artifact_dir)

    logger.info("Starting Week 5 - Portfolio Backtesting Engine")
    backtester = VectorizedBacktester(
        initial_capital=config.week5.initial_capital,
        rebalance_frequency=config.week5.rebalance_frequency,
        weighting_method=config.week5.weighting_method,
        max_position_size=config.week5.max_position_size,
        commission_bps=config.week5.commission_bps,
        bid_ask_spread_bps=config.week5.bid_ask_spread_bps,
        long_only=config.week5.long_only,
        net_exposure=config.week5.net_exposure,
        gross_exposure=config.week5.gross_exposure,
        logger=logger,
    )
    reporter = Reporter(logger=logger)
    backtest_summaries = []

    # Run backtests for all calculated factors
    for factor_name in library.registry.keys():
        factor_subset = factor_scores_df[factor_scores_df["factor_name"] == factor_name]
        if factor_subset.empty:
            continue

        backtest_name = f"{factor_name}_{config.week5.weighting_method}"

        # Run portfolio simulation
        summary = backtester.run(
            factor_scores_df=factor_subset,
            returns_df=monthly_returns.reset_index(),
            universe_membership_df=stored_membership,
            backtest_name=backtest_name,
        )

        if summary:
            # Save results to PostgreSQL database
            backtester.save_results(engine=engine, run_summary=summary)

            # Generate and save report artifacts (CSVs, markdown summary, plots)
            if config.week5.save_artifacts:
                factor_artifact_dir = Path(config.week5.artifact_dir) / factor_name
                reporter.generate_report(run_summary=summary, artifact_dir=factor_artifact_dir)

            backtest_summaries.append({
                "factor_name": factor_name,
                "cumulative_return_net": summary["metrics"].get("cumulative_return_net"),
                "sharpe_ratio": summary["metrics"].get("sharpe_ratio"),
                "max_drawdown": summary["metrics"].get("max_drawdown"),
                "annualized_turnover": summary["metrics"].get("annualized_turnover"),
            })

    if backtest_summaries:
        backtest_summary_df = pd.DataFrame(backtest_summaries)
        print("\n=== Week 5 - Vectorized Backtest Summary ===")
        print(backtest_summary_df.to_string(index=False))

        if config.week5.save_artifacts:
            backtest_summary_df.to_csv(Path(config.week5.artifact_dir) / "backtest_performance_summary.csv", index=False)

    # ======================================================================
    # Week 6 – Factor Combination & ML Integration
    # ======================================================================
    logger.info("Starting Week 6 - Factor Combination & ML Integration")

    from src.combination.preprocessing import FeaturePreprocessor
    from src.combination.ic_weighted import ICWeightedComposite
    from src.combination.fama_macbeth import FamaMacBethComposite
    from src.combination.xgboost_composite import XGBoostComposite
    from src.combination.shap_analysis import SHAPAnalyzer
    from src.combination.reporting import Week6Reporter
    from src.data.db import (
        combination_runs_table,
        composite_scores_table,
        combination_weights_table,
        combination_metrics_table,
    )

    week6_config = config.week6.model_dump()

    # 1. Prepare shared feature matrix from factor scores + monthly returns
    preprocessor = FeaturePreprocessor(logger=logger)
    monthly_returns_reset = monthly_returns.reset_index()
    monthly_returns_reset["date"] = pd.to_datetime(monthly_returns_reset["date"])

    try:
        feature_panel, fwd_returns, eval_dates, factor_names = preprocessor.prepare(
            factor_scores_df=factor_scores_df,
            returns_df=monthly_returns_reset,
            score_column="final_score",
        )
    except ValueError as exc:
        logger.error("Cannot run Week 6: %s", exc)
        print("\nWeek 1 + Week 2 + Week 3 + Week 4 + Week 5 pipeline complete.")
        return

    # 2. Determine OOS boundary
    oos_start = pd.Timestamp(config.week6.oos_start_date)
    oos_dates = [d for d in eval_dates if d >= oos_start]
    logger.info("Week 6: %d total eval dates, %d OOS dates (>= %s)",
                len(eval_dates), len(oos_dates), oos_start.date())

    # 3. Run each combination method
    composite_methods = {
        "ic_weighted": ICWeightedComposite(logger=logger),
        "fama_macbeth": FamaMacBethComposite(logger=logger),
        "xgboost": XGBoostComposite(logger=logger),
    }

    week6_reporter = Week6Reporter(logger=logger)
    method_all_scores: dict = {}
    method_metrics: dict = {}

    for method_name in config.week6.combination_methods:
        if method_name not in composite_methods:
            logger.warning("Unknown combination method: %s", method_name)
            continue

        model = composite_methods[method_name]
        model._generate_run_id(week6_config)
        logger.info("Running %s composite (run_id=%s)", method_name, model.run_id)

        all_composite_scores = []

        for eval_date in eval_dates[:-1]:  # skip last date (no forward return)
            model.fit(feature_panel, fwd_returns, eval_date, week6_config)
            preds = model.predict(feature_panel, eval_date)
            if preds.empty:
                continue

            for ticker, score in preds.items():
                all_composite_scores.append({
                    "date": eval_date,
                    "ticker": ticker,
                    "factor_name": f"{method_name}_composite",
                    "raw_score": float(score) if pd.notna(score) else None,
                    "final_score": float(score) if pd.notna(score) else None,
                })

        if not all_composite_scores:
            logger.warning("No composite scores generated for %s", method_name)
            continue

        composite_df = pd.DataFrame(all_composite_scores)
        method_all_scores[method_name] = composite_df

        # Save composite scores to DB
        scores_for_db = composite_df.rename(columns={"final_score": "composite_score"})
        model.save_composite_scores(engine, scores_for_db[["date", "ticker", "composite_score"]])

        # Save factor weights (IC-weighted and Fama-MacBeth)
        if hasattr(model, "get_weights_dataframe"):
            weights_df = model.get_weights_dataframe()
            if not weights_df.empty:
                model.save_factor_weights(engine, weights_df)
                if config.week6.save_artifacts:
                    week6_reporter.save_factor_weights_csv(
                        weights_df, method_name, config.week6.artifact_dir)

        # Save FM coefficients
        if method_name == "fama_macbeth" and hasattr(model, "get_coefficients_dataframe"):
            coeff_df = model.get_coefficients_dataframe()
            if not coeff_df.empty and config.week6.save_artifacts:
                week6_reporter.save_fm_coefficients_csv(coeff_df, config.week6.artifact_dir)

        # Backtest the composite
        backtest_name = f"{method_name}_composite_{config.week5.weighting_method}"
        composite_summary = backtester.run(
            factor_scores_df=composite_df,
            returns_df=monthly_returns.reset_index(),
            universe_membership_df=stored_membership,
            backtest_name=backtest_name,
        )

        if composite_summary:
            backtester.save_results(engine=engine, run_summary=composite_summary)
            full_metrics = composite_summary.get("metrics", {})

            # Compute OOS metrics (filter results to OOS period)
            oos_metrics = {}
            if "daily_returns" in composite_summary:
                oos_rets = composite_summary["daily_returns"]
                if hasattr(oos_rets, "index"):
                    oos_mask = oos_rets.index >= oos_start
                    oos_series = oos_rets.loc[oos_mask]
                    if len(oos_series) > 0:
                        from src.backtest.metrics import PortfolioMetrics
                        pm = PortfolioMetrics()
                        oos_metrics = pm.compute(oos_series)

            # Save metrics to DB
            model.save_run_metadata(
                engine=engine,
                config=week6_config,
                train_start=eval_dates[0].date() if eval_dates else None,
                train_end=eval_dates[-1].date() if eval_dates else None,
                test_start=oos_dates[0].date() if oos_dates else None,
                test_end=oos_dates[-1].date() if oos_dates else None,
            )
            model.save_metrics(engine, full_metrics, scope="full")
            if oos_metrics:
                model.save_metrics(engine, oos_metrics, scope="oos")

            method_metrics[method_name] = {
                "full_sharpe": full_metrics.get("sharpe_ratio"),
                "full_max_dd": full_metrics.get("max_drawdown"),
                "oos_sharpe": oos_metrics.get("sharpe_ratio"),
                "oos_max_dd": oos_metrics.get("max_drawdown"),
            }

            # Generate report artifacts
            if config.week6.save_artifacts:
                factor_artifact_dir = Path(config.week6.artifact_dir) / method_name
                reporter.generate_report(run_summary=composite_summary, artifact_dir=factor_artifact_dir)

    # 4. SHAP analysis on XGBoost model
    if "xgboost" in composite_methods and composite_methods["xgboost"].get_model() is not None:
        try:
            xgb_model = composite_methods["xgboost"]
            analyzer = SHAPAnalyzer(logger=logger)
            X_latest = feature_panel.loc[eval_dates[-2]]
            shap_df = analyzer.analyze(xgb_model.get_model(), X_latest, factor_names)

            if config.week6.save_artifacts:
                artifact_dir = Path(config.week6.artifact_dir)
                analyzer.save_plots(X_latest, artifact_dir)
                week6_reporter.save_shap_values_csv(shap_df, artifact_dir)

                # Save model
                xgb_model.save_model(str(artifact_dir / "xgb_model.json"))
        except Exception as exc:
            logger.warning("SHAP analysis failed: %s", exc)

    # 5. Factor correlation matrix (diagnostic, no filtering)
    if config.week6.save_artifacts:
        week6_reporter.plot_factor_correlation_matrix(
            feature_panel, config.week6.artifact_dir)

    # 6. Comparison report
    if method_metrics and config.week6.save_artifacts:
        week6_reporter.generate_comparison_report(method_metrics, config.week6.artifact_dir)

    if method_metrics:
        print("\n=== Week 6 - Composite Method Comparison ===")
        comp_df = pd.DataFrame([
            {"method": m, **v} for m, v in method_metrics.items()
        ])
        print(comp_df.to_string(index=False))

    print("\nWeek 1 + Week 2 + Week 3 + Week 4 + Week 5 + Week 6 pipeline complete.")


if __name__ == "__main__":
    main()

