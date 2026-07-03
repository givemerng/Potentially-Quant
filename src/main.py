from pathlib import Path
import numpy as np
import pandas as pd
from sqlalchemy import text

from src.config import AppConfig
from src.data.db import get_engine, factors_table, factor_metrics_table, upsert_rows
from src.data.ingestion import DataIngestionPipeline
from src.factors.analysis import Week2AnalysisPipeline
from src.factors.returns import ReturnCalculator
from src.factors.universe import UniverseFilter
from src.factors.neutralization import winsorize_series, z_score_series, neutralize_factor_scores
from src.factors.library import FactorLibrary
from src.factors.evaluation import FactorEvaluator
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

    print("\nWeek 1 + Week 2 + Week 3 pipeline complete.")



if __name__ == "__main__":
    main()
