from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy import text

from src.config import AppConfig
from src.data.db import create_schema, get_engine, rebuild_schema
from src.data.ingestion import DataIngestionPipeline
from src.factors.returns import ReturnCalculator
from src.factors.universe import UniverseFilter
from src.utils.logger import setup_logger


def _load_prices_from_db(engine, config: AppConfig) -> pd.DataFrame:
    """Load the full prices table from DB into a (date, ticker)-indexed DataFrame."""
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
    df = df.set_index(["date", "ticker"]).sort_index()
    return df


def main() -> None:
    config_path = Path("config/config.yaml")
    config = AppConfig.from_yaml(config_path)
    logger = setup_logger("quant_pipeline", level=config.logging.level, log_file=config.logging.file)

    # ------------------------------------------------------------------ #
    # Week 1 — Data Ingestion                                             #
    # ------------------------------------------------------------------ #
    if config.rebuild_on_run:
        logger.warning("Rebuild mode enabled; dropping and recreating schema")
        engine = get_engine()
        rebuild_schema(engine)
    else:
        engine = get_engine()
        create_schema(engine)

    logger.info("Starting data ingestion pipeline")
    pipeline = DataIngestionPipeline(config=config, logger=logger)
    summary = pipeline.run()

    print("\n=== Week 1 — Data Ingestion Summary ===")
    print(f"Tickers:              {summary.ticker_count}")
    print(f"Date range:           {summary.start_date} → {summary.end_date}")
    print(f"Price rows upserted:  {summary.price_rows_upserted}")
    print(f"Macro rows upserted:  {summary.macro_rows_upserted}")
    print(f"Total rows upserted:  {summary.total_rows_upserted}")
    print(f"Status:               {'FAILED_WITH_ERRORS' if summary.failed else 'SUCCESS'}")

    if summary.failed:
        logger.error("Ingestion pipeline failed; skipping Week 2 stages.")
        return

    # ------------------------------------------------------------------ #
    # Week 2 — Return Calculation & Universe Construction                 #
    # ------------------------------------------------------------------ #
    logger.info("Loading prices from DB for Week 2 computations")
    prices = _load_prices_from_db(engine, config)

    if prices.empty:
        logger.error("No prices found in DB — cannot proceed with Week 2 stages.")
        return

    logger.info("Loaded %d price rows for %d tickers", len(prices), prices.index.get_level_values("ticker").nunique())

    # --- 2a: Return Calculation ---
    logger.info("Starting return calculation")
    return_calc = ReturnCalculator(logger=logger)
    return_summary = return_calc.run(engine=engine, prices=prices)

    print("\n=== Week 2a — Return Calculation Summary ===")
    print(f"Daily return rows stored:    {return_summary.daily_rows_stored}")
    print(f"Monthly return rows stored:  {return_summary.monthly_rows_stored}")
    print(f"Validation passed:           {return_summary.validation_passed}")
    print(f"Validation message:          {return_summary.validation_message}")

    # --- 2b: Universe Construction ---
    logger.info("Starting universe membership computation")
    universe_filter = UniverseFilter(logger=logger)
    universe_summary = universe_filter.run(engine=engine, prices=prices, config=config.universe_filter)

    print("\n=== Week 2b — Universe Construction Summary ===")
    print(f"Total (date, ticker) pairs evaluated: {universe_summary.total_ticker_dates}")
    print(f"In-universe pairs:                    {universe_summary.in_universe_ticker_dates}")
    print(f"Inclusion rate:                       {universe_summary.inclusion_rate:.1%}")
    print(f"Universe membership rows stored:      {universe_summary.rows_stored}")

    print("\n✅ Week 1 + Week 2 pipeline complete.")


if __name__ == "__main__":
    main()
