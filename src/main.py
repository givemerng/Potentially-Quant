from __future__ import annotations

from pathlib import Path

from src.config import AppConfig
from src.data.ingestion import DataIngestionPipeline
from src.utils.logger import setup_logger


def main() -> None:
    config_path = Path("config/config.yaml")
    config = AppConfig.from_yaml(config_path)
    logger = setup_logger("quant_pipeline", level=config.logging.level, log_file=config.logging.file)

    logger.info("Starting data ingestion pipeline")
    pipeline = DataIngestionPipeline(config=config, logger=logger)
    summary = pipeline.run()

    print("Data ingestion summary")
    print(f"Tickers: {summary.ticker_count}")
    print(f"Date range: {summary.start_date} -> {summary.end_date}")
    print(f"Price rows upserted: {summary.price_rows_upserted}")
    print(f"Macro rows upserted: {summary.macro_rows_upserted}")
    print(f"Total rows upserted: {summary.total_rows_upserted}")
    print(f"Pipeline status: {'FAILED_WITH_ERRORS' if summary.failed else 'SUCCESS'}")


if __name__ == "__main__":
    main()
