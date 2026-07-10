from __future__ import annotations

import os
from typing import Dict

from dotenv import load_dotenv
from sqlalchemy import Boolean, Column, Date, Float, MetaData, String, Table, BigInteger, create_engine
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Engine


metadata = MetaData()

prices_table = Table(
    "prices",
    metadata,
    Column("date", Date, primary_key=True, nullable=False),
    Column("ticker", String(16), primary_key=True, nullable=False),
    Column("open", Float),
    Column("high", Float),
    Column("low", Float),
    Column("close", Float),
    Column("adj_close", Float),
    Column("volume", BigInteger),
)

macro_table = Table(
    "macro",
    metadata,
    Column("date", Date, primary_key=True, nullable=False),
    Column("series_name", String(64), primary_key=True, nullable=False),
    Column("value", Float),
)

returns_table = Table(
    "returns",
    metadata,
    Column("date", Date, primary_key=True, nullable=False),
    Column("ticker", String(16), primary_key=True, nullable=False),
    Column("freq", String(8), primary_key=True, nullable=False),  # 'daily' | 'monthly'
    Column("log_return", Float),
    Column("simple_return", Float),
)

universe_membership_table = Table(
    "universe_membership",
    metadata,
    Column("ticker", String(16), primary_key=True, nullable=False),
    Column("date", Date, primary_key=True, nullable=False),
    Column("in_universe", Boolean, nullable=False),
    Column("market_cap_proxy", Float),   # price * avg_volume (dollar-volume proxy)
    Column("avg_dollar_vol", Float),     # trailing 60-day avg daily dollar volume
    Column("days_since_first_price", Float),  # for IPO exclusion
)

ticker_metadata_table = Table(
    "ticker_metadata",
    metadata,
    Column("ticker", String(16), primary_key=True, nullable=False),
    Column("sector", String(64)),
    Column("industry", String(64)),
    Column("shares_outstanding", Float),
)

fundamentals_table = Table(
    "fundamentals",
    metadata,
    Column("date", Date, primary_key=True, nullable=False),
    Column("ticker", String(16), primary_key=True, nullable=False),
    Column("book_value", Float),
    Column("gross_profit", Float),
    Column("total_assets", Float),
    Column("eps", Float),
)

factors_table = Table(
    "factors",
    metadata,
    Column("date", Date, primary_key=True, nullable=False),
    Column("ticker", String(16), primary_key=True, nullable=False),
    Column("factor_name", String(64), primary_key=True, nullable=False),
    Column("raw_score", Float),
    Column("winsorized_score", Float),
    Column("z_score", Float),
    Column("final_score", Float),
)

factor_metrics_table = Table(
    "factor_metrics",
    metadata,
    Column("date", Date, primary_key=True, nullable=False),
    Column("factor_name", String(64), primary_key=True, nullable=False),
    Column("stage", String(16), primary_key=True, nullable=False),  # 'raw' or 'final'
    Column("metric_name", String(64), primary_key=True, nullable=False),
    Column("value", Float),
)

fama_french_table = Table(
    "fama_french",
    metadata,
    Column("date", Date, primary_key=True, nullable=False),
    Column("mkt_rf", Float),
    Column("smb", Float),
    Column("hml", Float),
    Column("rmw", Float),
    Column("cma", Float),
    Column("rf", Float),
)

portfolio_weights_table = Table(
    "portfolio_weights",
    metadata,
    Column("date", Date, primary_key=True, nullable=False),
    Column("ticker", String(16), primary_key=True, nullable=False),
    Column("backtest_name", String(64), primary_key=True, nullable=False),
    Column("weight", Float),
)

trades_table = Table(
    "trades",
    metadata,
    Column("date", Date, primary_key=True, nullable=False),
    Column("ticker", String(16), primary_key=True, nullable=False),
    Column("backtest_name", String(64), primary_key=True, nullable=False),
    Column("trade_type", String(16)),
    Column("weight_change", Float),
    Column("turnover_contribution", Float),
)

backtest_results_table = Table(
    "backtest_results",
    metadata,
    Column("date", Date, primary_key=True, nullable=False),
    Column("backtest_name", String(64), primary_key=True, nullable=False),
    Column("gross_return", Float),
    Column("net_return", Float),
    Column("portfolio_value", Float),
    Column("drawdown", Float),
    Column("turnover", Float),
    Column("transaction_costs", Float),
)

backtest_metrics_table = Table(
    "backtest_metrics",
    metadata,
    Column("backtest_name", String(64), primary_key=True, nullable=False),
    Column("metric_name", String(64), primary_key=True, nullable=False),
    Column("value", Float),
)

combination_runs_table = Table(
    "combination_runs",
    metadata,
    Column("run_id", String(64), primary_key=True, nullable=False),
    Column("method", String(32), nullable=False),
    Column("config_hash", String(64)),
    Column("train_start", Date),
    Column("train_end", Date),
    Column("test_start", Date),
    Column("test_end", Date),
    Column("created_at", Date, nullable=False),
)

composite_scores_table = Table(
    "composite_scores",
    metadata,
    Column("date", Date, primary_key=True, nullable=False),
    Column("ticker", String(16), primary_key=True, nullable=False),
    Column("method", String(32), primary_key=True, nullable=False),
    Column("run_id", String(64), nullable=False),
    Column("composite_score", Float),
)

combination_weights_table = Table(
    "combination_weights",
    metadata,
    Column("date", Date, primary_key=True, nullable=False),
    Column("factor_name", String(64), primary_key=True, nullable=False),
    Column("method", String(32), primary_key=True, nullable=False),
    Column("run_id", String(64), nullable=False),
    Column("weight", Float),
)

combination_metrics_table = Table(
    "combination_metrics",
    metadata,
    Column("run_id", String(64), primary_key=True, nullable=False),
    Column("method", String(32), primary_key=True, nullable=False),
    Column("metric_name", String(64), primary_key=True, nullable=False),
    Column("metric_scope", String(16), nullable=False),  # 'full' or 'oos'
    Column("value", Float),
)




def build_connection_string() -> str:
    load_dotenv()
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    database = os.getenv("POSTGRES_DB", "quant_research")
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "postgres")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"


def get_engine(connection_string: str | None = None) -> Engine:
    return create_engine(connection_string or build_connection_string(), future=True, pool_pre_ping=True)


def create_schema(engine: Engine) -> None:
    metadata.create_all(engine)


def rebuild_schema(engine: Engine) -> None:
    metadata.drop_all(engine)
    metadata.create_all(engine)


def upsert_rows(engine: Engine, table: Table, rows: list[Dict]) -> int:
    if not rows:
        return 0

    stmt = insert(table).values(rows)
    excluded = stmt.excluded
    update_columns = {
        column.name: getattr(excluded, column.name)
        for column in table.columns
        if not column.primary_key
    }
    upsert_stmt = stmt.on_conflict_do_update(
        index_elements=[column.name for column in table.primary_key.columns],
        set_=update_columns,
    )

    with engine.begin() as connection:
        result = connection.execute(upsert_stmt)
    return int(result.rowcount or 0)
