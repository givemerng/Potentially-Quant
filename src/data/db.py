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
