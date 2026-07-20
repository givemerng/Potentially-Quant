from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager, contextmanager
from datetime import date as datetime_date, datetime, timezone
from typing import Any, Dict, Generator

from dotenv import load_dotenv
from sqlalchemy import Boolean, Column, Date, Float, Integer, MetaData, String, Table, BigInteger, create_engine, event, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from src.config import DatabaseSettings


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
    Column("ebitda", Float),
    Column("total_debt", Float),
    Column("operating_cash_flow", Float),
    Column("capital_expenditures", Float),
    Column("net_income", Float),
)

insider_transactions_table = Table(
    "insider_transactions",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("ticker", String(16), nullable=False),
    Column("date", Date, nullable=False),
    Column("insider", String(128)),
    Column("position", String(128)),
    Column("shares", Float),
    Column("value", Float),
    Column("text", String(512)),
)

earnings_calendar_table = Table(
    "earnings_calendar",
    metadata,
    Column("ticker", String(16), primary_key=True, nullable=False),
    Column("date", Date, primary_key=True, nullable=False),
    Column("eps_estimate", Float),
    Column("reported_eps", Float),
    Column("surprise_pct", Float),
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

market_regimes_table = Table(
    "market_regimes",
    metadata,
    Column("date", Date, primary_key=True, nullable=False),
    Column("regime_id", Integer, nullable=False),
    Column("regime_label", String(32)),
    Column("prob_0", Float),
    Column("prob_1", Float),
    Column("prob_2", Float),
    Column("prob_3", Float),
)

regime_factor_ic_table = Table(
    "regime_factor_ic",
    metadata,
    Column("date", Date, primary_key=True, nullable=False),
    Column("factor_name", String(64), primary_key=True, nullable=False),
    Column("regime_id", Integer, primary_key=True, nullable=False),
    Column("regime_label", String(32)),
    Column("sample_ic", Float),
    Column("rolling_icir", Float),
    Column("posterior_ic", Float),
    Column("posterior_variance", Float),
    Column("effective_sample_size", Float),
)

regime_factor_weights_table = Table(
    "regime_factor_weights",
    metadata,
    Column("date", Date, primary_key=True, nullable=False),
    Column("factor_name", String(64), primary_key=True, nullable=False),
    Column("run_id", String(64), primary_key=True, nullable=False),
    Column("regime_probability", Float),
    Column("dynamic_weight", Float),
    Column("posterior_ic", Float),
)

adaptive_runs_table = Table(
    "adaptive_runs",
    metadata,
    Column("run_id", String(64), primary_key=True, nullable=False),
    Column("hmm_model_version", String(64)),
    Column("ic_lookback_months", Integer),
    Column("bayesian_prior_weight", Float),
    Column("decay_halflife", Float),
    Column("oos_start", Date),
    Column("oos_end", Date),
    Column("performance_metrics", String(2048)),
    Column("created_at", Date, nullable=False),
)





_ENGINE: Engine | None = None
logger = logging.getLogger(__name__)


def build_connection_string(db_config: DatabaseSettings | None = None) -> str:
    cfg = db_config or DatabaseSettings.from_env()
    return cfg.get_connection_string()


def _attach_event_listeners(engine: Engine) -> None:
    @event.listens_for(engine, "connect")
    def receive_connect(dbapi_connection, connection_record):
        logger.debug("DB Event: Connection established")

    @event.listens_for(engine, "checkout")
    def receive_checkout(dbapi_connection, connection_record, connection_proxy):
        logger.debug("DB Event: Connection checked out from pool")

    @event.listens_for(engine, "checkin")
    def receive_checkin(dbapi_connection, connection_record):
        logger.debug("DB Event: Connection returned to pool")

    @event.listens_for(engine, "disconnect")
    def receive_disconnect(dbapi_connection, connection_record):
        logger.debug("DB Event: Connection disconnected")


def get_engine(
    connection_string: str | None = None,
    db_config: DatabaseSettings | None = None,
    force_new: bool = False,
    set_as_global: bool = True,
) -> Engine:
    global _ENGINE

    if not force_new and connection_string is None and db_config is None and _ENGINE is not None:
        return _ENGINE

    cfg = db_config or DatabaseSettings.from_env()
    conn_str = connection_string or cfg.get_connection_string()

    engine_kwargs: dict[str, Any] = {
        "future": True,
        "pool_pre_ping": cfg.pool_pre_ping,
    }

    if conn_str.startswith("sqlite"):
        pass
    else:
        engine_kwargs["pool_size"] = cfg.pool_size
        engine_kwargs["max_overflow"] = cfg.max_overflow
        engine_kwargs["pool_recycle"] = cfg.pool_recycle
        engine_kwargs["pool_timeout"] = cfg.pool_timeout

        connect_args: dict[str, Any] = {}
        if cfg.ssl_mode and "localhost" not in conn_str and "127.0.0.1" not in conn_str and "sslmode" not in conn_str:
            connect_args["sslmode"] = cfg.ssl_mode

        if cfg.is_neon_pooled():
            logger.info("Neon Pooled endpoint detected (PgBouncer mode). Adjusting connection parameters.")
            if "psycopg" in conn_str:
                connect_args["prepare_threshold"] = None

        if connect_args:
            engine_kwargs["connect_args"] = connect_args

    engine = create_engine(conn_str, **engine_kwargs)

    if cfg.enable_event_logging:
        _attach_event_listeners(engine)

    if set_as_global:
        _ENGINE = engine

    return engine


def close_engine(engine: Engine | None = None) -> None:
    global _ENGINE
    target_engine = engine or _ENGINE
    if target_engine is not None:
        logger.info("Disposing database engine connection pool.")
        target_engine.dispose()
        if target_engine == _ENGINE:
            _ENGINE = None


def get_sessionmaker(engine: Engine | None = None) -> sessionmaker:
    eng = engine or get_engine()
    return sessionmaker(autocommit=False, autoflush=False, bind=eng)


@contextmanager
def get_db_session(engine: Engine | None = None) -> Generator[Session, None, None]:
    """Context manager for transaction-scoped session lifecycle."""
    session_factory = get_sessionmaker(engine)
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency generator for request-scoped database sessions."""
    session_factory = get_sessionmaker()
    session = session_factory()
    try:
        yield session
    finally:
        session.close()


def check_liveness() -> dict[str, Any]:
    """Liveness probe: verifies application process health."""
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": "potentially-quant-db",
    }


def check_readiness(engine: Engine | None = None) -> dict[str, Any]:
    """Readiness probe: executes 'SELECT 1' against the database."""
    eng = engine or get_engine()
    start_time = time.perf_counter()
    try:
        with eng.connect() as conn:
            result = conn.execute(text("SELECT 1")).scalar()
            latency_ms = (time.perf_counter() - start_time) * 1000
            if result == 1:
                return {
                    "status": "ready",
                    "database": "connected",
                    "latency_ms": round(latency_ms, 2),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            raise RuntimeError(f"Unexpected healthcheck result: {result}")
    except Exception as exc:
        logger.error("Readiness check failed: %s", exc)
        return {
            "status": "unready",
            "database": "disconnected",
            "error": str(exc),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


def connect_with_retry(
    engine: Engine | None = None,
    max_retries: int = 5,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
) -> bool:
    """Startup retry helper with exponential backoff for Render / Docker readiness."""
    eng = engine or get_engine()
    delay = initial_delay
    for attempt in range(1, max_retries + 1):
        readiness = check_readiness(eng)
        if readiness.get("status") == "ready":
            logger.info("Database connection established on attempt %d.", attempt)
            return True
        logger.warning(
            "Database connection attempt %d/%d failed. Retrying in %.1fs...",
            attempt,
            max_retries,
            delay,
        )
        time.sleep(delay)
        delay *= backoff_factor
    logger.error("Failed to connect to database after %d attempts.", max_retries)
    return False


@asynccontextmanager
async def db_lifespan(app: Any):
    """FastAPI application lifespan context manager for engine startup/shutdown."""
    engine = get_engine()
    connect_with_retry(engine)
    try:
        yield
    finally:
        close_engine(engine)


def create_schema(engine: Engine) -> None:
    metadata.create_all(engine)


def rebuild_schema(engine: Engine) -> None:
    metadata.drop_all(engine)
    metadata.create_all(engine)


def _convert_row_dates(row: dict[str, Any]) -> dict[str, Any]:
    new_row = dict(row)
    for k, v in new_row.items():
        if isinstance(v, str) and len(v) == 10 and v[4] == "-" and v[7] == "-":
            try:
                new_row[k] = datetime_date.fromisoformat(v)
            except ValueError:
                pass
    return new_row


def upsert_rows(engine: Engine, table: Table, rows: list[Dict], batch_size: int = 1000) -> int:
    if not rows:
        return 0

    total_upserted = 0
    # Process in batches to stay within PostgreSQL parameter limits (max 65,535 parameters per statement)
    for i in range(0, len(rows), batch_size):
        batch = [_convert_row_dates(r) for r in rows[i : i + batch_size]]
        stmt = insert(table).values(batch)
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
            total_upserted += int(result.rowcount or 0)

    return total_upserted


