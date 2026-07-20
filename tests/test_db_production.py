import pytest
from sqlalchemy import text
from src.config import DatabaseSettings, AppConfig
from src.data.db import (
    get_engine,
    close_engine,
    get_db_session,
    get_db,
    check_liveness,
    check_readiness,
    connect_with_retry,
)


def test_database_settings_url_normalization() -> None:
    # Test postgres:// -> postgresql+psycopg://
    cfg1 = DatabaseSettings(database_url="postgres://user:pass@ep-cool.neon.tech/neondb")
    assert cfg1.get_connection_string() == "postgresql+psycopg://user:pass@ep-cool.neon.tech/neondb"

    # Test postgresql:// -> postgresql+psycopg://
    cfg2 = DatabaseSettings(database_url="postgresql://user:pass@ep-cool.neon.tech/neondb")
    assert cfg2.get_connection_string() == "postgresql+psycopg://user:pass@ep-cool.neon.tech/neondb"

    # Test custom driver
    cfg3 = DatabaseSettings(
        postgres_host="remote.db",
        postgres_port=5432,
        postgres_db="mydb",
        postgres_user="myuser",
        postgres_password="mypassword",
        driver_name="psycopg",
        ssl_mode="require",
    )
    assert cfg3.get_connection_string() == "postgresql+psycopg://myuser:mypassword@remote.db:5432/mydb?sslmode=require"


def test_neon_pooled_detection() -> None:
    cfg_pooled = DatabaseSettings(database_url="postgresql+psycopg://user:pass@ep-cool-pooler.us-east-2.aws.neon.tech/neondb")
    assert cfg_pooled.is_neon_pooled() is True

    cfg_direct = DatabaseSettings(database_url="postgresql+psycopg://user:pass@ep-cool.us-east-2.aws.neon.tech/neondb")
    assert cfg_direct.is_neon_pooled() is False


def test_engine_singleton_and_closure() -> None:
    engine1 = get_engine(connection_string="sqlite:///:memory:", force_new=True, set_as_global=True)
    engine2 = get_engine()
    assert engine1 is engine2

    close_engine(engine1)
    engine3 = get_engine(connection_string="sqlite:///:memory:", force_new=True, set_as_global=True)
    assert engine3 is not engine1
    close_engine()


def test_session_lifecycle() -> None:
    engine = get_engine(connection_string="sqlite:///:memory:", force_new=True, set_as_global=True)
    with get_db_session(engine) as session:
        result = session.execute(text("SELECT 42")).scalar()
        assert result == 42

    gen = get_db()
    db_session = next(gen)
    assert db_session is not None
    db_session.close()
    close_engine()


def test_health_probes_and_retry() -> None:
    engine = get_engine(connection_string="sqlite:///:memory:", force_new=True, set_as_global=True)
    
    liveness = check_liveness()
    assert liveness["status"] == "ok"
    assert "timestamp" in liveness

    readiness = check_readiness(engine)
    assert readiness["status"] == "ready"
    assert readiness["database"] == "connected"

    retry_ok = connect_with_retry(engine, max_retries=2, initial_delay=0.1)
    assert retry_ok is True

    close_engine()
