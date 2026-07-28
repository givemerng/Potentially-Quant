"""Unit and Integration Tests for Week 10 FastAPI Backend, Middleware, Cache, and Scheduler."""

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.cache import MemoryCache, RedisCache, get_cache_provider
from src.config import AppConfig
from src.scheduler.tasks import task_compute_factors, task_detect_regimes, task_optimize_portfolio, task_invalidate_cache


@pytest.fixture
def test_client():
    app = create_app()
    return TestClient(app)


def test_health_liveness(test_client):
    response = test_client.get("/health/live")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "request_id" in data
    assert data["data"]["status"] == "ok"
    assert "X-Request-ID" in response.headers
    assert "X-Process-Time" in response.headers


def test_health_readiness(test_client):
    response = test_client.get("/health/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["data"]["status"] == "ready"
    assert data["data"]["database"] == "connected"


def test_health_startup(test_client):
    response = test_client.get("/health/startup")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["data"]["status"] in ("started", "starting")


def test_get_current_regime_endpoint(test_client):
    response = test_client.get("/api/v1/regime/current")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "regime_id" in data["data"]
    assert "regime_label" in data["data"]
    assert "probabilities" in data["data"]


def test_get_latest_factors_endpoint(test_client):
    response = test_client.get("/api/v1/factors/latest")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "scores" in data["data"]


def test_get_current_portfolio_endpoint(test_client):
    response = test_client.get("/api/v1/portfolio/current")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "weights" in data["data"]


def test_get_backtest_metrics_endpoint(test_client):
    response = test_client.get("/api/v1/backtest/metrics")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "strategies" in data["data"]


def test_memory_cache_provider():
    cache = MemoryCache()
    assert cache.set("key1", {"foo": "bar"}, ttl_seconds=60) is True
    assert cache.get("key1") == {"foo": "bar"}
    assert cache.delete("key1") is True
    assert cache.get("key1") is None


def test_redis_cache_fallback():
    # Attempt connecting to invalid Redis URL -> verifies graceful fallback to MemoryCache
    cache = RedisCache(redis_url="redis://invalid_host:6379/0")
    assert cache.set("test_key", "test_val", ttl_seconds=60) is True
    assert cache.get("test_key") == "test_val"
    assert cache.delete("test_key") is True


def test_scheduler_tasks():
    cfg = AppConfig.from_yaml("config/config.yaml")
    t2 = task_compute_factors(cfg)
    t3 = task_detect_regimes(cfg)
    t4 = task_optimize_portfolio(cfg)
    t5 = task_invalidate_cache(cfg)

    assert t2["status"] == "success"
    assert t3["status"] == "success"
    assert t4["status"] == "success"
    assert t5["status"] == "success"
