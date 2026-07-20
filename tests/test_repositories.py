import pytest
from sqlalchemy import create_engine
from src.data.db import create_schema
from src.data.repositories import (
    PricesRepository,
    MetadataRepository,
    FundamentalsRepository,
    FactorsRepository,
    RegimesRepository,
    PortfolioRepository,
)
from src.domain.models import TickerMetadata


@pytest.fixture
def memory_engine():
    engine = create_engine("sqlite:///:memory:")
    create_schema(engine)
    return engine


def test_metadata_repository(memory_engine):
    repo = MetadataRepository(engine=memory_engine)
    inserted = repo.save_ticker_metadata([{
        "ticker": "AAPL",
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "shares_outstanding": 15000000000.0,
    }])
    assert inserted >= 1

    df = repo.get_ticker_metadata(["AAPL"])
    assert not df.empty
    assert df.iloc[0]["sector"] == "Technology"

    models = repo.get_domain_metadata(["AAPL"])
    assert len(models) == 1
    assert isinstance(models[0], TickerMetadata)
    assert models[0].ticker == "AAPL"


def test_prices_repository(memory_engine):
    repo = PricesRepository(engine=memory_engine)
    inserted = repo.save_prices([{
        "date": "2024-01-02",
        "ticker": "AAPL",
        "open": 180.0,
        "high": 185.0,
        "low": 179.0,
        "close": 184.0,
        "adj_close": 184.0,
        "volume": 50000000,
    }])
    assert inserted >= 1

    df = repo.get_daily_prices(start_date="2024-01-01", tickers=["AAPL"])
    assert not df.empty


def test_factors_repository(memory_engine):
    repo = FactorsRepository(engine=memory_engine)
    inserted = repo.save_factor_scores([{
        "date": "2024-01-31",
        "ticker": "AAPL",
        "factor_name": "PriceMomentum",
        "raw_score": 0.15,
        "winsorized_score": 0.15,
        "z_score": 1.2,
        "final_score": 1.2,
    }])
    assert inserted >= 1

    scores = repo.get_factor_scores(factor_name="PriceMomentum")
    assert not scores.empty


def test_regimes_repository(memory_engine):
    repo = RegimesRepository(engine=memory_engine)
    inserted = repo.save_market_regimes([{
        "date": "2024-01-31",
        "regime_id": 0,
        "regime_label": "Bull_LowVol",
        "prob_0": 0.85,
        "prob_1": 0.10,
        "prob_2": 0.03,
        "prob_3": 0.02,
    }])
    assert inserted >= 1

    regimes = repo.get_market_regimes()
    assert not regimes.empty
