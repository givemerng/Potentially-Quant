import pytest
from sqlalchemy import create_engine
from src.data.db import create_schema
from src.data.repositories import (
    PricesRepository,
    MetadataRepository,
    FactorsRepository,
)
from src.services import MarketDataService, FactorService
from src.data.dataset_builders import MLDatasetBuilder, HMMDatasetBuilder


@pytest.fixture
def memory_engine():
    engine = create_engine("sqlite:///:memory:")
    create_schema(engine)
    return engine


def test_services_and_dataset_builders(memory_engine):
    prices_repo = PricesRepository(engine=memory_engine)
    factors_repo = FactorsRepository(engine=memory_engine)
    metadata_repo = MetadataRepository(engine=memory_engine)

    # Insert mock data
    prices_repo.save_prices([{
        "date": "2024-01-02",
        "ticker": "AAPL",
        "open": 180.0,
        "high": 185.0,
        "low": 179.0,
        "close": 184.0,
        "adj_close": 184.0,
        "volume": 50000000,
    }])
    prices_repo.save_returns([{
        "date": "2024-01-31",
        "ticker": "AAPL",
        "freq": "monthly",
        "log_return": 0.05,
        "simple_return": 0.051,
    }])
    factors_repo.save_factor_scores([{
        "date": "2024-01-31",
        "ticker": "AAPL",
        "factor_name": "PriceMomentum",
        "raw_score": 0.15,
        "winsorized_score": 0.15,
        "z_score": 1.2,
        "final_score": 1.2,
    }])

    market_service = MarketDataService(prices_repo=prices_repo, metadata_repo=metadata_repo)
    factor_service = FactorService(factors_repo=factors_repo)

    ml_builder = MLDatasetBuilder(factor_service=factor_service, market_data_service=market_service)
    X, y = ml_builder.build_ml_dataset()
    assert not X.empty
    assert not y.empty

    hmm_builder = HMMDatasetBuilder(factor_service=factor_service, market_data_service=market_service)
    features = hmm_builder.build_hmm_features()
    assert features.empty or isinstance(features, type(X))
