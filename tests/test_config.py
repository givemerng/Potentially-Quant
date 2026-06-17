from pathlib import Path

from src.config import AppConfig


def test_config_loads_from_yaml() -> None:
    config = AppConfig.from_yaml(Path("config/config.yaml"))
    assert config.stock_universe
    assert config.start_date == "2000-01-01"
    assert "DGS10" in config.fred_series
