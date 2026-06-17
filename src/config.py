from __future__ import annotations

from pathlib import Path
from typing import List

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class DownloadConfig(BaseModel):
    max_workers: int = 4
    retries: int = 3
    retry_delay_seconds: int = 2
    batch_size: int = 25


class LoggingConfig(BaseModel):
    level: str = "INFO"
    file: str = "logs/pipeline.log"


class AppConfig(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    stock_universe: List[str] = Field(default_factory=list)
    start_date: str
    end_date: str
    fred_series: List[str] = Field(default_factory=list)
    rebuild_on_run: bool = False
    download: DownloadConfig = Field(default_factory=DownloadConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @field_validator("stock_universe", mode="before")
    @classmethod
    def normalize_tickers(cls, value: List[str]) -> List[str]:
        return [ticker.upper() for ticker in value]

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "AppConfig":
        path = Path(config_path)
        with path.open("r", encoding="utf-8") as handle:
            raw_config = yaml.safe_load(handle) or {}
        return cls.model_validate(raw_config)
