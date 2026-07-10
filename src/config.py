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


class UniverseFilterConfig(BaseModel):
    """Point-in-time universe filter thresholds."""

    min_market_cap_bn: float = Field(default=1.0, ge=0, description="Minimum market cap in $B (dollar-volume proxy)")
    min_dollar_vol_percentile: float = Field(default=50.0, ge=0, le=100, description="Min daily dollar vol percentile")
    ipo_exclusion_months: int = Field(default=12, ge=0, description="Exclude stocks within N months of first price date")
    dollar_vol_lookback_days: int = Field(default=60, ge=1, description="Lookback window for avg daily dollar volume")


class Week2Config(BaseModel):
    return_frequency: str = Field(default="monthly", pattern="^(daily|monthly)$")
    return_column: str = Field(default="simple_return", pattern="^(simple_return|log_return)$")
    gap_fill_method: str = Field(default="none", pattern="^(none|zero|drop_cols)$")
    max_consecutive_missing: int = Field(default=5, ge=1)
    artifact_dir: str = "data/processed/week2"
    save_artifacts: bool = True


class Week3Config(BaseModel):
    momentum_lookback_months: int = Field(default=12, ge=2)
    short_momentum_lookback_months: int = Field(default=3, ge=2)
    volatility_lookback_days: int = Field(default=252, ge=1)
    winsorize_limits: List[float] = Field(default_factory=lambda: [0.01, 0.01])
    neutralize_size: bool = True
    neutralize_sector: bool = True
    save_artifacts: bool = True
    artifact_dir: str = "data/processed/week3"


class Week5Config(BaseModel):
    rebalance_frequency: str = Field(default="monthly", pattern="^(daily|monthly)$")
    initial_capital: float = Field(default=10000000.0, ge=0.0)
    weighting_method: str = Field(default="equal_weight_long_only", pattern="^(equal_weight_long_only|equal_weight_long_short)$")
    max_position_size: float = Field(default=0.05, ge=0.0, le=1.0)
    commission_bps: float = Field(default=5.0, ge=0.0)
    bid_ask_spread_bps: float = Field(default=10.0, ge=0.0)
    long_only: bool = True
    net_exposure: float = Field(default=1.0, ge=0.0)
    gross_exposure: float = Field(default=1.0, ge=0.0)
    save_artifacts: bool = True
    artifact_dir: str = "data/processed/week5"


class Week6Config(BaseModel):
    """Configuration for Week 6 Factor Combination & ML Integration."""

    combination_methods: List[str] = Field(default_factory=lambda: ["ic_weighted", "fama_macbeth", "xgboost"])
    xgb_n_estimators: int = Field(default=200, ge=10)
    xgb_max_depth: int = Field(default=4, ge=1, le=10)
    xgb_learning_rate: float = Field(default=0.05, ge=0.001, le=1.0)
    xgb_min_train_months: int = Field(default=36, ge=12)
    xgb_purge_gap_months: int = Field(default=1, ge=0)
    ic_lookback_months: int = Field(default=36, ge=6)
    oos_start_date: str = Field(default="2010-01-01")
    save_artifacts: bool = True
    artifact_dir: str = "data/processed/week6"


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
    universe_filter: UniverseFilterConfig = Field(default_factory=UniverseFilterConfig)
    week2: Week2Config = Field(default_factory=Week2Config)
    week3: Week3Config = Field(default_factory=Week3Config)
    week5: Week5Config = Field(default_factory=Week5Config)
    week6: Week6Config = Field(default_factory=Week6Config)
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
