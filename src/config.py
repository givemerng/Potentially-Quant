from __future__ import annotations

import os
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


class Week7Config(BaseModel):
    """Configuration for Week 7 Market Regime Detection with HMM."""

    n_components: int = Field(default=4, ge=2, le=10)
    covariance_type: str = Field(default="full", pattern="^(full|tied|diag|spherical)$")
    n_iter: int = Field(default=100, ge=10)
    random_state: int = Field(default=42)
    bic_selection: bool = True
    max_bic_components: int = Field(default=6, ge=3, le=12)
    save_artifacts: bool = True
    artifact_dir: str = "data/processed/week7"


class Week8Config(BaseModel):
    """Configuration for Week 8 Regime-Conditional Factor Analysis & Adaptive Weight Model."""

    ic_lookback_months: int = Field(default=36, ge=6)
    decay_halflife: float = Field(default=12.0, ge=1.0)
    prior_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    min_regime_samples: int = Field(default=10, ge=3)
    max_factor_weight: float = Field(default=0.25, ge=0.05, le=1.0)
    min_weight_threshold: float = Field(default=0.02, ge=0.0, le=0.1)
    oos_start_date: str = Field(default="2010-01-01")
    oos_end_date: str = Field(default="2024-12-31")
    save_artifacts: bool = True
    artifact_dir: str = "data/processed/week8"


class Week9Config(BaseModel):
    """Configuration for Week 9 Institutional Portfolio Construction, Risk Modeling & Analytics."""

    covariance_model: str = Field(default="ledoit_wolf", pattern="^(ledoit_wolf|oas|sample|factor)$")
    optimizer_type: str = Field(default="cvar", pattern="^(cvar|mean_variance|risk_parity)$")
    alpha_confidence: float = Field(default=0.95, ge=0.5, le=0.999)
    target_cvar: float = Field(default=0.20, ge=0.01, le=1.0)
    max_position_size: float = Field(default=0.05, ge=0.001, le=1.0)
    max_sector_exposure: float = Field(default=0.20, ge=0.01, le=1.0)
    tc_penalty_lambda: float = Field(default=1.0, ge=0.0)
    solver_chain: List[str] = Field(default_factory=lambda: ["CLARABEL", "OSQP", "ECOS"])
    cov_lookback_days: int = Field(default=252, ge=30)
    regularization_eps: float = Field(default=1e-6, ge=0.0)
    save_artifacts: bool = True
    artifact_dir: str = "data/processed/week9"


class Week10Config(BaseModel):
    """Configuration for Week 10 Production FastAPI Backend, Cache, and Scheduler."""

    api_title: str = Field(default="Regime-Adaptive Multi-Factor Alpha Engine API")
    api_version: str = Field(default="1.0.0")
    host: str = Field(default_factory=lambda: os.getenv("HOST", "0.0.0.0"))
    port: int = Field(default_factory=lambda: int(os.getenv("PORT", "8000")))
    cors_origins: List[str] = Field(default_factory=lambda: [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",")])
    redis_url: str = Field(default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    cache_provider: str = Field(default_factory=lambda: os.getenv("CACHE_PROVIDER", "memory"), pattern="^(redis|memory)$")
    cache_ttl_seconds: int = Field(default=86400, ge=60)
    scheduler_cron: str = Field(default="30 16 * * 1-5")
    timezone: str = Field(default="America/New_York")


class DatasetConfig(BaseModel):
    """Configuration for Machine Learning & Dataset Construction."""

    target_freq: str = Field(default="monthly", pattern="^(daily|monthly)$")
    lookahead_months: int = Field(default=1, ge=1)
    normalize_features: bool = Field(default=True)
    winsorize_features: bool = Field(default=True)
    min_train_months: int = Field(default=36, ge=12)
    purge_gap_months: int = Field(default=1, ge=0)


class DatabaseSettings(BaseModel):
    """Configuration for Database Connection & Connection Pooling (Neon / PostgreSQL)."""

    model_config = ConfigDict(str_strip_whitespace=True)

    database_url: str | None = None
    postgres_host: str = Field(default="localhost")
    postgres_port: int = Field(default=5432, ge=1, le=65535)
    postgres_db: str = Field(default="quant_research")
    postgres_user: str = Field(default="postgres")
    postgres_password: str = Field(default="postgres")
    driver_name: str = Field(default="psycopg")
    pool_size: int = Field(default=10, ge=1)
    max_overflow: int = Field(default=20, ge=0)
    pool_recycle: int = Field(default=300, ge=30)
    pool_timeout: int = Field(default=30, ge=1)
    pool_pre_ping: bool = Field(default=True)
    ssl_mode: str = Field(default="require")
    neon_pooled_mode: str = Field(default="auto", pattern="^(auto|pooled|direct)$")
    enable_event_logging: bool = Field(default=False)
    max_startup_retries: int = Field(default=5, ge=1)
    startup_retry_delay: float = Field(default=1.0, ge=0.1)

    @classmethod
    def _resolve_available_driver(cls, driver_name: str) -> str:
        if driver_name == "psycopg":
            try:
                import psycopg  # noqa: F401
                return "psycopg"
            except ImportError:
                try:
                    import psycopg2  # noqa: F401
                    return "psycopg2"
                except ImportError:
                    return "psycopg"
        return driver_name

    @field_validator("database_url", mode="before")
    @classmethod
    def normalize_database_url(cls, v: str | None) -> str | None:
        if not v or not isinstance(v, str):
            return v
        v = v.strip()
        target_driver = cls._resolve_available_driver("psycopg")
        if v.startswith("postgres://"):
            v = f"postgresql+{target_driver}://" + v[len("postgres://"):]
        elif v.startswith("postgresql://") and not v.startswith("postgresql+"):
            v = f"postgresql+{target_driver}://" + v[len("postgresql://"):]
        return v

    def is_neon_pooled(self) -> bool:
        if self.neon_pooled_mode == "pooled":
            return True
        if self.neon_pooled_mode == "direct":
            return False
        url_str = self.database_url or self.postgres_host
        return "-pooler" in url_str or ":6543" in url_str

    def get_connection_string(self) -> str:
        """
        Resolution Hierarchy:
        1. Explicit DATABASE_URL (env or config)
        2. Discrete POSTGRES_* env vars / settings
        3. Local fallback defaults
        """
        if self.database_url:
            return self.database_url

        effective_driver = self._resolve_available_driver(self.driver_name)
        driver = f"postgresql+{effective_driver}" if effective_driver else "postgresql+psycopg"
        base_url = f"{driver}://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        if self.postgres_host not in ("localhost", "127.0.0.1") and self.ssl_mode:
            base_url += f"?sslmode={self.ssl_mode}"
        return base_url

    @classmethod
    def from_env(cls) -> "DatabaseSettings":
        import os
        from dotenv import load_dotenv

        load_dotenv()
        db_url = os.getenv("DATABASE_URL")
        host = os.getenv("POSTGRES_HOST")
        port = os.getenv("POSTGRES_PORT")
        db_name = os.getenv("POSTGRES_DB")
        user = os.getenv("POSTGRES_USER")
        password = os.getenv("POSTGRES_PASSWORD")
        driver = os.getenv("DB_DRIVER")
        pool_size = os.getenv("DB_POOL_SIZE")
        max_overflow = os.getenv("DB_MAX_OVERFLOW")
        pool_recycle = os.getenv("DB_POOL_RECYCLE")
        pool_timeout = os.getenv("DB_POOL_TIMEOUT")
        pool_pre_ping = os.getenv("DB_POOL_PRE_PING")
        ssl_mode = os.getenv("DB_SSL_MODE")
        neon_mode = os.getenv("DB_NEON_POOLED_MODE")
        event_logging = os.getenv("DB_ENABLE_EVENT_LOGGING")

        kwargs = {}
        if db_url is not None:
            kwargs["database_url"] = db_url
        if host is not None:
            kwargs["postgres_host"] = host
        if port is not None:
            kwargs["postgres_port"] = int(port)
        if db_name is not None:
            kwargs["postgres_db"] = db_name
        if user is not None:
            kwargs["postgres_user"] = user
        if password is not None:
            kwargs["postgres_password"] = password
        if driver is not None:
            kwargs["driver_name"] = driver
        if pool_size is not None:
            kwargs["pool_size"] = int(pool_size)
        if max_overflow is not None:
            kwargs["max_overflow"] = int(max_overflow)
        if pool_recycle is not None:
            kwargs["pool_recycle"] = int(pool_recycle)
        if pool_timeout is not None:
            kwargs["pool_timeout"] = int(pool_timeout)
        if pool_pre_ping is not None:
            kwargs["pool_pre_ping"] = pool_pre_ping.lower() in ("true", "1", "yes")
        if ssl_mode is not None:
            kwargs["ssl_mode"] = ssl_mode
        if neon_mode is not None:
            kwargs["neon_pooled_mode"] = neon_mode
        if event_logging is not None:
            kwargs["enable_event_logging"] = event_logging.lower() in ("true", "1", "yes")

        return cls(**kwargs)


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
    db: DatabaseSettings = Field(default_factory=DatabaseSettings.from_env)
    download: DownloadConfig = Field(default_factory=DownloadConfig)
    universe_filter: UniverseFilterConfig = Field(default_factory=UniverseFilterConfig)
    week2: Week2Config = Field(default_factory=Week2Config)
    week3: Week3Config = Field(default_factory=Week3Config)
    week5: Week5Config = Field(default_factory=Week5Config)
    week6: Week6Config = Field(default_factory=Week6Config)
    week7: Week7Config = Field(default_factory=Week7Config)
    week8: Week8Config = Field(default_factory=Week8Config)
    week9: Week9Config = Field(default_factory=Week9Config)
    week10: Week10Config = Field(default_factory=Week10Config)
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

