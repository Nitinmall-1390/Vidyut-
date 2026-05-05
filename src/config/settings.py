"""
VIDYUT Settings
===========================================================================
Centralised configuration loaded from environment variables via
pydantic-settings. Single source of truth for all tuneable parameters.
===========================================================================
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    All configuration drawn from .env / environment variables.
    Pydantic validates types and provides defaults where safe.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ────────────────────────────────────────────────────────
    app_env: str = "development"
    app_version: str = "0.1.0"
    secret_key: str = "change-me-in-production"
    log_level: str = "INFO"

    # ── Data Paths ────────────────────────────────────────────────────────
    data_raw_dir: Path = Path("data/raw")
    data_processed_dir: Path = Path("data/processed")
    data_synthetic_dir: Path = Path("data/synthetic")
    data_models_dir: Path = Path("data/models")
    data_latest_model: Path = Path("data/models/LATEST")

    # ── NASA POWER ────────────────────────────────────────────────────────
    nasa_power_url: str = "https://power.larc.nasa.gov/api/temporal/daily/point"
    nasa_lat: float = 12.97
    nasa_lon: float = 77.59
    nasa_params: str = "T2M,T2M_MAX,T2M_MIN,RH2M,WS2M,PRECTOTCORR,ALLSKY_SFC_SW_DWN"

    # ── Ensemble Weights ──────────────────────────────────────────────────
    prophet_weight: float = 0.4
    lgbm_weight: float = 0.6

    # ── LSTM Autoencoder ──────────────────────────────────────────────────
    lstm_seq_len: int = 96          # 96 × 15-min = 24 hours
    lstm_latent_dim: int = 64
    lstm_recon_percentile: int = 95

    # ── Isolation Forest ──────────────────────────────────────────────────
    iso_forest_contamination: float = 0.05
    iso_forest_n_estimators: int = 100

    # ── Theft Ring Detection ──────────────────────────────────────────────
    theft_ring_anomaly_threshold: float = 0.60
    geohash_precision: int = 6

    # ── Redis ─────────────────────────────────────────────────────────────
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str = ""
    redis_ttl_seconds: int = 3600

    # ── Database ─────────────────────────────────────────────────────────
    audit_db_path: Path = Path("data/audit.db")
    database_url: str = ""


    # ── API ───────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    api_rate_limit_per_minute: int = 60
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60

    # ── Dashboard ─────────────────────────────────────────────────────────
    dashboard_host: str = "0.0.0.0"
    dashboard_port: int = 8501

    # ── Derived helpers ───────────────────────────────────────────────────
    @property
    def nasa_params_list(self) -> List[str]:
        return [p.strip() for p in self.nasa_params.split(",")]

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() == "production"

    @field_validator("prophet_weight", "lgbm_weight", mode="before")
    @classmethod
    def _validate_weight(cls, v: float) -> float:
        assert 0.0 <= float(v) <= 1.0, "Weight must be in [0, 1]"
        return float(v)

    def ensure_directories(self) -> None:
        """Create all required data directories if they don't exist."""
        dirs = [
            self.data_raw_dir,
            self.data_processed_dir / "features",
            self.data_processed_dir / "datasets",
            self.data_synthetic_dir,
            self.data_models_dir / "v1",
            self.data_models_dir / "v2",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
        self.audit_db_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    settings = Settings()
    settings.ensure_directories()
    return settings
