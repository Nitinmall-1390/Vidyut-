"""
VIDYUT Feature Configuration
===========================================================================
Central registry for all feature names, lag windows, rolling windows,
categorical encodings, and schema definitions used across the pipeline.
===========================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ── Demand Forecasting Features ───────────────────────────────────────────

DEMAND_LAG_HOURS: List[int] = [1, 2, 3, 6, 12, 24, 48, 168]  # hours
DEMAND_LAG_PERIODS: List[int] = [h * 4 for h in DEMAND_LAG_HOURS]  # 15-min periods

DEMAND_ROLLING_WINDOWS: List[int] = [4, 8, 16, 32, 96, 192, 672]  # periods
# 4=1h, 8=2h, 16=4h, 32=8h, 96=24h, 192=48h, 672=7d

DEMAND_CALENDAR_FEATURES: List[str] = [
    "hour",
    "dayofweek",
    "dayofmonth",
    "dayofyear",
    "weekofyear",
    "month",
    "quarter",
    "is_weekend",
    "is_holiday",
    "is_peak_hour",  # 07:00–10:00 and 18:00–22:00 IST
    "season",        # 0=winter, 1=spring, 2=summer, 3=monsoon
]

DEMAND_WEATHER_FEATURES: List[str] = [
    "T2M",            # temperature at 2m (°C)
    "T2M_MAX",        # daily max temperature
    "T2M_MIN",        # daily min temperature
    "RH2M",           # relative humidity at 2m (%)
    "WS2M",           # wind speed at 2m (m/s)
    "PRECTOTCORR",    # precipitation (mm/day)
    "ALLSKY_SFC_SW_DWN",  # solar irradiance (W/m²)
]

DEMAND_TARGET_COL: str = "demand_kw"
DEMAND_DATETIME_COL: str = "timestamp"
DEMAND_ZONE_COL: str = "feeder_id"


# ── Theft Detection Features ───────────────────────────────────────────────

THEFT_CONSUMPTION_COLS_PREFIX: str = "day_"  # SGCC: day_0 … day_1033
THEFT_LABEL_COL: str = "FLAG"                # 0=normal, 1=theft
THEFT_CONSUMER_ID_COL: str = "CONS_NO"

THEFT_AGGREGATE_FEATURES: List[str] = [
    "mean_consumption",
    "std_consumption",
    "cv_consumption",           # coefficient of variation
    "min_consumption",
    "max_consumption",
    "median_consumption",
    "q25_consumption",
    "q75_consumption",
    "iqr_consumption",
    "zero_ratio",               # fraction of zero readings
    "negative_ratio",           # fraction of negative readings
    "trend_slope",              # linear trend coefficient
    "entropy",                  # Shannon entropy of normalised consumption
    "consecutive_zeros_max",    # longest consecutive zero-reading streak
    "mom_drop_max",             # max month-over-month % drop
    "night_day_ratio",          # night vs day consumption ratio
    "weekend_weekday_ratio",
    "spike_count",              # readings > mean + 3σ
]

THEFT_LSTM_WINDOW_DAYS: int = 14   # 14-day sequences for LSTM AE
THEFT_LSTM_FEATURES: List[str] = ["daily_consumption_normalised"]

# XGBoost 3-class target
THEFT_CLASS_LABELS: Dict[int, str] = {
    0: "normal",
    1: "theft",
    2: "technical_loss",
    3: "billing_error",
}

# Rule-engine thresholds
RULE_ZERO_READING_DAYS_THRESHOLD: int = 5
RULE_MOM_DROP_PCT_THRESHOLD: float = 60.0   # 60% month-on-month drop
RULE_MIN_CHARGE_MONTHS_THRESHOLD: int = 3   # bill < minimum charge for N months
CONFIDENCE_HIGH_THRESHOLD: int = 80
CONFIDENCE_MEDIUM_THRESHOLD: int = 50


# ── Network / Ring Detection ───────────────────────────────────────────────

RING_GEOHASH_PRECISION: int = 6   # ~0.6 km × 1.2 km cells
RING_MIN_COMMUNITY_SIZE: int = 3  # ignore communities with < 3 members
RING_ANOMALY_FRACTION_THRESHOLD: float = 0.60

# ── Schema Definitions ────────────────────────────────────────────────────

@dataclass
class DemandSchema:
    """Column schema expected by the demand pipeline."""
    timestamp: str = "timestamp"
    feeder_id: str = "feeder_id"
    demand_kw: str = "demand_kw"
    zone: str = "zone"
    required_columns: List[str] = field(default_factory=lambda: [
        "timestamp", "feeder_id", "demand_kw"
    ])


@dataclass
class TheftSchema:
    """Column schema expected by the theft detection pipeline."""
    consumer_id: str = "CONS_NO"
    label: str = "FLAG"
    lat: str = "lat"
    lon: str = "lon"
    transformer_id: str = "transformer_id"
    required_columns: List[str] = field(default_factory=lambda: [
        "CONS_NO", "FLAG"
    ])


DEMAND_SCHEMA = DemandSchema()
THEFT_SCHEMA = TheftSchema()

# ── Seasonal Map (Bangalore) ──────────────────────────────────────────────

# Month → season index (0=winter, 1=spring, 2=summer, 3=monsoon)
BANGALORE_SEASONS: Dict[int, int] = {
    1: 0,   # January  → winter
    2: 0,   # February → winter
    3: 1,   # March    → spring
    4: 2,   # April    → summer
    5: 2,   # May      → summer
    6: 3,   # June     → monsoon
    7: 3,   # July     → monsoon
    8: 3,   # August   → monsoon
    9: 3,   # September → monsoon
    10: 1,  # October  → spring
    11: 0,  # November → winter
    12: 0,  # December → winter
}

# Peak hour ranges (IST, 24h)
PEAK_HOUR_RANGES: List[Tuple[int, int]] = [(7, 10), (18, 22)]
