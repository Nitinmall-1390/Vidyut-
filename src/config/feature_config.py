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


# ── Transformer Capacity & Load-Shedding Thresholds (BESCOM Standard) ────────

TRANSFORMER_CAPACITY_MAP: Dict[str, int] = {
    "25_kVA":   25,
    "63_kVA":   63,
    "100_kVA":  100,   # Most common in Bangalore urban
    "200_kVA":  200,   # Commercial/mixed
    "315_kVA":  315,
    "400_kVA":  400,
    "630_kVA":  630,   # Heavy industrial
}

# Default for hackathon demo: 100 kVA @ PF 0.85 → 85 kW max
DEFAULT_TRANSFORMER_KVA: int = 100
DEFAULT_POWER_FACTOR: float = 0.85

# Alert thresholds (fraction of rated capacity)
LOAD_SHEDDING_RULES: Dict[str, float] = {
    "YELLOW_ALERT":   0.80,   # 80%  → warning, monitor closely
    "ORANGE_ALERT":   0.90,   # 90%  → high risk, prepare shedding
    "RED_ALERT":      0.95,   # 95%  → load shedding imminent (15-30 min)
    "TRIP_THRESHOLD": 1.10,   # 110% → transformer trip (auto-protection)
}

# 8 BESCOM operational zones with real Bangalore centroids
BESCOM_ZONES: Dict[str, Dict] = {
    "Zone_East":    {"lat": 12.96, "lon": 77.65, "color": "#FF4757"},
    "Zone_West":    {"lat": 12.97, "lon": 77.55, "color": "#00D4AA"},
    "Zone_North":   {"lat": 13.03, "lon": 77.58, "color": "#FFB800"},
    "Zone_South":   {"lat": 12.90, "lon": 77.58, "color": "#4A7FA5"},
    "Zone_Central": {"lat": 12.97, "lon": 77.60, "color": "#A855F7"},
    "Zone_NE":      {"lat": 13.05, "lon": 77.65, "color": "#F97316"},
    "Zone_SE":      {"lat": 12.90, "lon": 77.68, "color": "#06B6D4"},
    "Zone_NW":      {"lat": 13.03, "lon": 77.52, "color": "#84CC16"},
}
