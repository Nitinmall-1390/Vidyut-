"""
===========================================================================
VIDYUT Demand Forecast Routes
===========================================================================
POST /api/v1/demand/forecast        — get forecast for one feeder
POST /api/v1/demand/forecast/batch  — forecast many feeders
GET  /api/v1/demand/feeders          — list known feeders

Author: Vidyut Team
License: MIT
===========================================================================
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, validator

from src.models.inference.batch_predictor import DemandBatchPredictor
from src.models.inference.inference_cache import InferenceCache
from src.models.inference.model_loader import get_model_loader

router = APIRouter()


# ===========================================================================
# Schemas
# ===========================================================================
class WeatherFeatures(BaseModel):
    temperature_c: float = Field(..., description="Avg temperature °C")
    humidity_pct: float = Field(..., ge=0, le=100)
    wind_mps: float = Field(..., ge=0)
    solar_kwhm2: float = Field(..., ge=0)
    precipitation_mm: float = Field(..., ge=0)


class DemandForecastRequest(BaseModel):
    feeder_id: str = Field(..., description="11kV feeder identifier")
    forecast_start: Optional[datetime] = Field(
        None, description="Start of forecast window (UTC). Defaults to now."
    )
    horizon_hours: int = Field(24, ge=1, le=168)
    granularity_minutes: int = Field(15, ge=15, le=60)
    weather: Optional[WeatherFeatures] = None
    is_holiday: bool = False
    use_cache: bool = True

    @validator("granularity_minutes")
    def validate_granularity(cls, v):
        if v not in (15, 30, 60):
            raise ValueError("granularity_minutes must be 15, 30, or 60")
        return v


class DemandForecastPoint(BaseModel):
    timestamp: datetime
    yhat_prophet: float
    yhat_lgbm: float
    yhat_ensemble: float


class DemandForecastResponse(BaseModel):
    feeder_id: str
    forecast_start: datetime
    horizon_hours: int
    granularity_minutes: int
    model_version: str
    points: List[DemandForecastPoint]
    summary: Dict[str, float]


class BatchDemandRequest(BaseModel):
    feeder_ids: List[str] = Field(..., min_items=1, max_items=200)
    forecast_start: Optional[datetime] = None
    horizon_hours: int = Field(24, ge=1, le=168)
    weather: Optional[WeatherFeatures] = None


# ===========================================================================
# Routes
# ===========================================================================
@router.post("/forecast", response_model=DemandForecastResponse)
async def forecast(request: Request, body: DemandForecastRequest) -> DemandForecastResponse:
    start = body.forecast_start or datetime.now(timezone.utc).replace(
        minute=0, second=0, microsecond=0
    )
    future_df = _build_future_frame(
        start, body.horizon_hours, body.granularity_minutes,
        body.weather, body.is_holiday,
    )
    future_df["feeder_id"] = body.feeder_id
    future_df["demand_kw"] = 0.0 # dummy value for feature engineering
    
    # Prophet (Stage 1) does not support timezones in ds column
    if "ds" in future_df.columns:
        future_df["ds"] = pd.to_datetime(future_df["ds"]).dt.tz_localize(None)

    predictor = _get_predictor(request)
    predictor.feeder_ids = [body.feeder_id]
    
    try:
        df = predictor.predict(
            future_df, datetime_col="ds", value_col="demand_kw", feeder_col="feeder_id"
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Forecast unavailable for feeder {body.feeder_id}: {str(e)}")
        
    if df.empty:
        raise HTTPException(
            status_code=503,
            detail=f"Forecast unavailable for feeder {body.feeder_id}",
        )

    # In case the model returns yhat, map to ensemble, prophet, lgbm
    points = []
    for _, row in df.iterrows():
        yhat_ensemble = float(row.get("yhat", 0.0))
        yhat_prophet = float(row.get("prophet_yhat", yhat_ensemble))
        yhat_lgbm = float(row.get("lgbm_yhat", yhat_ensemble))
        points.append(
            DemandForecastPoint(
                timestamp=pd.to_datetime(row["timestamp"]).to_pydatetime(),
                yhat_prophet=yhat_prophet,
                yhat_lgbm=yhat_lgbm,
                yhat_ensemble=yhat_ensemble,
            )
        )

    summ = {
        "min_kw": float(df["yhat"].min()) if "yhat" in df.columns else 0.0,
        "max_kw": float(df["yhat"].max()) if "yhat" in df.columns else 0.0,
        "mean_kw": float(df["yhat"].mean()) if "yhat" in df.columns else 0.0,
        "total_kwh": float(df.get("yhat", pd.Series([0.0])).sum() * (body.granularity_minutes / 60.0)),
    }

    return DemandForecastResponse(
        feeder_id=body.feeder_id,
        forecast_start=start,
        horizon_hours=body.horizon_hours,
        granularity_minutes=body.granularity_minutes,
        model_version=request.app.state.model_loader.default_version,
        points=points,
        summary=summ,
    )


@router.post("/forecast/batch")
async def forecast_batch(request: Request, body: BatchDemandRequest) -> Dict[str, Any]:
    start = body.forecast_start or datetime.now(timezone.utc).replace(
        minute=0, second=0, microsecond=0
    )
    
    frames = []
    for fid in body.feeder_ids:
        df = _build_future_frame(start, body.horizon_hours, 15, body.weather, False)
        df["feeder_id"] = fid
        df["demand_kw"] = 0.0
        frames.append(df)
        
    future_df = pd.concat(frames, ignore_index=True)
    
    predictor = _get_predictor(request)
    predictor.feeder_ids = body.feeder_ids
    
    df = predictor.predict(
        future_df, datetime_col="ds", value_col="demand_kw", feeder_col="feeder_id"
    )
    
    results = {fid: df[df["feeder_id"] == fid] for fid in body.feeder_ids if not df[df["feeder_id"] == fid].empty}
    
    summary = {
        "total_feeders": len(body.feeder_ids),
        "successful_forecasts": len(results),
        "errors": []
    }
    
    return {
        "summary": summary,
        "forecasts": {
            fid: rdf.assign(timestamp=rdf["timestamp"].astype(str)).to_dict(orient="records")
            for fid, rdf in results.items()
        },
    }


@router.get("/feeders")
async def list_feeders(request: Request) -> Dict[str, Any]:
    loader = request.app.state.model_loader
    available = loader.list_available_models()
    feeders = sorted({
        n.replace("prophet_", "")
        for n in available
        if n.startswith("prophet_") and n != "prophet_global"
    })
    return {"count": len(feeders), "feeders": feeders}


# ===========================================================================
# Helpers
# ===========================================================================
def _build_future_frame(
    start: datetime,
    horizon_hours: int,
    granularity_minutes: int,
    weather: Optional[WeatherFeatures],
    is_holiday: bool,
) -> pd.DataFrame:
    n_points = int((horizon_hours * 60) / granularity_minutes)
    timestamps = [start + timedelta(minutes=granularity_minutes * i) for i in range(n_points)]

    df = pd.DataFrame({"ds": timestamps})
    df["hour_of_day"] = [t.hour for t in timestamps]
    df["day_of_week"] = [t.weekday() for t in timestamps]
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_holiday"] = int(bool(is_holiday))
    df["month"] = [t.month for t in timestamps]
    df["quarter"] = [(t.month - 1) // 3 + 1 for t in timestamps]

    if weather is not None:
        df["temperature_c"] = weather.temperature_c
        df["humidity_pct"] = weather.humidity_pct
        df["wind_mps"] = weather.wind_mps
        df["solar_kwhm2"] = weather.solar_kwhm2
        df["precipitation_mm"] = weather.precipitation_mm
    else:
        df["temperature_c"] = 27.0
        df["humidity_pct"] = 65.0
        df["wind_mps"] = 3.0
        df["solar_kwhm2"] = 4.5
        df["precipitation_mm"] = 0.0

    # Lag/rolling placeholders (would be filled from store in real ingestion)
    for lag in (1, 4, 96, 672):
        df[f"y_lag_{lag}"] = 0.0
    df["y_roll_mean_24"] = 0.0
    df["y_roll_std_24"] = 0.0
    return df


def _get_predictor(request: Request) -> DemandBatchPredictor:
    if not hasattr(request.app.state, "predictor"):
        request.app.state.predictor = DemandBatchPredictor()
    return request.app.state.predictor