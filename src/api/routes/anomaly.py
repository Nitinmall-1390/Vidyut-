"""
===========================================================================
VIDYUT Anomaly Detection Routes
===========================================================================
POST /api/v1/anomaly/detect       — single-consumer anomaly check
POST /api/v1/anomaly/detect/batch — batch anomaly check
GET  /api/v1/anomaly/{consumer_id}/score — fetch latest cached score

Author: Vidyut Team
License: MIT
===========================================================================
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from src.api.routes.demand import _get_predictor as _get_predictor_demand

router = APIRouter()


# ===========================================================================
# Schemas
# ===========================================================================
class ConsumerSequence(BaseModel):
    consumer_id: str
    sequence: List[List[float]] = Field(
        ..., description="14-day x feature_count nested list"
    )
    flat_features: Dict[str, float] = Field(default_factory=dict)


class AnomalyDetectRequest(BaseModel):
    consumer: ConsumerSequence
    sequence_end_date: Optional[datetime] = None


class AnomalyDetectResponse(BaseModel):
    consumer_id: str
    sequence_end_date: datetime
    lstm_recon_error: float
    iso_score: float
    lstm_flag: bool
    iso_flag: bool
    intersection_flag: bool
    confidence_score: float
    model_version: str


class BatchAnomalyRequest(BaseModel):
    consumers: List[ConsumerSequence] = Field(..., min_items=1, max_items=500)
    sequence_end_date: Optional[datetime] = None
    lstm_threshold_percentile: float = Field(95.0, ge=50.0, le=99.9)


# ===========================================================================
# Routes
# ===========================================================================
from src.models.inference.batch_predictor import TheftBatchPredictor

def _get_theft_predictor(request: Request) -> TheftBatchPredictor:
    if not hasattr(request.app.state, "theft_predictor"):
        request.app.state.theft_predictor = TheftBatchPredictor()
    return request.app.state.theft_predictor

@router.post("/detect", response_model=AnomalyDetectResponse)
async def detect(request: Request, body: AnomalyDetectRequest) -> AnomalyDetectResponse:
    end_date = body.sequence_end_date or datetime.now(timezone.utc)
    cid = body.consumer.consumer_id
    seq = np.asarray(body.consumer.sequence, dtype=np.float32)
    if seq.ndim != 2:
        raise HTTPException(400, "consumer.sequence must be 2D (timesteps x features)")

    flat = pd.DataFrame([{
        "CONS_NO": cid, **body.consumer.flat_features
    }])

    # Add dummy day_0 ... day_N columns required by TheftBatchPredictor for sequence rebuilding
    for i in range(14):
        flat[f"day_{i}"] = 0.0

    predictor = _get_theft_predictor(request)
    results = predictor.predict(consumers_df=flat, consumer_col="CONS_NO")
    
    stage1 = results.get("stage1_results", pd.DataFrame())
    
    if stage1.empty:
        raise HTTPException(503, f"Anomaly detection failed")
        
    row = stage1.iloc[0].to_dict()
    intersection = bool(row.get("dual_anomaly", False))
    lstm_err = float(row.get("max_recon_error", 0.0))
    iso_score = float(row.get("if_score", 0.0))
    confidence = _confidence_from_anomaly(lstm_err, iso_score)

    return AnomalyDetectResponse(
        consumer_id=cid,
        sequence_end_date=end_date,
        lstm_recon_error=lstm_err,
        iso_score=iso_score,
        lstm_flag=bool(row.get("lstm_anomaly", False)),
        iso_flag=bool(row.get("if_anomaly", False)),
        intersection_flag=intersection,
        confidence_score=confidence,
        model_version=request.app.state.model_loader.default_version,
    )


@router.post("/detect/batch")
async def detect_batch(request: Request, body: BatchAnomalyRequest) -> Dict[str, Any]:
    end_date = body.sequence_end_date or datetime.now(timezone.utc)
    flat_rows = []
    for c in body.consumers:
        flat_rows.append({"CONS_NO": c.consumer_id, **c.flat_features})
    flat_df = pd.DataFrame(flat_rows)
    
    for i in range(14):
        flat_df[f"day_{i}"] = 0.0

    predictor = _get_theft_predictor(request)
    results = predictor.predict(consumers_df=flat_df, consumer_col="CONS_NO")
    
    stage1 = results.get("stage1_results", pd.DataFrame())
    summary = results.get("summary", {})
    
    return {
        "summary": summary,
        "results": stage1.to_dict(orient="records") if not stage1.empty else [],
    }


@router.get("/{consumer_id}/score")
async def get_cached_score(request: Request, consumer_id: str) -> Dict[str, Any]:
    cache = request.app.state.cache
    from src.cache.cache_key_builder import CacheKeyBuilder, CacheNamespace
    builder = CacheKeyBuilder(model_version=request.app.state.model_loader.default_version)
    pattern = builder.entity_pattern(CacheNamespace.ANOMALY_SCORE, consumer_id)
    if cache.is_redis:
        raise HTTPException(501, "Real-time scan not implemented; use POST /detect.")
    raise HTTPException(404, "No cached anomaly score; call POST /detect first.")


def _confidence_from_anomaly(lstm_err: float, iso_score: float) -> float:
    """Map dual scores to 0-100 confidence."""
    # iso_score: lower = more anomalous; lstm_err: higher = more anomalous
    iso_norm = float(np.clip(0.5 - iso_score, 0.0, 1.0))
    lstm_norm = float(np.clip(lstm_err / max(lstm_err + 1e-6, 1.0), 0.0, 1.0))
    combined = (iso_norm + lstm_norm) / 2
    return round(combined * 100.0, 2)