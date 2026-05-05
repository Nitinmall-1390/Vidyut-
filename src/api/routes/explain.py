"""
===========================================================================
VIDYUT Explainability Routes
===========================================================================
POST /api/v1/explain/theft       — SHAP explanation for theft prediction
POST /api/v1/explain/rules       — rule-engine flags for a consumer
GET  /api/v1/explain/{event_uuid}— audit trail lookup

Author: Vidyut Team
License: MIT
===========================================================================
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

router = APIRouter()


class ExplainRequest(BaseModel):
    consumer_id: str
    features: Dict[str, float]
    top_k: int = Field(10, ge=1, le=50)
    model_name: str = "theft_xgboost"


class ShapFeature(BaseModel):
    feature: str
    value: float
    shap_value: float


class ExplainResponse(BaseModel):
    consumer_id: str
    model_name: str
    model_version: str
    base_value: float
    prediction: float
    top_features: List[ShapFeature]
    feature_hash: str


class RuleCheckRequest(BaseModel):
    consumer_id: str
    daily_consumption: List[float] = Field(
        ..., description="Latest 60 days of daily kWh"
    )
    last_bill_amount: Optional[float] = None
    minimum_charge: float = 50.0
    months_below_min: int = 0


class RuleCheckResponse(BaseModel):
    consumer_id: str
    flags: List[str]
    rule_score: float
    severity: str


@router.post("/theft", response_model=ExplainResponse)
async def explain_theft(request: Request, body: ExplainRequest) -> ExplainResponse:
    try:
        from src.explainability.shap_explainer import explain_prediction
    except ImportError:
        raise HTTPException(501, "SHAP explainer module not available")

    loader = request.app.state.model_loader
    cache = request.app.state.cache

    try:
        model = loader.get(body.model_name, version=loader.default_version)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))

    feat_series = pd.Series(body.features)
    from src.cache.cache_key_builder import CacheKeyBuilder
    builder = CacheKeyBuilder(model_version=loader.default_version)
    feat_hash = builder.fingerprint_features(body.features)
    cache_key = builder.shap_explanation_key(body.consumer_id, body.model_name, feat_hash)

    def _compute() -> Dict[str, Any]:
        try:
            return explain_prediction(model, feat_series, top_k=body.top_k)
        except Exception as e:
            raise HTTPException(500, f"SHAP failed: {e}")

    result = cache.get_or_compute(cache_key, _compute, ttl=86400)

    top_features = [
        ShapFeature(feature=f["feature"], value=float(f["value"]),
                    shap_value=float(f["shap_value"]))
        for f in result.get("top_features", [])
    ]
    return ExplainResponse(
        consumer_id=body.consumer_id,
        model_name=body.model_name,
        model_version=loader.default_version,
        base_value=float(result.get("base_value", 0.0)),
        prediction=float(result.get("prediction", 0.0)),
        top_features=top_features,
        feature_hash=feat_hash,
    )


@router.post("/rules", response_model=RuleCheckResponse)
async def explain_rules(body: RuleCheckRequest) -> RuleCheckResponse:
    try:
        from src.explainability.rule_engine import RuleEngine
    except ImportError:
        raise HTTPException(501, "rule_engine module not available")

    engine = RuleEngine()
    import numpy as np
    
    # We can pass minimum_charge if we temporarily inject it or it's hardcoded in the engine.
    # The engine uses MINIMUM_MONTHLY_CHARGE_KWH = 30.0 hardcoded.
    res = engine.evaluate_consumer(body.consumer_id, np.array(body.daily_consumption))
    
    flags = [f.rule_id for f in res.triggered_flags]
    score = len(flags) / 6.0
    
    return RuleCheckResponse(
        consumer_id=body.consumer_id,
        flags=flags,
        rule_score=round(score, 3),
        severity=res.highest_severity,
    )


@router.get("/audit/{event_uuid}")
async def get_audit_record(event_uuid: str) -> Dict[str, Any]:
    from src.audit.audit_db import get_audit_db
    rec = get_audit_db().get_event(event_uuid)
    if not rec:
        raise HTTPException(404, f"No audit record for {event_uuid}")
    return rec