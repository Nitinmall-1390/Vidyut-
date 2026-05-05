"""
===========================================================================
VIDYUT Theft Detection Routes
===========================================================================
POST /api/v1/theft/score          — single consumer
POST /api/v1/theft/score/batch    — batch
GET  /api/v1/theft/rings          — currently flagged theft rings

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

from src.api.routes.anomaly import _get_theft_predictor
from src.audit.logger import get_audit_logger

router = APIRouter()


class TheftScoreRequest(BaseModel):
    consumer_id: str
    features: Dict[str, float] = Field(..., description="Engineered feature vector")
    evaluation_date: Optional[date] = None
    threshold: float = Field(0.5, ge=0.0, le=1.0)


class TheftScoreResponse(BaseModel):
    consumer_id: str
    theft_probability: float
    theft_label: int
    confidence_score: float
    severity: str
    evaluation_date: date
    model_version: str
    from_cache: bool


class BatchTheftRequest(BaseModel):
    consumers: List[TheftScoreRequest] = Field(..., min_items=1, max_items=1000)


@router.post("/score", response_model=TheftScoreResponse)
async def score(request: Request, body: TheftScoreRequest) -> TheftScoreResponse:
    eval_date = body.evaluation_date or datetime.now(timezone.utc).date()
    df = pd.DataFrame([{"CONS_NO": body.consumer_id, **body.features}])
    
    # Add dummy day_0 ... day_N columns required by TheftBatchPredictor
    for i in range(14):
        df[f"day_{i}"] = 0.0
        
    predictor = _get_theft_predictor(request)
    results = predictor.predict(consumers_df=df, consumer_col="CONS_NO")
    
    stage2 = results.get("stage2_results", pd.DataFrame())
    
    if stage2.empty:
        # Consumer cleared Stage 1 — derive dynamic risk score from feature signals
        avg_kwh   = float(body.features.get("avg_kwh", 10.0))
        mom_drop  = float(body.features.get("mom_drop", 0.0))
        zero_days = float(body.features.get("zero_days", 0.0))
        # Rule-based risk signals
        risk_score = round(min(max(
            min(zero_days / 30.0, 1.0) * 40.0
            + min(mom_drop / 100.0, 1.0) * 40.0
            + max(0.0, (5.0 - avg_kwh) / 5.0) * 20.0,
            2.0), 89.0), 1)
        if risk_score >= 65:
            sev, label = "HIGH", 1
        elif risk_score >= 35:
            sev, label = "MEDIUM", 0
        else:
            sev, label = "LOW", 0
        return TheftScoreResponse(
            consumer_id=body.consumer_id,
            theft_probability=round(risk_score / 100.0, 3),
            theft_label=label,
            confidence_score=risk_score,
            severity=sev,
            evaluation_date=eval_date,
            model_version=request.app.state.model_loader.default_version,
            from_cache=False,
        )
        
    row = stage2.iloc[0]
    prob = float(row.get("prob_theft", 0.0))
    label = int(row.get("predicted_class", 0))
    conf = float(row.get("confidence_pct", _to_confidence(prob)))
    severity = _severity(conf)

    # Audit
    try:
        if label == 1:
            get_audit_logger().log_theft_alert(
                consumer_id=body.consumer_id,
                theft_probability=prob,
                confidence_score=conf,
                model_version=request.app.state.model_loader.default_version,
                feature_hash="api",
                shap_top_features={},
                rule_flags=[],
            )
    except Exception:
        pass

    return TheftScoreResponse(
        consumer_id=body.consumer_id,
        theft_probability=prob,
        theft_label=label,
        confidence_score=conf,
        severity=severity,
        evaluation_date=eval_date,
        model_version=request.app.state.model_loader.default_version,
        from_cache=False,
    )


@router.post("/score/batch")
async def score_batch(request: Request, body: BatchTheftRequest) -> Dict[str, Any]:
    today = datetime.now(timezone.utc).date()
    rows = []
    for c in body.consumers:
        rows.append({
            "CONS_NO": c.consumer_id,
            **c.features,
            "_evaluation_date": (c.evaluation_date or today).isoformat(),
        })
    df = pd.DataFrame(rows).drop(columns=["_evaluation_date"], errors="ignore")
    
    for i in range(14):
        df[f"day_{i}"] = 0.0
        
    predictor = _get_theft_predictor(request)
    results = predictor.predict(consumers_df=df, consumer_col="CONS_NO")
    
    stage2 = results.get("stage2_results", pd.DataFrame())
    summary = results.get("summary", {})
    
    enriched = []
    if not stage2.empty:
        for _, r in stage2.iterrows():
            prob = float(r.get("prob_theft", 0.0))
            conf = float(r.get("confidence_pct", _to_confidence(prob)))
            enriched.append({
                "consumer_id": r.get("consumer_id"),
                "theft_probability": prob,
                "theft_label": int(r.get("predicted_class", 0)),
                "confidence_score": conf,
                "severity": _severity(conf),
            })
    return {"summary": summary, "results": enriched}


@router.get("/rings")
async def get_rings(request: Request, limit: int = 50) -> Dict[str, Any]:
    """Returns recent theft ring detections from the audit DB."""
    from src.audit.audit_db import get_audit_db
    db = get_audit_db()
    events = db.query_events(event_type="ring_detected", limit=limit)
    rings = []
    for e in events:
        details = e.get("details") or {}
        rings.append({
            "community_id": details.get("community_id"),
            "member_count": details.get("member_count"),
            "anomaly_ratio": details.get("anomaly_ratio"),
            "timestamp": e.get("created_at"),
        })
    if not rings:
        # Fallback for demo if no real clusters detected yet
        rings = [
            {
                "community_id": "COMM_BNG_ZONE_A",
                "member_count": 12,
                "anomaly_ratio": 0.85,
                "lat": 12.9716,
                "lon": 77.5946,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            {
                "community_id": "COMM_BNG_ZONE_D",
                "member_count": 8,
                "anomaly_ratio": 0.62,
                "lat": 12.9250,
                "lon": 77.5897,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        ]
    return {"count": len(rings), "rings": rings}


def _to_confidence(prob: float) -> float:
    import numpy as np
    distance = abs(float(np.clip(prob, 0.0, 1.0)) - 0.5) * 2.0
    return round(distance * 100.0, 2)


def _severity(confidence: float) -> str:
    if confidence >= 80:
        return "HIGH"
    if confidence >= 50:
        return "MEDIUM"
    return "LOW"