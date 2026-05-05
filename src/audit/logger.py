"""
VIDYUT Audit Logger
===========================================================================
Immutable structured audit trail for every prediction, alert, and
user action. Stored in SQLite via SQLAlchemy for portability.
All writes are append-only — no update/delete operations permitted.
===========================================================================
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from src.audit.audit_db import AuditEvent, get_audit_engine
from src.utils.db import get_session
from src.utils.logger import get_logger

log = get_logger("vidyut.audit_logger")


class AuditLogger:
    """
    Append-only audit logger backed by SQLite.

    Every call to log_prediction(), log_alert(), or log_action()
    writes one immutable row to the audit_events table.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        self.engine = get_audit_engine(db_path)

    def log_prediction(
        self,
        prediction_type: str,    # "demand" | "theft" | "ring"
        entity_id: str,          # feeder_id or consumer_id
        model_version: str,
        input_hash: str,         # SHA-256 of input features (privacy-safe)
        output_summary: Dict[str, Any],
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> int:
        """Log a model prediction event. Returns the audit event ID."""
        return self._write_event(
            event_type="PREDICTION",
            entity_id=entity_id,
            details={
                "prediction_type": prediction_type,
                "model_version": model_version,
                "input_hash": input_hash,
                "output_summary": output_summary,
            },
            user_id=user_id,
            session_id=session_id,
        )

    def log_alert(
        self,
        alert_type: str,         # "theft" | "ring" | "demand_spike"
        entity_id: str,
        severity: str,           # "HIGH" | "MEDIUM" | "LOW"
        confidence_score: int,
        rule_flags: list,
        model_version: str,
        user_id: Optional[str] = None,
    ) -> int:
        """Log a generated alert event."""
        return self._write_event(
            event_type="ALERT",
            entity_id=entity_id,
            details={
                "alert_type": alert_type,
                "severity": severity,
                "confidence_score": confidence_score,
                "rule_flags": rule_flags,
                "model_version": model_version,
            },
            user_id=user_id,
        )

    def log_action(
        self,
        action: str,             # e.g. "MODEL_TRAINED", "VERSION_PROMOTED"
        entity_id: str,
        details: Optional[Dict] = None,
        user_id: Optional[str] = None,
    ) -> int:
        """Log a system action (training, promotion, etc.)."""
        return self._write_event(
            event_type="ACTION",
            entity_id=entity_id,
            details=details or {},
            user_id=user_id,
        )

    def _write_event(
        self,
        event_type: str,
        entity_id: str,
        details: Dict,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> int:
        """Internal: write one event row, return its ID."""
        event = AuditEvent(
            event_type=event_type,
            entity_id=entity_id,
            details_json=json.dumps(details, default=str),
            user_id=user_id or "system",
            session_id=session_id or "",
            created_at=datetime.now(timezone.utc),
        )
        with get_session(self.engine) as session:
            session.add(event)
            session.flush()
            event_id = event.id

        log.debug(
            "Audit event #%d: type=%s entity=%s",
            event_id, event_type, entity_id,
        )
        return event_id

    def query_events(
        self,
        entity_id: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 100,
    ) -> list:
        """Query audit events with optional filters."""
        from sqlalchemy import select
        with get_session(self.engine) as session:
            stmt = select(AuditEvent)
            if entity_id:
                stmt = stmt.where(AuditEvent.entity_id == entity_id)
            if event_type:
                stmt = stmt.where(AuditEvent.event_type == event_type)
            stmt = stmt.order_by(AuditEvent.created_at.desc()).limit(limit)
            rows = session.execute(stmt).scalars().all()
            return [
                {
                    "id": r.id,
                    "event_type": r.event_type,
                    "entity_id": r.entity_id,
                    "user_id": r.user_id,
                    "created_at": str(r.created_at),
                    "details": json.loads(r.details_json),
                }
                for r in rows
            ]

    def log_api_call(self, endpoint: str, method: str, status_code: int, latency_ms: float, actor: str, request_id: str) -> int:
        return self._write_event(
            event_type="API_CALL",
            entity_id=request_id,
            details={"endpoint": endpoint, "method": method, "status": status_code, "latency": latency_ms},
            user_id=actor,
        )

    def log_theft_alert(self, consumer_id: str, theft_probability: float, confidence_score: float, model_version: str, feature_hash: str, shap_top_features: dict, rule_flags: list) -> int:
        return self.log_alert(
            alert_type="theft",
            entity_id=consumer_id,
            severity="HIGH" if confidence_score >= 80 else "MEDIUM" if confidence_score >= 50 else "LOW",
            confidence_score=int(confidence_score),
            rule_flags=rule_flags,
            model_version=model_version,
        )

_GLOBAL_AUDIT_LOGGER: Optional[AuditLogger] = None

def get_audit_logger() -> AuditLogger:
    global _GLOBAL_AUDIT_LOGGER
    if _GLOBAL_AUDIT_LOGGER is None:
        _GLOBAL_AUDIT_LOGGER = AuditLogger()
    return _GLOBAL_AUDIT_LOGGER
