"""
VIDYUT Audit Database
===========================================================================
SQLAlchemy ORM model for the audit_events table.
Schema is auto-created on first use.
===========================================================================
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from src.config.settings import get_settings
from src.utils.db import Base, build_engine

settings = get_settings()

_AUDIT_ENGINE = None


def get_audit_engine(db_path: Optional[str] = None):
    """Return (and cache) the audit database engine, creating tables if needed."""
    global _AUDIT_ENGINE
    if _AUDIT_ENGINE is None:
        path = db_path or settings.database_url or str(settings.audit_db_path)
        _AUDIT_ENGINE = build_engine(path)
        Base.metadata.create_all(_AUDIT_ENGINE)
    return _AUDIT_ENGINE


class AuditEvent(Base):
    """
    Immutable audit event record.

    Every model prediction, generated alert, and system action
    writes one row here. No UPDATE or DELETE queries are ever run.
    """

    __tablename__ = "audit_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    entity_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    user_id: Mapped[str] = mapped_column(String(64), nullable=False, default="system")
    session_id: Mapped[str] = mapped_column(String(64), nullable=True, default="")
    details_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    def __repr__(self) -> str:
        return (
            f"<AuditEvent id={self.id} type={self.event_type} "
            f"entity={self.entity_id} at={self.created_at}>"
        )

def get_audit_db():
    from src.audit.logger import get_audit_logger
    return get_audit_logger()
