"""
VIDYUT Database Utility
===========================================================================
SQLAlchemy engine factory for SQLite (audit) and in-memory (testing).
Provides session context manager used by audit_db and other modules.
===========================================================================
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from src.utils.logger import get_logger

log = get_logger("vidyut.db")


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""
    pass


def build_engine(db_url_or_path: str | Path = ":memory:", echo: bool = False) -> Engine:
    """
    Create a SQLAlchemy engine for SQLite or PostgreSQL.

    Parameters
    ----------
    db_url_or_path : Path | str
        SQLAlchemy URL or path to .db file, or ':memory:' for an in-memory DB.
    echo : bool
        If True, SQLAlchemy echoes all SQL statements.
    """
    url = str(db_url_or_path)

    # Check if it's a URL (like postgresql:// or sqlite://)
    if "://" in url:
        pass
    elif url != ":memory:":
        db_path = Path(url)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        url = f"sqlite:///{db_path}"
    else:
        url = "sqlite:///:memory:"

    connect_args = {}
    if "sqlite" in url:
        connect_args["check_same_thread"] = False

    engine = create_engine(
        url,
        echo=echo,
        connect_args=connect_args,
        pool_pre_ping=True,
    )

    if "sqlite" in url:
        # Enable WAL mode for concurrent reads
        @event.listens_for(engine, "connect")
        def _set_wal(dbapi_conn, _conn_record):
            try:
                cursor = dbapi_conn.cursor()
                cursor.execute("PRAGMA journal_mode=WAL;")
                cursor.execute("PRAGMA synchronous=NORMAL;")
                cursor.close()
            except Exception as e:
                log.debug("Could not set WAL mode: %s", e)

    log.debug("SQLAlchemy engine created for: %s", url.split("@")[-1] if "@" in url else url)
    return engine



def build_session_factory(engine: Engine) -> sessionmaker:
    return sessionmaker(bind=engine, autocommit=False, autoflush=False)


@contextmanager
def get_session(engine: Engine) -> Generator[Session, None, None]:
    """Context manager yielding a SQLAlchemy Session with auto-rollback on error."""
    factory = build_session_factory(engine)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
