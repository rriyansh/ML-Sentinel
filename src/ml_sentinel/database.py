"""
Database engine + session factory.

Uses SQLAlchemy 2.x with a pooled engine. Provides:
- ``get_engine()``   — cached engine factory
- ``session_scope()`` — context-managed transaction
- ``get_db()``       — FastAPI dependency
- ``init_db()``      — create tables (for dev/CI; use Alembic in prod)
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from ml_sentinel.logger import get_logger
from ml_sentinel.settings import get_settings

logger = get_logger(__name__)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def get_engine() -> Engine:
    """Return the shared, pooled SQLAlchemy engine."""
    global _engine, _SessionLocal
    if _engine is not None:
        return _engine

    settings = get_settings()
    _engine = create_engine(
        settings.database_url,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_timeout=settings.db_pool_timeout,
        pool_pre_ping=True,
        future=True,
    )
    _SessionLocal = sessionmaker(
        bind=_engine, autocommit=False, autoflush=False, expire_on_commit=False
    )
    logger.info("db_engine_created", url=_engine.url.render_as_string(hide_password=True))
    return _engine


def get_sessionmaker() -> sessionmaker[Session]:
    global _SessionLocal
    if _SessionLocal is None:
        get_engine()
    assert _SessionLocal is not None
    return _SessionLocal


@contextmanager
def session_scope() -> Iterator[Session]:
    """Provide a transactional scope with commit/rollback semantics."""
    session = get_sessionmaker()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        logger.exception("db_session_rollback")
        raise
    finally:
        session.close()


def get_db() -> Iterator[Session]:
    """FastAPI dependency that yields a scoped DB session."""
    with session_scope() as session:
        yield session


def init_db() -> None:
    """Create all tables (idempotent). Use Alembic in production."""
    from ml_sentinel import db_models  # noqa: F401  (side-effect: register models)

    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    logger.info("db_init_complete", tables=list(Base.metadata.tables.keys()))


def ping_db() -> bool:
    """Lightweight health check."""
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:  # noqa: BLE001
        logger.exception("db_ping_failed")
        return False
