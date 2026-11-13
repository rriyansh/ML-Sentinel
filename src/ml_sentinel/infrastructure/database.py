"""
SQLAlchemy 2.x engine + session factory.

Features:
- Pooled engine (pre-ping, statement timeout)
- Context-managed sessions with rollback-on-error
- FastAPI dependency
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from ml_sentinel.core.logging import get_logger
from ml_sentinel.core.settings import get_settings

logger = get_logger(__name__)


class Base(DeclarativeBase):
    """ORM declarative base."""


_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def get_engine() -> Engine:
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

    # PostgreSQL statement timeout
    if "postgres" in settings.database_url:
        @event.listens_for(_engine, "connect")
        def _set_stmt_timeout(dbapi_conn, _):
            with dbapi_conn.cursor() as cur:
                cur.execute(f"SET statement_timeout = {settings.db_statement_timeout_ms}")
                cur.close()

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
    with session_scope() as session:
        yield session


def init_db() -> None:
    from ml_sentinel.infrastructure import db_models  # noqa: F401

    Base.metadata.create_all(bind=get_engine())
    logger.info("db_init_complete", tables=list(Base.metadata.tables.keys()))


def ping_db() -> bool:
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:  # noqa: BLE001
        logger.exception("db_ping_failed")
        return False
