"""Create all database tables (dev/CI convenience; use Alembic in production)."""

from __future__ import annotations

from ml_sentinel.database import init_db, ping_db
from ml_sentinel.logger import get_logger

logger = get_logger("init_db")


def main() -> None:
    if not ping_db():
        raise SystemExit("Cannot reach database — check DATABASE_URL")
    init_db()
    logger.info("db_init_done")


if __name__ == "__main__":
    main()
