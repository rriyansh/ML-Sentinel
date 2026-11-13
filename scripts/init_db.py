"""Initialize DB tables."""
from __future__ import annotations

from ml_sentinel.core.logging import get_logger
from ml_sentinel.infrastructure.database import init_db, ping_db

logger = get_logger("init_db")


def main() -> None:
    if not ping_db():
        raise SystemExit("Cannot reach database")
    init_db()
    logger.info("db_init_done")


if __name__ == "__main__":
    main()
