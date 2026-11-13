"""Redis cache with in-memory fallback."""
from __future__ import annotations

import json
import time
from typing import Any

from ml_sentinel.core.logging import get_logger
from ml_sentinel.core.settings import get_settings

logger = get_logger(__name__)

try:
    import redis as _redis
except ImportError:
    _redis = None


class _MemoryCache:
    def __init__(self) -> None:
        self._store: dict[str, tuple[str, float]] = {}

    def get(self, key: str) -> str | None:
        entry = self._store.get(key)
        if not entry:
            return None
        val, exp = entry
        if exp and time.time() > exp:
            self._store.pop(key, None)
            return None
        return val

    def setex(self, key: str, ttl: int, value: str) -> None:
        self._store[key] = (value, time.time() + ttl)


_cache = None


def _get_cache():
    global _cache
    if _cache is not None:
        return _cache
    s = get_settings()
    if _redis is not None:
        try:
            _cache = _redis.from_url(s.redis_url, decode_responses=True)
            _cache.ping()
            logger.info("redis_cache_active")
            return _cache
        except Exception as exc:  # noqa: BLE001
            logger.warning("redis_cache_unavailable_fallback", error=str(exc))
    _cache = _MemoryCache()
    return _cache


def cache_get_json(key: str) -> Any:
    raw = _get_cache().get(key)
    return json.loads(raw) if raw else None


def cache_set_json(key: str, value: Any, ttl: int = 60) -> None:
    _get_cache().setex(key, ttl, json.dumps(value, default=str))
