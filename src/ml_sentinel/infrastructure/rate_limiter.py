"""
Redis-backed token-bucket rate limiter (distributed).

Falls back to in-memory bucket if Redis is unavailable (single-node only).
"""
from __future__ import annotations

import time
from collections import defaultdict, deque

from ml_sentinel.core.errors import RateLimitError
from ml_sentinel.core.logging import get_logger
from ml_sentinel.core.settings import get_settings

logger = get_logger(__name__)
_WINDOW = 60  # seconds

try:
    import redis as _redis
except ImportError:  # pragma: no cover
    _redis = None


class InMemoryRateLimiter:
    def __init__(self, limit: int, burst: int) -> None:
        self.limit = limit
        self.burst = burst
        self._buckets: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str) -> None:
        now = time.time()
        bucket = self._buckets[key]
        while bucket and now - bucket[0] > _WINDOW:
            bucket.popleft()
        if len(bucket) >= self.limit + self.burst:
            raise RateLimitError(f"Rate limit exceeded for {key}")
        bucket.append(now)


class RedisRateLimiter:
    def __init__(self, redis_url: str, prefix: str, limit: int, burst: int) -> None:
        assert _redis is not None
        self.client = _redis.from_url(redis_url, decode_responses=True)
        self.prefix = prefix
        self.limit = limit
        self.burst = burst

    def check(self, key: str) -> None:
        now = int(time.time())
        bucket_key = f"{self.prefix}{key}:{now // _WINDOW}"
        try:
            count = self.client.incr(bucket_key)
            if count == 1:
                self.client.expire(bucket_key, _WINDOW * 2)
            if count > self.limit + self.burst:
                raise RateLimitError(f"Rate limit exceeded for {key}")
        except Exception as exc:  # noqa: BLE001
            if isinstance(exc, RateLimitError):
                raise
            logger.warning("redis_rate_limit_failed_fallback", error=str(exc))
            # Fail open (allow) — safer than 500ing on Redis outage for non-critical traffic
            return


_limiter = None


def get_rate_limiter():
    global _limiter
    if _limiter is not None:
        return _limiter
    s = get_settings()
    if _redis is not None:
        try:
            _limiter = RedisRateLimiter(s.redis_url, s.redis_rate_limit_prefix,
                                        s.rate_limit_per_minute, s.rate_limit_burst)
            logger.info("redis_rate_limiter_active")
            return _limiter
        except Exception as exc:  # noqa: BLE001
            logger.warning("redis_init_failed_fallback_memory", error=str(exc))
    _limiter = InMemoryRateLimiter(s.rate_limit_per_minute, s.rate_limit_burst)
    return _limiter
