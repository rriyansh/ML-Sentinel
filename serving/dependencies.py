"""FastAPI dependencies: auth, DB, rate limiter."""

from __future__ import annotations

import time
from collections import defaultdict, deque

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError
from sqlalchemy.orm import Session

from ml_sentinel.database import get_db
from ml_sentinel.logger import get_logger
from ml_sentinel.metrics import RATE_LIMITED
from ml_sentinel.security import decode_access_token, verify_api_key
from ml_sentinel.settings import get_settings

logger = get_logger("serving.deps")
settings = get_settings()

_bearer = HTTPBearer(auto_error=False)
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


# ── In-memory rate limiter (use Redis in multi-node deployments) ──
_buckets: dict[str, deque[float]] = defaultdict(deque)
_WINDOW_SECONDS = 60


def rate_limit(request: Request) -> None:
    ip = request.client.host if request.client else "unknown"
    now = time.time()
    bucket = _buckets[ip]

    while bucket and now - bucket[0] > _WINDOW_SECONDS:
        bucket.popleft()

    if len(bucket) >= settings.rate_limit_per_minute:
        RATE_LIMITED.labels(endpoint=request.url.path).inc()
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers={"Retry-After": str(_WINDOW_SECONDS)},
        )
    bucket.append(now)


# ── Auth ───────────────────────────────────────────────────────────
def require_auth(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    api_key: str | None = Depends(_api_key_header),
) -> str:
    """
    Accept either a Bearer JWT or a raw API key header.
    Returns the authenticated subject (``sub`` claim or "api-key-user").
    """
    # 1. Bearer JWT preferred
    if credentials is not None:
        try:
            payload = decode_access_token(credentials.credentials)
            subject = payload.get("sub", "unknown")
            request.state.subject = subject
            return subject
        except JWTError:
            pass

    # 2. Fallback: raw API key
    if verify_api_key(api_key):
        request.state.subject = "api-key-user"
        return "api-key-user"

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )


__all__ = ["get_db", "rate_limit", "require_auth", "Session"]
