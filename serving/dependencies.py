"""FastAPI dependencies: auth (JWT/API-key/OIDC), DB, RBAC, rate limit."""
from __future__ import annotations

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from ml_sentinel.core.errors import AuthenticationError, AuthorizationError, RateLimitError
from ml_sentinel.core.logging import get_logger
from ml_sentinel.core.metrics import AUTH_FAILURES, RATE_LIMITED
from ml_sentinel.core.security import decode_access_token, verify_api_key, verify_oidc_token
from ml_sentinel.core.settings import get_settings
from ml_sentinel.infrastructure.database import get_db
from ml_sentinel.infrastructure.rate_limiter import get_rate_limiter

logger = get_logger("serving.deps")
settings = get_settings()

_bearer = HTTPBearer(auto_error=False)
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def rate_limit(request: Request) -> None:
    ip = request.client.host if request.client else "unknown"
    try:
        get_rate_limiter().check(f"{ip}:{request.url.path}")
    except RateLimitError:
        RATE_LIMITED.labels(endpoint=request.url.path).inc()
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers={"Retry-After": "60"},
        ) from None


async def _authenticate(
    credentials: HTTPAuthorizationCredentials | None,
    api_key: str | None,
    request: Request,
) -> tuple[str, list[str]]:
    """
    Return (subject, scopes).
    Priority: OIDC → HS256 JWT → API key (dev only).
    """
    # 1. OIDC
    if settings.oidc_enabled and credentials is not None:
        try:
            claims = await verify_oidc_token(credentials.credentials)
            sub = claims.get("sub", "oidc-user")
            scopes = claims.get("scope", "").split() or ["predict"]
            request.state.subject = sub
            request.state.scopes = scopes
            return sub, scopes
        except AuthenticationError as exc:
            AUTH_FAILURES.labels(reason="oidc").inc()
            logger.warning("oidc_failed", error=str(exc))

    # 2. JWT (HS256)
    if credentials is not None:
        try:
            payload = decode_access_token(credentials.credentials)
            sub = payload.get("sub", "jwt-user")
            scopes = payload.get("scopes", ["predict"])
            request.state.subject = sub
            request.state.scopes = scopes
            return sub, scopes
        except AuthenticationError:
            AUTH_FAILURES.labels(reason="jwt").inc()

    # 3. API key (rejected in production unless OIDC is off AND API_KEYS rotated)
    if verify_api_key(api_key):
        if settings.is_production and settings.oidc_enabled:
            AUTH_FAILURES.labels(reason="api_key_production_disallowed").inc()
            raise HTTPException(401, "Use OIDC in production")
        request.state.subject = "api-key-user"
        request.state.scopes = ["predict", "explain"]
        return "api-key-user", ["predict", "explain"]

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def require_auth(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    api_key: str | None = Depends(_api_key_header),
) -> str:
    subject, _ = await _authenticate(credentials, api_key, request)
    return subject


def require_scope(scope: str):
    async def _checker(request: Request) -> None:
        scopes = getattr(request.state, "scopes", []) or []
        if "admin" in scopes or scope in scopes:
            return
        AUTH_FAILURES.labels(reason="missing_scope").inc()
        raise HTTPException(status_code=403, detail=f"Missing scope: {scope}")
    return _checker


__all__ = ["get_db", "rate_limit", "require_auth", "require_scope", "Session"]
