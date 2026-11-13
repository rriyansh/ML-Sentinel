"""
Security primitives: JWT, API keys, RBAC scopes, OIDC validation.

- Constant-time API-key comparison
- JWT with iss/aud/exp validation
- Scope-based RBAC
- OIDC JWKS-based verification (optional)
"""
from __future__ import annotations

import hmac
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

from jose import JWTError, jwt
from passlib.context import CryptContext

from ml_sentinel.core.errors import AuthenticationError, AuthorizationError
from ml_sentinel.core.logging import get_logger
from ml_sentinel.core.settings import get_settings

logger = get_logger(__name__)
_pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
_TOKEN_ISSUER = "ml-sentinel"
_TOKEN_AUDIENCE = "ml-sentinel-api"

ALLOWED_SCOPES = {"read", "predict", "explain", "train", "promote", "admin"}


# ── API keys ──────────────────────────────────────────────
def verify_api_key(candidate: str | None) -> bool:
    if not candidate:
        return False
    settings = get_settings()
    for key in settings.api_key_set():
        if hmac.compare_digest(candidate, key):
            return True
    return False


# ── Passwords (for admin tooling) ─────────────────────────
def hash_password(plain: str) -> str:
    return _pwd_ctx.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return _pwd_ctx.verify(plain, hashed)
    except Exception:  # noqa: BLE001
        return False


# ── JWT ───────────────────────────────────────────────────
def create_access_token(
    subject: str,
    *,
    scopes: list[str] | None = None,
    expires_minutes: int | None = None,
    extra_claims: dict[str, Any] | None = None,
) -> str:
    settings = get_settings()
    _validate_scopes(scopes or [])
    now = datetime.now(timezone.utc)
    expire = now + timedelta(minutes=expires_minutes or settings.jwt_expire_minutes)
    payload: dict[str, Any] = {
        "sub": subject,
        "iss": _TOKEN_ISSUER,
        "aud": _TOKEN_AUDIENCE,
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
        "scopes": scopes or ["predict"],
        "jti": secrets.token_urlsafe(16),
    }
    if extra_claims:
        payload.update(extra_claims)
    token = jwt.encode(
        payload,
        settings.secret_key.get_secret_value(),
        algorithm=settings.jwt_algorithm,
    )
    logger.info("token_issued", subject=subject, expires_at=expire.isoformat(),
                scopes=payload["scopes"])
    return token


def create_refresh_token(subject: str) -> str:
    settings = get_settings()
    now = datetime.now(timezone.utc)
    expire = now + timedelta(days=settings.jwt_refresh_expire_days)
    payload = {
        "sub": subject,
        "iss": _TOKEN_ISSUER,
        "aud": _TOKEN_AUDIENCE,
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
        "typ": "refresh",
        "jti": secrets.token_urlsafe(16),
    }
    return jwt.encode(
        payload,
        settings.secret_key.get_secret_value(),
        algorithm=settings.jwt_algorithm,
    )


def decode_access_token(token: str) -> dict[str, Any]:
    settings = get_settings()
    try:
        payload = jwt.decode(
            token,
            settings.secret_key.get_secret_value(),
            algorithms=[settings.jwt_algorithm],
            audience=_TOKEN_AUDIENCE,
            issuer=_TOKEN_ISSUER,
            options={"require": ["exp", "iat", "sub", "iss", "aud"]},
        )
    except JWTError as exc:
        logger.warning("token_invalid", error=str(exc))
        raise AuthenticationError(f"Invalid token: {exc}") from exc
    return payload


# ── RBAC ──────────────────────────────────────────────────
def _validate_scopes(scopes: list[str]) -> None:
    unknown = set(scopes) - ALLOWED_SCOPES
    if unknown:
        raise AuthorizationError(f"Unknown scopes requested: {unknown}")


def require_scopes(required: list[str]):
    """FastAPI dependency factory: enforces required scopes."""

    def _checker(request) -> None:
        granted = getattr(request.state, "scopes", []) or []
        if "admin" in granted:
            return
        missing = set(required) - set(granted)
        if missing:
            raise AuthorizationError(f"Missing scopes: {missing}")

    return _checker


# ── OIDC (JWKS verification) ──────────────────────────────
async def verify_oidc_token(token: str) -> dict[str, Any]:
    """
    Verify an OIDC JWT against the configured issuer's JWKS.

    Requires ``authlib`` — loaded lazily so dev/test runs without OIDC work.
    """
    settings = get_settings()
    if not settings.oidc_enabled:
        raise AuthenticationError("OIDC not enabled")

    from authlib.jose import JsonWebToken
    from authlib.jose.errors import JoseError
    import httpx

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            discovery = await client.get(
                f"{settings.oidc_issuer_url.rstrip('/')}/.well-known/openid-configuration"
            )
            discovery.raise_for_status()
            jwks_uri = discovery.json()["jwks_uri"]
            jwks = (await client.get(jwks_uri)).json()

        jwt_lib = JsonWebToken(["RS256", "RS384", "RS512", "ES256"])
        claims = jwt_lib.decode(token, jwks)
        claims.validate()
        if settings.oidc_audience and claims.get("aud") != settings.oidc_audience:
            raise AuthenticationError("Audience mismatch")
        return dict(claims)
    except (JoseError, httpx.HTTPError) as exc:
        logger.warning("oidc_verification_failed", error=str(exc))
        raise AuthenticationError(f"OIDC verification failed: {exc}") from exc
