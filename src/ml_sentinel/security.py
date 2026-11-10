"""
Security primitives: JWT issuance/verification and API-key authentication.

- API keys are compared with ``secrets.compare_digest`` to avoid timing attacks.
- JWTs are signed with HS256 (configurable) and carry a ``sub`` + ``iss`` claim.
- All sensitive settings use ``SecretStr`` and are never logged.
"""
from __future__ import annotations

import hmac
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

from jose import JWTError, jwt
from passlib.context import CryptContext

from ml_sentinel.logger import get_logger
from ml_sentinel.settings import get_settings

logger = get_logger(__name__)
_pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")

_TOKEN_ISSUER = "ml-sentinel"


# ── API keys ────────────────────────────────────────────────────────
def verify_api_key(candidate: str | None) -> bool:
    """Constant-time check against the configured API key set."""
    if not candidate:
        return False
    settings = get_settings()
    for key in settings.api_key_set():
        if hmac.compare_digest(candidate, key):
            return True
    return False


# ── Passwords (for admin tooling; not used for API auth) ────────────
def hash_password(plain: str) -> str:
    return _pwd_ctx.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return _pwd_ctx.verify(plain, hashed)
    except Exception:  # noqa: BLE001
        return False


# ── JWT ─────────────────────────────────────────────────────────────
def create_access_token(
    subject: str,
    *,
    scopes: list[str] | None = None,
    expires_minutes: int | None = None,
) -> str:
    """Create a signed JWT for the given subject (usually the API-key owner)."""
    settings = get_settings()
    now = datetime.now(timezone.utc)
    expire = now + timedelta(minutes=expires_minutes or settings.jwt_expire_minutes)
    payload: dict[str, Any] = {
        "sub": subject,
        "iss": _TOKEN_ISSUER,
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
        "scopes": scopes or ["predict"],
        "jti": secrets.token_urlsafe(16),
    }
    token = jwt.encode(
        payload,
        settings.secret_key.get_secret_value(),
        algorithm=settings.jwt_algorithm,
    )
    logger.info("token_issued", subject=subject, expires_at=expire.isoformat())
    return token


def decode_access_token(token: str) -> dict[str, Any]:
    """Decode & validate a JWT. Raises ``JWTError`` on failure."""
    settings = get_settings()
    try:
        payload = jwt.decode(
            token,
            settings.secret_key.get_secret_value(),
            algorithms=[settings.jwt_algorithm],
            issuer=_TOKEN_ISSUER,
            options={"require": ["exp", "iat", "sub", "iss"]},
        )
    except JWTError as exc:
        logger.warning("token_invalid", error=str(exc))
        raise
    return payload
