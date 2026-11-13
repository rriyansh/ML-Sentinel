"""
Field-level encryption for PII at rest (Fernet = AES-128-CBC + HMAC-SHA256).
"""
from __future__ import annotations

from cryptography.fernet import Fernet, InvalidToken

from ml_sentinel.core.errors import ConfigurationError
from ml_sentinel.core.logging import get_logger
from ml_sentinel.core.settings import get_settings

logger = get_logger(__name__)
_fernet: Fernet | None = None


def _get_fernet() -> Fernet:
    global _fernet
    if _fernet is None:
        key = get_settings().encryption_key.get_secret_value().encode()
        try:
            _fernet = Fernet(key)
        except Exception as exc:  # noqa: BLE001
            raise ConfigurationError(f"Invalid ENCRYPTION_KEY: {exc}") from exc
    return _fernet


def encrypt_field(plaintext: str) -> str:
    if not plaintext:
        return plaintext
    return _get_fernet().encrypt(plaintext.encode()).decode()


def decrypt_field(ciphertext: str) -> str:
    if not ciphertext:
        return ciphertext
    try:
        return _get_fernet().decrypt(ciphertext.encode()).decode()
    except InvalidToken as exc:
        logger.error("decrypt_failed")
        raise ConfigurationError("Could not decrypt field — key mismatch?") from exc


def encrypt_dict(data: dict, fields: list[str]) -> dict:
    """Return a copy with the given string fields encrypted."""
    out = dict(data)
    for f in fields:
        if f in out and isinstance(out[f], str):
            out[f] = encrypt_field(out[f])
    return out


def decrypt_dict(data: dict, fields: list[str]) -> dict:
    out = dict(data)
    for f in fields:
        if f in out and isinstance(out[f], str):
            out[f] = decrypt_field(out[f])
    return out


def mask_pii(value: str, visible_prefix: int = 2, mask_char: str = "*") -> str:
    """Return a masked version (e.g. "ab*****") for safe logging."""
    if not value:
        return value
    if len(value) <= visible_prefix:
        return mask_char * len(value)
    return value[:visible_prefix] + mask_char * (len(value) - visible_prefix)
