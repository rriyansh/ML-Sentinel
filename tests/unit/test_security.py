"""Security tests."""
from __future__ import annotations

import os

import pytest

from ml_sentinel.core.errors import AuthenticationError, AuthorizationError
from ml_sentinel.core.security import (
    _validate_scopes,
    create_access_token, decode_access_token,
    hash_password, verify_api_key, verify_password,
)

os.environ.setdefault("SECRET_KEY", "unit-test-secret-key-with-min-32-characters-x")
os.environ.setdefault("API_KEYS", "unit-test-key-1,unit-test-key-2")


def test_api_key_accepts_configured():
    assert verify_api_key("unit-test-key-1") is True
    assert verify_api_key("unit-test-key-2") is True


def test_api_key_rejects_unknown():
    assert verify_api_key("nope") is False
    assert verify_api_key(None) is False


def test_jwt_roundtrip():
    token = create_access_token(subject="tester", scopes=["predict"])
    claims = decode_access_token(token)
    assert claims["sub"] == "tester"
    assert claims["iss"] == "ml-sentinel"
    assert "predict" in claims["scopes"]


def test_jwt_invalid_raises():
    token = create_access_token(subject="tester")[:-5] + "xxxxx"
    with pytest.raises(AuthenticationError):
        decode_access_token(token)


def test_unknown_scope_rejected():
    with pytest.raises(AuthorizationError):
        _validate_scopes(["delete_everything"])


def test_password_hashing():
    hashed = hash_password("correct-horse")
    assert verify_password("correct-horse", hashed)
    assert not verify_password("wrong", hashed)
