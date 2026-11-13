"""Encryption tests."""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("ENCRYPTION_KEY", "kNfK8XbRw3h9nPZQhG7jCkYe0gY2mAVa6IioK5zPq4U=")

from ml_sentinel.core.encryption import decrypt_field, encrypt_field, mask_pii


def test_roundtrip():
    plain = "hello@example.com"
    cipher = encrypt_field(plain)
    assert cipher != plain
    assert decrypt_field(cipher) == plain


def test_mask_pii():
    assert mask_pii("alice@x.com") == "al******************"
    assert mask_pii("ab") == "**"
    assert mask_pii("") == ""


def test_encrypt_empty():
    assert encrypt_field("") == ""
    assert decrypt_field("") == ""
