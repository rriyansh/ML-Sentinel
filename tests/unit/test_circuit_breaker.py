"""Circuit breaker tests."""
from __future__ import annotations

import time

import pytest

from ml_sentinel.core.circuit_breaker import CircuitBreaker, State
from ml_sentinel.core.errors import CircuitOpenError


def test_circuit_opens_after_failures():
    cb = CircuitBreaker("test", failure_threshold=2, recovery_seconds=1)
    def boom(): raise RuntimeError("fail")
    for _ in range(2):
        with pytest.raises(RuntimeError):
            cb.call(boom)
    assert cb.state == State.OPEN
    with pytest.raises(CircuitOpenError):
        cb.call(lambda: "ok")


def test_circuit_recovers():
    cb = CircuitBreaker("test2", failure_threshold=1, recovery_seconds=0.2)
    def boom(): raise RuntimeError("fail")
    with pytest.raises(RuntimeError):
        cb.call(boom)
    time.sleep(0.25)
    assert cb.state == State.HALF_OPEN
    assert cb.call(lambda: "ok") == "ok"
    assert cb.state == State.CLOSED
