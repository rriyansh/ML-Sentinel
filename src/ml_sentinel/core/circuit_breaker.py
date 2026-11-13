"""
Circuit breaker with three states (CLOSED → OPEN → HALF_OPEN) and
jittered recovery to prevent thundering-herd retries.
"""
from __future__ import annotations

import random
import threading
import time
from enum import IntEnum
from typing import Callable, TypeVar

from ml_sentinel.core.errors import CircuitOpenError
from ml_sentinel.core.logging import get_logger
from ml_sentinel.core.metrics import CIRCUIT_STATE

logger = get_logger(__name__)
T = TypeVar("T")


class State(IntEnum):
    CLOSED = 0
    HALF_OPEN = 1
    OPEN = 2


class CircuitBreaker:
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_seconds: float = 30.0,
        half_open_max_calls: int = 1,
        jitter_ratio: float = 0.2,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_seconds = recovery_seconds
        self.half_open_max_calls = half_open_max_calls
        self.jitter_ratio = jitter_ratio
        self._state = State.CLOSED
        self._failures = 0
        self._opened_at = 0.0
        self._half_open_calls = 0
        self._lock = threading.RLock()
        self._update_metric()

    @property
    def state(self) -> State:
        with self._lock:
            self._maybe_transition_to_half_open()
            return self._state

    def _update_metric(self) -> None:
        CIRCUIT_STATE.labels(dependency=self.name).set(int(self._state))

    def _maybe_transition_to_half_open(self) -> None:
        if self._state == State.OPEN and time.time() - self._opened_at >= self.recovery_seconds:
            self._state = State.HALF_OPEN
            self._half_open_calls = 0
            self._update_metric()
            logger.info("circuit_half_open", dependency=self.name)

    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        with self._lock:
            self._maybe_transition_to_half_open()

            if self._state == State.OPEN:
                raise CircuitOpenError(f"Circuit open for {self.name}")

            if self._state == State.HALF_OPEN:
                if self._half_open_calls >= self.half_open_max_calls:
                    raise CircuitOpenError(f"Circuit half-open for {self.name}, no slot")
                self._half_open_calls += 1

        try:
            result = func(*args, **kwargs)
        except Exception:
            self._on_failure()
            raise
        else:
            self._on_success()
            return result

    def _on_success(self) -> None:
        with self._lock:
            if self._state == State.HALF_OPEN:
                logger.info("circuit_closed", dependency=self.name)
            self._failures = 0
            self._state = State.CLOSED
            self._update_metric()

    def _on_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= self.failure_threshold or self._state == State.HALF_OPEN:
                self._state = State.OPEN
                jitter = self.recovery_seconds * self.jitter_ratio * random.random()
                self._opened_at = time.time()
                self._update_metric()
                logger.warning("circuit_open", dependency=self.name,
                               failures=self._failures, jitter=jitter)
