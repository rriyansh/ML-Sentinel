"""
Retry helper: exponential backoff + jitter via tenacity.

Decorator usage::

    @with_retry(max_attempts=3, base=0.5, max=10, jitter=True)
    def flaky_call(): ...
"""
from __future__ import annotations

import random
from typing import Callable, TypeVar

from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

T = TypeVar("T")


def with_retry(
    *,
    max_attempts: int = 3,
    base: float = 0.5,
    max: float = 10.0,  # noqa: A002
    retry_on: tuple[type[Exception], ...] = (Exception,),
    reraise: bool = True,
):
    """Decorator factory with jittered exponential backoff."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        wrapped = retry(
            reraise=reraise,
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential_jitter(initial=base, max=max, jitter=base * random.random()),
            retry=retry_if_exception_type(retry_on),
        )(func)
        return wrapped

    return decorator
