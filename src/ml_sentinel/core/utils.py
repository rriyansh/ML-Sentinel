"""Shared utilities."""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pandas as pd


def new_request_id() -> str:
    return str(uuid.uuid4())


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def df_digest(df: pd.DataFrame) -> str:
    return hashlib.sha256(
        pd.util.hash_pandas_object(df, index=True).values.tobytes()
    ).hexdigest()


def dict_digest(d: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(d, sort_keys=True, default=str).encode()
    ).hexdigest()


@contextmanager
def timer() -> Iterator[callable]:
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start


def truncate(text: str, max_len: int = 500) -> str:
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def safe_json_load(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Config file not found: {p}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {p}: {exc}") from exc
