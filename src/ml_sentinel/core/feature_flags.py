"""
Simple YAML-backed feature flags with default fallback.

Hot-reloaded when file mtime changes (no restart required).
"""
from __future__ import annotations

from pathlib import Path
from threading import RLock

import yaml

from ml_sentinel.core.logging import get_logger

logger = get_logger(__name__)

_FLAGS_PATH = Path(__file__).resolve().parent.parent.parent.parent / "config" / "feature_flags.yaml"
_lock = RLock()
_cache: dict[str, bool] = {}
_mtime: float = 0.0


def _load() -> None:
    global _cache, _mtime
    try:
        st = _FLAGS_PATH.stat()
        if st.st_mtime == _mtime and _cache:
            return
        with _FLAGS_PATH.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        flags = raw.get("flags", {}) or {}
        _cache = {name: bool(cfg.get("enabled", False)) for name, cfg in flags.items()}
        _mtime = st.st_mtime
        logger.info("feature_flags_loaded", flags=_cache)
    except FileNotFoundError:
        logger.warning("feature_flags_file_missing", path=str(_FLAGS_PATH))
        _cache = {}


def is_enabled(flag: str, default: bool = False) -> bool:
    with _lock:
        _load()
        return _cache.get(flag, default)
