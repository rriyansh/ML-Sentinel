"""
Generate a synthetic ``driver_stats.parquet`` used by Feast and training.

Idempotent: safe to run multiple times. Produces ~2000 rows spanning 30 days.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def main(rows: int = 2000, days: int = 30, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)

    out_dir = Path(__file__).resolve().parent.parent / "feast_repo" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "driver_stats.parquet"

    now = pd.Timestamp.utcnow().tz_localize(None)
    start = now - pd.Timedelta(days=days)

    timestamps = pd.to_datetime(
        rng.integers(
            start.value // 10**9, now.value // 10**9, size=rows
        ),
        unit="s",
    )

    df = pd.DataFrame(
        {
            "driver_id": rng.integers(1000, 1100, size=rows, dtype=np.int64),
            "event_timestamp": timestamps,
            "created_timestamp": timestamps + pd.Timedelta(minutes=5),
            "conv_rate": np.clip(rng.beta(2, 5, size=rows), 0.0, 1.0).astype("float32"),
            "acc_rate": np.clip(rng.beta(5, 2, size=rows), 0.0, 1.0).astype("float32"),
            "avg_daily_trips": rng.integers(1, 100, size=rows, dtype=np.int64),
        }
    ).sort_values("event_timestamp").reset_index(drop=True)

    df.to_parquet(out_path, index=False)
    print(f"Wrote {len(df):,} rows → {out_path}")

    # Also mirror to data/reference for drift comparison
    ref_dir = Path(__file__).resolve().parent.parent / "data" / "reference"
    ref_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(ref_dir / "driver_stats.parquet", index=False)
    print(f"Wrote reference dataset → {ref_dir / 'driver_stats.parquet'}")


if __name__ == "__main__":
    main()
