"""
Feast feature definitions for ML-Sentinel.

A single ``driver`` entity with hourly aggregated statistics serves as
the toy domain. Replace with your own entities / feature views when
adapting the platform.
"""
from __future__ import annotations

from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64
from feast.value_type import ValueType

# ── Entity ─────────────────────────────────────────────────────────
driver = Entity(
    name="driver_id",
    value_type=ValueType.INT64,
    description="Unique identifier for a driver (or user/customer in your domain)",
)

# ── Data source ────────────────────────────────────────────────────
driver_stats_source = FileSource(
    name="driver_stats_source",
    path="data/driver_stats.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# ── Feature view ───────────────────────────────────────────────────
driver_hourly_stats = FeatureView(
    name="driver_hourly_stats",
    entities=[driver],
    ttl=timedelta(days=365),
    schema=[
        Field(name="conv_rate", dtype=Float32, description="Conversion rate"),
        Field(name="acc_rate", dtype=Float32, description="Acceptance rate"),
        Field(name="avg_daily_trips", dtype=Int64, description="Average daily trips"),
    ],
    online=True,
    source=driver_stats_source,
    tags={"team": "ml-platform", "domain": "mobility"},
)
