from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64

driver = Entity(name="driver_id", join_keys=["driver_id"], description="Driver identifier")

driver_stats_source = FileSource(
    path="data/driver_stats.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created",
)

driver_stats_view = FeatureView(
    name="driver_stats",
    entities=[driver],
    ttl=timedelta(days=7),
    source=driver_stats_source,
    fields=[
        Field(name="conv_rate", dtype=Float32),
        Field(name="acc_rate", dtype=Float32),
        Field(name="avg_daily_trips", dtype=Int64),
    ],
    online=True,
    tags={"team": "ml-sentinel-enhanced"},
)
