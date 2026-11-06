from pydantic import BaseModel, Field


class DriverFeatures(BaseModel):
    conv_rate: float = Field(..., ge=0.0, le=1.0, description="Conversion rate")
    acc_rate: float = Field(..., ge=0.0, le=1.0, description="Acceptance rate")
    avg_daily_trips: int = Field(..., ge=0, description="Average daily trips")


class PredictionResponse(BaseModel):
    prediction: list
    model_version: str = "1.0.0"
