from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    conv_rate: float = Field(..., ge=0, le=1, description="Conversion rate between 0 and 1")
    acc_rate: float = Field(..., ge=0, le=1, description="Accuracy rate between 0 and 1")

class PredictionResponse(BaseModel):
    prediction: list[float]
    model_version: str
