import os

import mlflow.pyfunc
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_URI = os.getenv("MODEL_URI", "models:/ml_sentinel_driver/Production")

app = FastAPI(title="ML-Sentinel Serving", version="1.0.0")

model = None


class DriverFeatures(BaseModel):
    conv_rate: float
    acc_rate: float
    avg_daily_trips: int


@app.on_event("startup")
def load_model() -> None:
    global model
    model = mlflow.pyfunc.load_model(MODEL_URI)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(features: DriverFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    import pandas as pd

    df = pd.DataFrame([features.model_dump()])
    prediction = model.predict(df)
    return {"prediction": prediction.tolist()}
