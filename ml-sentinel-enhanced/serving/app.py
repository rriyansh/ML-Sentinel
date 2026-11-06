import os

import mlflow.pyfunc
from fastapi import FastAPI, HTTPException, Security, status
from fastapi.security import APIKeyHeader

from schemas import DriverFeatures, PredictionResponse

MODEL_URI = os.getenv("MODEL_URI", "models:/ml_sentinel_driver/Production")
API_KEY = os.getenv("API_KEY", "ml-sentinel-enhanced-key")

app = FastAPI(title="ML-Sentinel Enhanced Serving", version="1.0.0")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

model = None


def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
    return api_key


@app.on_event("startup")
def load_model() -> None:
    global model
    model = mlflow.pyfunc.load_model(MODEL_URI)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse, dependencies=[Security(verify_api_key)])
def predict(features: DriverFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    import pandas as pd

    df = pd.DataFrame([features.model_dump()])
    prediction = model.predict(df)
    return PredictionResponse(prediction=prediction.tolist())
