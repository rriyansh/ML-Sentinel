import mlflow.pyfunc
from fastapi import FastAPI, HTTPException
from serving.schemas import PredictionRequest, PredictionResponse
from src.ml_sentinel.logger import setup_logger

logger = setup_logger("serving")
app = FastAPI(title="ML-Sentinel Serving API")

model = None
model_version = "not_loaded"

@app.on_event("startup")
async def load_model():
    global model, model_version
    try:
        # Load the Production model from the registry
        model = mlflow.pyfunc.load_model(model_uri="models:/MLSentinelModel/Production")
        model_version = "Production"
        logger.info("Production model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")

@app.get("/health/liveness")
def liveness():
    return {"status": "alive"}

@app.get("/health/readiness")
def readiness():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready", "model_version": model_version}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    data = [[request.conv_rate, request.acc_rate]]
    prediction = model.predict(data)
    return PredictionResponse(prediction=prediction.tolist(), model_version=model_version)
