from typing import Any, Dict, List

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.config import AVAILABLE_MODELS
from src.inference import load_model_metadata, predict_transaction


app = FastAPI(
    title="Fraud Detection Service",
    description="Service for detecting fraudulent financial transactions",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    model_name: str = "lgbm"
    transaction: Dict[str, Any]


class PredictionResponse(BaseModel):
    model_name: str
    model_display_name: str
    model_version: str
    fraud_probability: float
    prediction: int
    prediction_label: str
    threshold: float
    explanation: List[Dict[str, Any]]


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/models")
def models() -> Dict[str, Any]:
    return {
        "available_models": [
            {
                "name": name,
                "display_name": config["display_name"],
                "uses_anomaly_score": config["uses_anomaly_score"],
            }
            for name, config in AVAILABLE_MODELS.items()
        ]
    }


@app.get("/model_info")
def model_info() -> Dict[str, Any]:
    metadata = load_model_metadata()

    return {
        "metadata": metadata,
        "available_models": [
            {
                "name": name,
                "display_name": config["display_name"],
                "uses_anomaly_score": config["uses_anomaly_score"],
            }
            for name, config in AVAILABLE_MODELS.items()
        ],
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> Dict[str, Any]:
    try:
        df = pd.DataFrame([request.transaction])
        result = predict_transaction(df=df, model_name=request.model_name)
        return result

    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error