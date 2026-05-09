import json
from functools import lru_cache
from typing import Any, Dict

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import AVAILABLE_MODELS, MODEL_METADATA_PATH, THRESHOLD, LGBM_GAN_FEATURE_ORDER_PATH
from src.preprocessing import preprocess_transactions
from src.gan import calculate_gan_anomaly_score
from src.preprocessing import load_json

@lru_cache(maxsize=2)
def load_model(model_name: str):
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model: {model_name}")

    model_path = AVAILABLE_MODELS[model_name]["path"]

    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")

    return lgb.Booster(model_file=str(model_path))


@lru_cache(maxsize=1)
def load_model_metadata() -> Dict[str, Any]:
    if not MODEL_METADATA_PATH.exists():
        return {}

    with open(MODEL_METADATA_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def calculate_demo_anomaly_score(features: pd.DataFrame) -> pd.Series:
    numeric_features = features.select_dtypes(include=[np.number])

    if numeric_features.empty:
        return pd.Series(0.0, index=features.index)

    z_scores = np.abs(
        (numeric_features - numeric_features.mean()) / (numeric_features.std() + 1e-6)
    )

    score = z_scores.mean(axis=1)
    score = (score - score.min()) / (score.max() - score.min() + 1e-6)

    return score.fillna(0.0)


def align_features_to_model(
    features: pd.DataFrame,
    raw_df: pd.DataFrame,
    model,
    model_name: str,
) -> pd.DataFrame:
    if model_name == "lgbm_gan":
        model_features = load_json(LGBM_GAN_FEATURE_ORDER_PATH)
    else:
        model_features = model.feature_name()

    aligned = features.copy()

    for col in model_features:
        if col not in aligned.columns:
            if col in raw_df.columns:
                aligned[col] = raw_df[col].values
            else:
                aligned[col] = 0

    aligned = aligned[model_features]
    aligned = aligned.apply(pd.to_numeric, errors="coerce").fillna(0)

    return aligned


def prepare_features(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    model = load_model(model_name)

    features = preprocess_transactions(df)

    if AVAILABLE_MODELS[model_name]["uses_anomaly_score"]:
        features = features.copy()
        features["anomaly_score"] = calculate_gan_anomaly_score(features, df)

    features = align_features_to_model(features, df, model, model_name)

    return features


def predict_transaction(df: pd.DataFrame, model_name: str) -> Dict[str, Any]:
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model: {model_name}")

    model = load_model(model_name)
    features = prepare_features(df, model_name)

    probabilities = model.predict(features)
    fraud_probability = float(probabilities[0])
    prediction = int(fraud_probability >= THRESHOLD)

    metadata = load_model_metadata()

    from src.explainability import explain_transaction

    explanation = explain_transaction(df, model_name)

    return {
        "model_name": model_name,
        "model_display_name": AVAILABLE_MODELS[model_name]["display_name"],
        "model_version": metadata.get("version", "1.0.0"),
        "fraud_probability": fraud_probability,
        "prediction": prediction,
        "prediction_label": "fraud" if prediction == 1 else "non-fraud",
        "threshold": THRESHOLD,
        "explanation": explanation,
    }


def predict_batch(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model: {model_name}")

    model = load_model(model_name)
    features = prepare_features(df, model_name)

    probabilities = model.predict(features)
    predictions = (probabilities >= THRESHOLD).astype(int)

    result = df.copy()
    result["fraud_probability"] = probabilities
    result["prediction"] = predictions
    result["prediction_label"] = np.where(predictions == 1, "fraud", "non-fraud")

    return result