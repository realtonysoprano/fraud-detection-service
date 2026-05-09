from functools import lru_cache
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import shap

from src.inference import load_model, prepare_features


@lru_cache(maxsize=2)
def load_shap_explainer(model_name: str):
    model = load_model(model_name)
    return shap.TreeExplainer(model)


def normalize_shap_values(shap_values):
    if isinstance(shap_values, list):
        return shap_values[1]

    if len(shap_values.shape) == 3:
        return shap_values[:, :, 1]

    return shap_values


def explain_transaction(
    df: pd.DataFrame,
    model_name: str,
    top_n: int = 10,
) -> List[Dict[str, Any]]:
    features = prepare_features(df, model_name)
    explainer = load_shap_explainer(model_name)

    shap_values = explainer.shap_values(features)
    shap_values = normalize_shap_values(np.array(shap_values))

    row_values = shap_values[0]
    feature_values = features.iloc[0]

    indices = np.argsort(np.abs(row_values))[::-1][:top_n]

    explanation = []

    for idx in indices:
        feature = features.columns[idx]
        shap_value = float(row_values[idx])
        value = feature_values.iloc[idx]

        explanation.append(
            {
                "feature": feature,
                "value": float(value),
                "shap_value": shap_value,
                "impact": abs(shap_value),
                "direction": "positive" if shap_value >= 0 else "negative",
            }
        )

    return explanation