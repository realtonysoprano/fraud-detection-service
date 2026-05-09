import sys
from pathlib import Path
import json
from typing import Any, Dict

import pandas as pd
import requests
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import API_URL, SAMPLE_DATA_PATH


st.set_page_config(
    page_title="Fraud Detection Service",
    page_icon="💳",
    layout="wide",
)


def load_sample_data() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_DATA_PATH)


def get_models() -> Dict[str, Any]:
    response = requests.get(f"{API_URL}/models", timeout=10)
    response.raise_for_status()
    return response.json()


def get_model_info() -> Dict[str, Any]:
    response = requests.get(f"{API_URL}/model_info", timeout=10)
    response.raise_for_status()
    return response.json()


def predict_transaction(
    transaction: Dict[str, Any],
    model_name: str,
) -> Dict[str, Any]:
    payload = {
        "model_name": model_name,
        "transaction": transaction,
    }

    payload = json.loads(json.dumps(payload, default=str, allow_nan=False))

    response = requests.post(
        f"{API_URL}/predict",
        json=payload,
        timeout=30,
    )

    if not response.ok:
        raise RuntimeError(response.text)

    return response.json()


def render_prediction(result: Dict[str, Any]) -> None:
    probability = result["fraud_probability"]
    label = result["prediction_label"]
    risk_level = get_risk_level(probability)

    col1, col2, col3 = st.columns(3)

    col1.metric("Fraud probability", f"{probability:.4f}")
    col2.metric("Prediction", label)
    col3.metric("Model version", result["model_version"])

    if risk_level == "HIGH RISK":
        st.error("HIGH RISK: transaction requires manual review")
    elif risk_level == "MEDIUM RISK":
        st.warning("MEDIUM RISK: transaction looks suspicious")
    else:
        st.success("LOW RISK: transaction looks safe")

    st.progress(min(probability, 1.0))


def render_explanation(result: Dict[str, Any]) -> None:
    explanation = result.get("explanation", [])

    if not explanation:
        st.warning("No explanation returned")
        return

    explanation_df = pd.DataFrame(explanation)

    st.subheader("Top SHAP feature impacts")
    st.dataframe(explanation_df, use_container_width=True)

    chart_df = explanation_df[["feature", "shap_value"]].set_index("feature")
    st.bar_chart(chart_df)


def render_model_info() -> None:
    try:
        info = get_model_info()
    except Exception as error:
        st.error(f"Failed to load model info: {error}")
        return

    st.subheader("Available models")
    st.dataframe(pd.DataFrame(info["available_models"]), use_container_width=True)

    st.subheader("Model metadata")
    st.json(info.get("metadata", {}))


def get_risk_level(probability: float) -> str:
    if probability >= 0.7:
        return "HIGH RISK"
    if probability >= 0.3:
        return "MEDIUM RISK"
    return "LOW RISK"


def find_top_risky_transactions(
    df: pd.DataFrame,
    model_name: str,
    limit: int = 20,
) -> pd.DataFrame:
    rows = []

    sample_df = df.head(100)

    for idx, row in sample_df.iterrows():
        transaction = (
            row.replace({pd.NA: None})
            .where(pd.notnull(row), None)
            .to_dict()
        )

        try:
            result = predict_transaction(transaction, model_name)

            rows.append(
                {
                    "row_index": idx,
                    "fraud_probability": result["fraud_probability"],
                    "prediction_label": result["prediction_label"],
                }
            )
        except Exception:
            continue

    result_df = pd.DataFrame(rows)

    if result_df.empty:
        return result_df

    return result_df.sort_values(
        "fraud_probability",
        ascending=False,
    ).head(limit)


def main() -> None:
    st.title("Fraud Detection Service")
    st.caption("IEEE-CIS Fraud Detection")

    tab_predict, tab_model = st.tabs(["Prediction", "Model info"])

    with tab_predict:
        st.sidebar.header("Settings")

        try:
            models_response = get_models()
            available_models = models_response["available_models"]
        except Exception as error:
            st.error(f"FastAPI service is unavailable: {error}")
            return

        model_options = {
            model["display_name"]: model["name"]
            for model in available_models
        }

        selected_display_name = st.sidebar.selectbox(
            "Model",
            options=list(model_options.keys()),
        )

        selected_model_name = model_options[selected_display_name]

        uploaded_file = st.file_uploader(
            "Upload transactions CSV",
            type=["csv"],
        )

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        else:
            df = load_sample_data()

        st.subheader("Transactions")
        st.dataframe(df.head(100), use_container_width=True)
        st.subheader("Top risky transactions")

        if st.button("Calculate top risky transactions"):
            top_risky_df = find_top_risky_transactions(
                df=df,
                model_name=selected_model_name,
                limit=10,
            )

            if top_risky_df.empty:
                st.warning("Failed to calculate top risky transactions")
            else:
                st.dataframe(top_risky_df, use_container_width=True)
                st.bar_chart(
                    top_risky_df.set_index("row_index")[["fraud_probability"]]
                )

        max_index = len(df) - 1

        selected_index = st.number_input(
            "Select transaction row index",
            min_value=0,
            max_value=max_index,
            value=0,
            step=1,
        )

        selected_transaction = (
            df.iloc[int(selected_index)]
            .replace({pd.NA: None})
            .where(pd.notnull(df.iloc[int(selected_index)]), None)
            .to_dict()
        )

        with st.expander("Selected transaction JSON"):
            st.json(json.loads(json.dumps(selected_transaction, default=str)))

        if st.button("Predict fraud probability", type="primary"):
            try:
                result = predict_transaction(
                    transaction=selected_transaction,
                    model_name=selected_model_name,
                )

                render_prediction(result)
                render_explanation(result)

            except Exception as error:
                st.error(f"Prediction failed: {error}")

    with tab_model:
        render_model_info()


if __name__ == "__main__":
    main()