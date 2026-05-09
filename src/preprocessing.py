import json
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.config import FEATURE_ORDER_PATH, PREPROCESSING_STATS_PATH


def load_json(path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_feature_order() -> List[str]:
    return load_json(FEATURE_ORDER_PATH)


def load_preprocessing_stats() -> Dict[str, Any]:
    return load_json(PREPROCESSING_STATS_PATH)


def simple_label_encode(series: pd.Series) -> pd.Series:
    series = series.astype(str)
    codes, _ = pd.factorize(series, sort=True)
    return pd.Series(codes, index=series.index)


def add_missing_columns(df: pd.DataFrame, feature_order: List[str]) -> pd.DataFrame:
    for col in feature_order:
        if col not in df.columns:
            df[col] = np.nan

    return df


def drop_unused_columns(df: pd.DataFrame, feature_order: List[str]) -> pd.DataFrame:
    keep_cols = [col for col in df.columns if col in feature_order]
    return df[keep_cols]


def fill_missing_values(
    df: pd.DataFrame,
    preprocessing_stats: Dict[str, Any],
) -> pd.DataFrame:
    numeric_fill_values = preprocessing_stats.get("numeric_fill_values", {})
    categorical_fill_values = preprocessing_stats.get("categorical_fill_values", {})

    for col, value in numeric_fill_values.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(value)

    for col, value in categorical_fill_values.items():
        if col in df.columns:
            df[col] = df[col].astype("object")
            df[col] = df[col].fillna(value)

    return df


def encode_categorical_columns(
    df: pd.DataFrame,
    preprocessing_stats: Dict[str, Any],
) -> pd.DataFrame:
    categorical_cols = preprocessing_stats.get("categorical_columns", [])

    for col in categorical_cols:
        if col in df.columns:
            df[col] = simple_label_encode(df[col])

    return df


def ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


def preprocess_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Главная функция preprocessing для inference.

    Input:
        raw merged IEEE-CIS dataframe или одна строка в виде dataframe.

    Output:
        dataframe с колонками в порядке feature_order.
    """
    feature_order = load_feature_order()
    preprocessing_stats = load_preprocessing_stats()

    df = df.copy()

    df = df.drop(columns=["isFraud"], errors="ignore")

    df = add_missing_columns(df, feature_order)
    df = drop_unused_columns(df, feature_order)

    df = fill_missing_values(df, preprocessing_stats)
    df = encode_categorical_columns(df, preprocessing_stats)

    df = ensure_numeric(df)

    df = df[feature_order]

    return df