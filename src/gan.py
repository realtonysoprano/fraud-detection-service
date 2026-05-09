import json
from functools import lru_cache
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.config import GAN_DISCRIMINATOR_PATH, GAN_FEATURE_ORDER_PATH, GAN_SCALER_STATS_PATH


class Discriminator(nn.Module):
    def __init__(self, input_dim: int = 112):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


@lru_cache(maxsize=1)
def load_json(path) -> Dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


@lru_cache(maxsize=1)
def load_gan_feature_order() -> List[str]:
    return load_json(GAN_FEATURE_ORDER_PATH)


@lru_cache(maxsize=1)
def load_gan_scaler_stats() -> Dict:
    return load_json(GAN_SCALER_STATS_PATH)


@lru_cache(maxsize=1)
def load_discriminator() -> Discriminator:
    stats = load_gan_scaler_stats()
    input_dim = int(stats.get("input_dim", 112))

    model = Discriminator(input_dim=input_dim)
    state_dict = torch.load(
        GAN_DISCRIMINATOR_PATH,
        map_location=torch.device("cpu"),
    )

    model.load_state_dict(state_dict)
    model.eval()

    return model


def prepare_gan_features(features: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    gan_feature_order = load_gan_feature_order()

    result = features.copy()

    for col in gan_feature_order:
        if col not in result.columns:
            if col in raw_df.columns:
                result[col] = raw_df[col].values
            else:
                result[col] = 0

    result = result[gan_feature_order]
    result = result.apply(pd.to_numeric, errors="coerce").fillna(0)

    return result


def apply_minmax_scaler(features: pd.DataFrame) -> np.ndarray:
    stats = load_gan_scaler_stats()

    min_values = np.array(stats["min"], dtype=np.float32)
    scale_values = np.array(stats["scale"], dtype=np.float32)

    values = features.to_numpy(dtype=np.float32)
    scaled = values * scale_values + min_values

    return np.clip(scaled, 0.0, 1.0)


def calculate_gan_anomaly_score(
    features: pd.DataFrame,
    raw_df: pd.DataFrame,
) -> pd.Series:
    gan_features = prepare_gan_features(features, raw_df)
    discriminator = load_discriminator()
    scaled_features = apply_minmax_scaler(gan_features)

    tensor = torch.FloatTensor(scaled_features)

    with torch.no_grad():
        probabilities = discriminator(tensor).cpu().numpy().flatten()

    anomaly_score = 1.0 - probabilities

    return pd.Series(anomaly_score, index=features.index).clip(0.0, 1.0)