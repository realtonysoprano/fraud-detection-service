from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

SAMPLE_DATA_PATH = DATA_DIR / "sample_transactions.csv"

LGB_MODEL_PATH = ARTIFACTS_DIR / "lgb_model.txt"
LGBM_GAN_MODEL_PATH = ARTIFACTS_DIR / "lgbm_gan_aug.txt"

GAN_DISCRIMINATOR_PATH = ARTIFACTS_DIR / "gan_discriminator.pth"
GAN_SCALER_STATS_PATH = ARTIFACTS_DIR / "gan_scaler_stats.json"

FEATURE_ORDER_PATH = ARTIFACTS_DIR / "feature_order.json"
GAN_FEATURE_ORDER_PATH = ARTIFACTS_DIR / "gan_feature_order.json"
LGBM_GAN_FEATURE_ORDER_PATH = ARTIFACTS_DIR / "lgbm_gan_feature_order.json"
PREPROCESSING_STATS_PATH = ARTIFACTS_DIR / "preprocessing_stats.json"
MODEL_METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"

DEFAULT_MODEL_NAME = "lgbm"
THRESHOLD = 0.30

AVAILABLE_MODELS = {
    "lgbm": {
        "path": LGB_MODEL_PATH,
        "display_name": "LightGBM",
        "uses_anomaly_score": False,
    },
    "lgbm_gan": {
        "path": LGBM_GAN_MODEL_PATH,
        "display_name": "LightGBM + GAN anomaly score",
        "uses_anomaly_score": True,
    },
}

API_HOST = "0.0.0.0"
API_PORT = 8000
API_URL = "http://api:8000"