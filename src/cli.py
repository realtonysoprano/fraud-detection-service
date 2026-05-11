import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import AVAILABLE_MODELS, MODEL_METADATA_PATH, THRESHOLD
from src.inference import load_model_metadata, predict_batch, predict_transaction


def print_json(data):
    print(json.dumps(data, indent=4, ensure_ascii=False))


def command_info(_args):
    metadata = load_model_metadata()

    result = {
        "service": "Fraud Detection Service",
        "threshold": THRESHOLD,
        "available_models": [
            {
                "name": name,
                "display_name": config["display_name"],
                "uses_anomaly_score": config["uses_anomaly_score"],
                "path": str(config["path"]),
            }
            for name, config in AVAILABLE_MODELS.items()
        ],
        "metadata": metadata,
        "metadata_path": str(MODEL_METADATA_PATH),
    }

    print_json(result)


def command_predict(args):
    input_path = Path(args.input)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    if args.row < 0 or args.row >= len(df):
        raise ValueError(f"Row index out of range: {args.row}")

    transaction = df.iloc[args.row].where(pd.notnull(df.iloc[args.row]), None).to_dict()
    result = predict_transaction(
        df=pd.DataFrame([transaction]),
        model_name=args.model,
    )

    print_json(result)


def command_batch_predict(args):
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    if args.limit is not None:
        df = df.head(args.limit)

    result = predict_batch(df=df, model_name=args.model)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    print_json(
        {
            "status": "ok",
            "model": args.model,
            "input": str(input_path),
            "output": str(output_path),
            "rows": len(result),
        }
    )


def build_parser():
    parser = argparse.ArgumentParser(
        prog="fraud-cli",
        description="CLI for Fraud Detection Service",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_info = subparsers.add_parser(
        "info",
        help="Show service and model information",
    )
    parser_info.set_defaults(func=command_info)

    parser_predict = subparsers.add_parser(
        "predict",
        help="Predict fraud probability for one transaction",
    )
    parser_predict.add_argument(
        "--input",
        required=True,
        help="Path to input CSV file",
    )
    parser_predict.add_argument(
        "--row",
        type=int,
        default=0,
        help="Row index for prediction",
    )
    parser_predict.add_argument(
        "--model",
        choices=list(AVAILABLE_MODELS.keys()),
        default="lgbm",
        help="Model name",
    )
    parser_predict.set_defaults(func=command_predict)

    parser_batch = subparsers.add_parser(
        "batch-predict",
        help="Run batch prediction for CSV file",
    )
    parser_batch.add_argument(
        "--input",
        required=True,
        help="Path to input CSV file",
    )
    parser_batch.add_argument(
        "--output",
        required=True,
        help="Path to output CSV file",
    )
    parser_batch.add_argument(
        "--model",
        choices=list(AVAILABLE_MODELS.keys()),
        default="lgbm",
        help="Model name",
    )
    parser_batch.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of rows to process",
    )
    parser_batch.set_defaults(func=command_batch_predict)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()