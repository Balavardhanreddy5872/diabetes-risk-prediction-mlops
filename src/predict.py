import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from config import (
    FEATURE_COLUMNS,
    FINAL_MODEL_METADATA_PATH,
    FINAL_MODEL_PATH,
    PREDICTION_THRESHOLD,
)


def load_model(model_path: Path = FINAL_MODEL_PATH):
    """Load the trained final model pipeline."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Final model not found at {model_path}. "
            "Run `python src/train_advance.py` first to create it."
        )

    return joblib.load(model_path)


def load_model_metadata(metadata_path: Path = FINAL_MODEL_METADATA_PATH) -> dict[str, Any]:
    """Load metadata for the final model, if available."""
    if not metadata_path.exists():
        return {}

    with metadata_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def prepare_input(data: dict[str, Any] | pd.DataFrame) -> pd.DataFrame:
    """Validate and order input features for the trained pipeline."""
    if isinstance(data, dict):
        input_df = pd.DataFrame([data])
    else:
        input_df = data.copy()

    missing_columns = [column for column in FEATURE_COLUMNS if column not in input_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required feature columns: {missing_columns}")

    return input_df[FEATURE_COLUMNS]


def predict_risk(
    data: dict[str, Any] | pd.DataFrame,
    model_path: Path = FINAL_MODEL_PATH,
    threshold: float = PREDICTION_THRESHOLD,
) -> pd.DataFrame:
    """Return diabetes-risk predictions and probabilities."""
    model = load_model(model_path)
    input_df = prepare_input(data)

    risk_probability = model.predict_proba(input_df)[:, 1]
    prediction = (risk_probability >= threshold).astype(int)

    results = input_df.copy()
    results["diabetes_risk_probability"] = risk_probability
    results["diabetes_risk_prediction"] = prediction

    return results


def main():
    parser = argparse.ArgumentParser(description="Generate diabetes-risk predictions.")
    parser.add_argument("--input", required=True, help="Path to an input CSV file.")
    parser.add_argument("--output", help="Optional path to save predictions as CSV.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=PREDICTION_THRESHOLD,
        help="Probability threshold for assigning the positive class.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    input_df = pd.read_csv(input_path)
    predictions = predict_risk(input_df, threshold=args.threshold)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(output_path, index=False)
        print(f"Saved predictions to: {output_path}")
    else:
        print(predictions.to_string(index=False))


if __name__ == "__main__":
    main()
