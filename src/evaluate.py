import argparse
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from config import (
    FEATURE_COLUMNS,
    FIGURES_DIR,
    FINAL_MODEL_CLASSIFICATION_REPORT_PATH,
    FINAL_MODEL_METRICS_PATH,
    FINAL_MODEL_TEST_PREDICTIONS_PATH,
    PREDICTION_THRESHOLD,
    PROCESSED_DATA_PATH,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
)
from predict import load_model, load_model_metadata


FINAL_EVALUATION_FIGURES_DIR = FIGURES_DIR / "final_model"


def load_test_data():
    """Load processed data and recreate the same test split used during training."""
    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(f"Processed dataset not found: {PROCESSED_DATA_PATH}")

    df = pd.read_csv(PROCESSED_DATA_PATH)

    missing_columns = [column for column in [TARGET, *FEATURE_COLUMNS] if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    X = df[FEATURE_COLUMNS]
    y = df[TARGET].astype(int)

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    return X_test, y_test


def calculate_metrics(y_true, y_pred, y_proba) -> dict[str, float]:
    """Calculate final model evaluation metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
    }


def save_confusion_matrix(y_true, y_pred):
    """Save final model confusion matrix as a PNG."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(6, 5))
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Non-Diabetic", "Diabetic"],
    )
    display.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title("Final Model - Confusion Matrix")
    fig.tight_layout()

    output_path = FINAL_EVALUATION_FIGURES_DIR / "confusion_matrix.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path


def save_roc_curve(y_true, y_proba):
    """Save final model ROC curve as a PNG."""
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax)
    ax.set_title("Final Model - ROC Curve")
    fig.tight_layout()

    output_path = FINAL_EVALUATION_FIGURES_DIR / "roc_curve.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path


def save_precision_recall_curve(y_true, y_proba):
    """Save final model precision-recall curve as a PNG."""
    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y_true, y_proba, ax=ax)
    ax.set_title("Final Model - Precision-Recall Curve")
    fig.tight_layout()

    output_path = FINAL_EVALUATION_FIGURES_DIR / "precision_recall_curve.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path


def evaluate_final_model(threshold: float = PREDICTION_THRESHOLD) -> dict[str, float]:
    """Evaluate the saved final model on the held-out test split."""
    FINAL_EVALUATION_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_MODEL_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    model = load_model()
    metadata = load_model_metadata()
    X_test, y_test = load_test_data()

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    metrics = calculate_metrics(y_test, y_pred, y_proba)
    metrics["threshold"] = float(threshold)
    metrics["test_rows"] = int(len(X_test))

    if metadata:
        metrics["model"] = metadata.get("model", "unknown")
        metrics["mlflow_run_id"] = metadata.get("mlflow_run_id")

    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Non-Diabetic", "Diabetic"],
        output_dict=True,
        zero_division=0,
    )

    with FINAL_MODEL_METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    with FINAL_MODEL_CLASSIFICATION_REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    predictions = X_test.copy()
    predictions[TARGET] = y_test.to_numpy()
    predictions["diabetes_risk_probability"] = y_proba
    predictions["diabetes_risk_prediction"] = y_pred
    predictions.to_csv(FINAL_MODEL_TEST_PREDICTIONS_PATH, index=False)

    confusion_matrix_path = save_confusion_matrix(y_test, y_pred)
    roc_curve_path = save_roc_curve(y_test, y_proba)
    precision_recall_curve_path = save_precision_recall_curve(y_test, y_proba)

    print("\nFinal model evaluation")
    print(pd.Series(metrics).to_string())
    print(f"\nSaved metrics to: {FINAL_MODEL_METRICS_PATH}")
    print(f"Saved classification report to: {FINAL_MODEL_CLASSIFICATION_REPORT_PATH}")
    print(f"Saved test predictions to: {FINAL_MODEL_TEST_PREDICTIONS_PATH}")
    print(f"Saved confusion matrix to: {confusion_matrix_path}")
    print(f"Saved ROC curve to: {roc_curve_path}")
    print(f"Saved precision-recall curve to: {precision_recall_curve_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate the saved final diabetes-risk model.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=PREDICTION_THRESHOLD,
        help="Probability threshold for assigning the positive class.",
    )
    args = parser.parse_args()

    evaluate_final_model(threshold=args.threshold)


if __name__ == "__main__":
    main()
