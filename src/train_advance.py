import json
import re
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from mlflow.models import infer_signature
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from data_preprocessing import build_preprocessor


RANDOM_STATE = 42
TARGET = "Diabetes_binary"
EXPERIMENT_NAME = "diabetes-risk-advanced-models"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "diabetes_no_duplicates.csv"
BASELINE_RESULTS_PATH = PROJECT_ROOT / "reports" / "baseline_model_results.csv"
ADVANCED_RESULTS_PATH = PROJECT_ROOT / "reports" / "advanced_model_results.csv"
ALL_RESULTS_PATH = PROJECT_ROOT / "reports" / "all_model_comparison.csv"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures" / "day4"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"


def slugify(value: str) -> str:
    """Convert a model name into a safe filename."""
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def load_and_split_data():
    """Load the processed dataset and create the Day 3-compatible split."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' is missing from the dataset.")

    df[TARGET] = df[TARGET].astype(int)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


def calculate_metrics(y_true, y_pred, y_proba) -> dict[str, float]:
    """Calculate the same classification metrics used for baseline models."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def save_evaluation_artifacts(model_name: str, y_true, y_pred, metrics: dict[str, float]):
    """Save confusion matrix, classification report, and metrics files."""
    model_slug = slugify(model_name)

    cm_path = FIGURES_DIR / f"{model_slug}_confusion_matrix.png"
    report_path = FIGURES_DIR / f"{model_slug}_classification_report.json"
    metrics_path = FIGURES_DIR / f"{model_slug}_metrics.json"

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Non-Diabetic", "Diabetic"],
    )
    display.plot(ax=ax, values_format="d")
    ax.set_title(f"{model_name} - Confusion Matrix")
    fig.tight_layout()
    fig.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Non-Diabetic", "Diabetic"],
        output_dict=True,
        zero_division=0,
    )

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    return cm_path, report_path, metrics_path


def build_model_configs(scale_pos_weight: float, gradient_boosting_weights):
    """Create the four Day 4 advanced-model configurations."""
    return {
        "Gradient Boosting": {
            "model": GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.80,
                random_state=RANDOM_STATE,
            ),
            "fit_params": {
                "model__sample_weight": gradient_boosting_weights,
            },
            "log_params": {
                "n_estimators": 150,
                "learning_rate": 0.05,
                "max_depth": 3,
                "subsample": 0.80,
                "random_state": RANDOM_STATE,
                "imbalance_strategy": "balanced_sample_weight",
            },
        },
        "XGBoost": {
            "model": XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=5,
                min_child_weight=1,
                subsample=0.80,
                colsample_bytree=0.80,
                reg_lambda=1.0,
                scale_pos_weight=scale_pos_weight,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "fit_params": {},
            "log_params": {
                "n_estimators": 300,
                "learning_rate": 0.05,
                "max_depth": 5,
                "min_child_weight": 1,
                "subsample": 0.80,
                "colsample_bytree": 0.80,
                "reg_lambda": 1.0,
                "scale_pos_weight": scale_pos_weight,
                "tree_method": "hist",
                "random_state": RANDOM_STATE,
            },
        },
        "LightGBM": {
            "model": LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=31,
                max_depth=-1,
                subsample=0.80,
                subsample_freq=1,
                colsample_bytree=0.80,
                reg_lambda=1.0,
                scale_pos_weight=scale_pos_weight,
                objective="binary",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbosity=-1,
            ),
            "fit_params": {},
            "log_params": {
                "n_estimators": 300,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "max_depth": -1,
                "subsample": 0.80,
                "subsample_freq": 1,
                "colsample_bytree": 0.80,
                "reg_lambda": 1.0,
                "scale_pos_weight": scale_pos_weight,
                "random_state": RANDOM_STATE,
            },
        },
        "CatBoost": {
            "model": CatBoostClassifier(
                iterations=300,
                learning_rate=0.05,
                depth=6,
                loss_function="Logloss",
                eval_metric="AUC",
                auto_class_weights="Balanced",
                random_seed=RANDOM_STATE,
                verbose=0,
                allow_writing_files=False,
                thread_count=-1,
            ),
            "fit_params": {},
            "log_params": {
                "iterations": 300,
                "learning_rate": 0.05,
                "depth": 6,
                "loss_function": "Logloss",
                "eval_metric": "AUC",
                "auto_class_weights": "Balanced",
                "random_seed": RANDOM_STATE,
            },
        },
    }


def train_and_log_model(
    model_name: str,
    config: dict[str, Any],
    X_train,
    X_test,
    y_train,
    y_test,
) -> dict[str, Any]:
    """Train one pipeline, evaluate it, and record the run in MLflow."""
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", config["model"]),
        ]
    )

    with mlflow.start_run(run_name=model_name) as run:
        mlflow.set_tags(
            {
                "project": "diabetes-risk-prediction",
                "stage": "advanced-model-training",
                "dataset": "BRFSS-2015",
                "model_name": model_name,
            }
        )

        mlflow.log_params(config["log_params"])
        mlflow.log_params(
            {
                "train_rows": len(X_train),
                "test_rows": len(X_test),
                "feature_count": X_train.shape[1],
                "test_size": 0.20,
                "positive_class_threshold": 0.50,
            }
        )

        start_time = time.perf_counter()
        pipeline.fit(X_train, y_train, **config["fit_params"])
        training_time = time.perf_counter() - start_time

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        metrics = calculate_metrics(y_test, y_pred, y_proba)
        metrics["training_time_seconds"] = float(training_time)

        mlflow.log_metrics(metrics)

        cm_path, report_path, metrics_path = save_evaluation_artifacts(
            model_name=model_name,
            y_true=y_test,
            y_pred=y_pred,
            metrics=metrics,
        )

        mlflow.log_artifact(str(cm_path), artifact_path="evaluation")
        mlflow.log_artifact(str(report_path), artifact_path="evaluation")
        mlflow.log_artifact(str(metrics_path), artifact_path="evaluation")

        input_example = X_test.head(5).copy()
        signature = infer_signature(input_example, pipeline.predict(input_example))

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            signature=signature,
            input_example=input_example,
        )

        result = {
            "model": model_name,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "roc_auc": metrics["roc_auc"],
            "training_time_seconds": metrics["training_time_seconds"],
            "mlflow_run_id": run.info.run_id,
        }

    print(f"\nFinished training: {model_name}")
    print(pd.Series(result))

    return result


def main():
    ADVANCED_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test = load_and_split_data()

    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())

    if positive_count == 0:
        raise ValueError("Training data contains no positive-class samples.")

    scale_pos_weight = float(negative_count / positive_count)
    gradient_boosting_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_train,
    )

    print(f"Training rows: {len(X_train):,}")
    print(f"Testing rows: {len(X_test):,}")
    print(f"Negative training samples: {negative_count:,}")
    print(f"Positive training samples: {positive_count:,}")
    print(f"Scale positive weight: {scale_pos_weight:.4f}")

    model_configs = build_model_configs(
        scale_pos_weight=scale_pos_weight,
        gradient_boosting_weights=gradient_boosting_weights,
    )

    results = []

    for model_name, config in model_configs.items():
        result = train_and_log_model(
            model_name=model_name,
            config=config,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )
        results.append(result)

    advanced_results = pd.DataFrame(results).sort_values(
        by=["recall", "roc_auc", "f1_score"],
        ascending=False,
    )
    advanced_results.to_csv(ADVANCED_RESULTS_PATH, index=False)

    print("\nAdvanced model results:")
    print(advanced_results.to_string(index=False))
    print(f"\nSaved advanced results to: {ADVANCED_RESULTS_PATH}")

    if BASELINE_RESULTS_PATH.exists():
        baseline_results = pd.read_csv(BASELINE_RESULTS_PATH)
        baseline_results["model_group"] = "baseline"
        baseline_results["training_time_seconds"] = pd.NA
        baseline_results["mlflow_run_id"] = pd.NA

        advanced_for_comparison = advanced_results.copy()
        advanced_for_comparison["model_group"] = "advanced"

        all_results = pd.concat(
            [baseline_results, advanced_for_comparison],
            ignore_index=True,
            sort=False,
        )

        comparison_columns = [
            "model",
            "model_group",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "roc_auc",
            "training_time_seconds",
            "mlflow_run_id",
        ]

        all_results = all_results[comparison_columns].sort_values(
            by=["recall", "roc_auc", "f1_score"],
            ascending=False,
        )
        all_results.to_csv(ALL_RESULTS_PATH, index=False)

        print("\nBaseline and advanced model comparison:")
        print(all_results.to_string(index=False))
        print(f"\nSaved full comparison to: {ALL_RESULTS_PATH}")
    else:
        print(f"\nBaseline results file not found: {BASELINE_RESULTS_PATH}")


if __name__ == "__main__":
    main()
