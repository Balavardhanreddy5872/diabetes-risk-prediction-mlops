import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from data_preprocessing import build_preprocessor


RANDOM_STATE = 42
TARGET = "Diabetes_binary"


def evaluate_model(model_name, pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    return {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba)
    }


def main():
    data_path = Path("data/processed/diabetes_no_duplicates.csv")
    output_path = Path("reports/baseline_model_results.csv")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    }

    results = []

    for model_name, model in models.items():
        preprocessor = build_preprocessor()

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model)
            ]
        )

        pipeline.fit(X_train, y_train)

        model_results = evaluate_model(model_name, pipeline, X_test, y_test)
        results.append(model_results)

        print(f"Finished training: {model_name}")

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by=["recall", "roc_auc", "f1_score"],
        ascending=False
    )

    results_df.to_csv(output_path, index=False)

    print("\nBaseline model results:")
    print(results_df)
    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()