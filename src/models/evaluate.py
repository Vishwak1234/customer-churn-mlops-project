
import pandas as pd
import joblib
import json

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


def evaluate_model():

    model_path = Path("artifacts/model/model.joblib")
    test_path = Path("data/processed/test.csv")

    if not model_path.exists():
        raise FileNotFoundError("Trained model not found!")

    if not test_path.exists():
        raise FileNotFoundError("Test dataset not found!")

    print("Loading saved model...")

    model = joblib.load(model_path)

    test_df = pd.read_csv(test_path)

    X_test = test_df.drop(columns=["Churn"])
    y_test = test_df["Churn"].astype(int)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_test, predictions, zero_division=0)
        ),
        "roc_auc": float(
            roc_auc_score(y_test, probabilities)
        )
    }

    print("\n========== MODEL EVALUATION ==========")

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        zero_division=0
    ))

    output_path = Path("artifacts/model/metrics.json")

    with open(output_path, "w") as file:
        json.dump(metrics, file, indent=4)

    print("\nEvaluation completed successfully!")
    print(f"Metrics saved at: {output_path}")


if __name__ == "__main__":
    evaluate_model()
