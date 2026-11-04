
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib

from pathlib import Path

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

from src.features.build_features import build_preprocessor


def train_models():

    train_path = Path("data/processed/train.csv")
    test_path = Path("data/processed/test.csv")

    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")

    if not test_path.exists():
        raise FileNotFoundError(f"Testing data not found: {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if "Churn" not in train_df.columns or "Churn" not in test_df.columns:
        raise ValueError("Churn target column is missing!")

    X_train = train_df.drop(columns=["Churn"])
    y_train = train_df["Churn"].astype(int)

    X_test = test_df.drop(columns=["Churn"])
    y_test = test_df["Churn"].astype(int)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),

        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    }

    # Configure MLflow using SQLite
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Customer-Churn-Prediction")

    best_f1 = -1
    best_model = None
    best_model_name = ""

    results = []

    for name, classifier in models.items():

        print(f"\nTraining {name}...")

        pipeline = Pipeline([
            ("preprocessor", build_preprocessor(X_train)),
            ("classifier", classifier)
        ])

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, predictions)

        precision = precision_score(
            y_test, predictions, zero_division=0
        )

        recall = recall_score(
            y_test, predictions, zero_division=0
        )

        f1 = f1_score(
            y_test, predictions, zero_division=0
        )

        roc_auc = roc_auc_score(y_test, probabilities)

        # Log parameters, metrics and model in MLflow
        with mlflow.start_run(run_name=name):

            mlflow.log_param("model", name)

            mlflow.log_metrics({
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "roc_auc": roc_auc
            })

            mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="model",
                serialization_format="cloudpickle"
            )

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "ROC AUC": roc_auc
        })

        print(f"\nModel: {name}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")

        # Select the best model using F1 score
        if f1 > best_f1:
            best_f1 = f1
            best_model = pipeline
            best_model_name = name

    # Save the best complete pipeline
    output_dir = Path("artifacts/model")
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, output_dir / "model.joblib")

    # Save model comparison results
    results_df = pd.DataFrame(results)

    results_df.to_csv(
        output_dir / "model_comparison.csv",
        index=False
    )

    print("\n========== TRAINING COMPLETED ==========")

    print("\nModel Comparison:")
    print(results_df.to_string(index=False))

    print(f"\nBest Model: {best_model_name}")
    print(f"Best F1 Score: {best_f1:.4f}")

    print("\nBest model saved successfully!")
    print(f"Model location: {output_dir / 'model.joblib'}")
    print("MLflow database: mlflow.db")


if __name__ == "__main__":
    train_models()
