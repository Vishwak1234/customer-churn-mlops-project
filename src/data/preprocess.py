
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


def preprocess_data():

    input_path = Path("data/raw/dataset.csv")
    output_dir = Path("data/processed")

    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Dataset not found: {input_path}")

    if input_path.stat().st_size == 0:
        raise ValueError("Dataset file is empty!")

    # Load tab-separated dataset
    df = pd.read_csv(input_path, sep="\t", encoding="utf-8-sig")

    # Clean column names
    df.columns = df.columns.str.strip()

    print("Dataset loaded successfully!")
    print("Original shape:", df.shape)
    print("Available columns:", df.columns.tolist())

    # Validate dataset
    required_columns = ["customerID", "TotalCharges", "Churn"]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column missing: {col}")

    # Remove duplicate records
    df.drop_duplicates(inplace=True)

    # Remove customer ID
    df.drop(columns=["customerID"], inplace=True)

    # Convert TotalCharges into numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].astype(str).str.strip(),
        errors="coerce"
    )

    # Replace empty values with NaN
    df.replace(r"^\s*$", np.nan, regex=True, inplace=True)

    # Convert TotalCharges again after cleaning
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    )

    # Handle missing target values
    df.dropna(subset=["Churn"], inplace=True)

    # Separate features and target
    X = df.drop(columns=["Churn"])
    y = df["Churn"].map({"Yes": 1, "No": 0})

    if y.isnull().any():
        raise ValueError("Invalid values found in Churn column!")

    # Split training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Combine features and target
    train_df = X_train.copy()
    train_df["Churn"] = y_train

    test_df = X_test.copy()
    test_df["Churn"] = y_test

    # Save processed datasets
    train_df.to_csv(output_dir / "train.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    print("\nPreprocessing completed successfully!")
    print("Training data shape:", train_df.shape)
    print("Testing data shape:", test_df.shape)

    print("\nMissing values in training data:")
    print(train_df.isnull().sum())

    print("\nFiles saved successfully!")


if __name__ == "__main__":
    preprocess_data()
