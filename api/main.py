
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
from pathlib import Path
import joblib
import pandas as pd


app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn using Machine Learning",
    version="1.0.0"
)


MODEL_PATH = Path("artifacts/model/model.joblib")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)


class CustomerData(BaseModel):

    gender: Literal["Male", "Female"]
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, le=100)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


@app.get("/")
def home():

    return {
        "message": "Customer Churn Prediction API is running"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True
    }


@app.get("/model-info")
def model_info():

    return {
        "model_name": "Logistic Regression",
        "model_type": "Classification",
        "selection_metric": "F1 Score",
        "f1_score": 0.6136,
        "roc_auc": 0.8413
    }


@app.post("/predict")
def predict_churn(customer: CustomerData):

    try:

        customer_data = pd.DataFrame(
            [customer.model_dump()]
        )

        prediction = int(model.predict(customer_data)[0])

        probability = float(
            model.predict_proba(customer_data)[0][1]
        )

        result = (
            "Churn" if prediction == 1
            else "No Churn"
        )

        return {
            "prediction": result,
            "churn_probability": round(probability, 4),
            "message": (
                "Customer is likely to churn"
                if prediction == 1
                else "Customer is unlikely to churn"
            )
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )
