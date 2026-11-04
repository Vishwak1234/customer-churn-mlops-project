# AI-Assisted MLOps Customer Churn Prediction

## 1. Project Overview

This project predicts whether a telecom customer is likely to leave a company using Machine Learning. It implements an end-to-end MLOps workflow, including data preprocessing, model training, experiment tracking, model versioning, API development, frontend development, containerization and cloud deployment.

## 2. Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- MLflow
- DVC
- FastAPI
- Streamlit
- Docker and Docker Compose
- Git and GitHub
- GitHub Actions
- AWS EC2

## 3. Dataset

The project uses the IBM Telco Customer Churn dataset containing customer information, services, account details and churn status.

Target variable: `Churn`

## 4. Machine Learning Models

Three classification algorithms were trained and compared:

1. Logistic Regression
2. Decision Tree
3. Random Forest

The best model was selected based on F1-score.

## 5. Model Evaluation

| Metric | Logistic Regression |
|---|---:|
| Accuracy | 73.81% |
| Precision | 50.43% |
| Recall | 78.34% |
| F1-score | 61.36% |
| ROC-AUC | 84.13% |

## 6. MLOps Implementation

### MLflow
Used to track experiments, parameters, models and evaluation metrics.

### DVC
Used to manage the machine learning pipeline, including preprocessing, training and evaluation.

### FastAPI
Provides API endpoints for health checks, model information and customer churn predictions.

### Streamlit
Provides an interactive user interface for entering customer details and viewing predictions.

### Docker
Containerizes the backend and frontend services using Docker Compose.

### GitHub Actions
Automates testing and Docker image builds through a CI workflow.

## 7. Project Structure

```text
customer-churn-mlops/
├── api/
├── frontend/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── utils/
├── data/
│   ├── raw/
│   └── processed/
├── artifacts/
│   └── model/
├── tests/
├── notebooks/
├── .github/
│   └── workflows/
│       └── ci.yml
├── Dockerfile
├── Dockerfile.frontend
├── docker-compose.yml
├── requirements.txt
├── requirements-docker.txt
├── dvc.yaml
└── README.md
```

## 8. Run Locally

Install dependencies:

```bash
pip install -r requirements-docker.txt
```

Start the application:

```bash
docker compose up -d --build
```

Open the frontend:

```text
http://localhost:8501
```

FastAPI documentation (when running locally with backend port published):

```text
http://localhost:8000/docs
```

## 9. AWS Deployment

The application is deployed using an AWS EC2 Ubuntu instance.

Deployment URL:

```text
http://YOUR_EC2_PUBLIC_IP:8501
```

The Streamlit frontend is publicly accessible, while the FastAPI backend communicates internally through Docker networking.

## 10. Future Improvements

- Model performance optimization
- Automated model retraining
- Model monitoring and drift detection
- Enhanced deployment automation

## 11. Author

Customer Churn Prediction – MLOps Project