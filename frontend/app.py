
import streamlit as st
import requests
import os

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000") + "/predict"

st.title("Customer Churn Prediction")
st.write("Predict whether a customer is likely to leave the company.")

st.divider()

st.subheader("Enter Customer Details")

with st.form("customer_form"):

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.number_input("Tenure (months)", 0, 100, 12)
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multiple = st.selectbox(
            "Multiple Lines",
            ["Yes", "No", "No phone service"]
        )
        internet = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )
        security = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"]
        )
        backup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"]
        )

    with col2:
        protection = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"]
        )
        support = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"]
        )
        tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"]
        )
        movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"]
        )
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )
        paperless = st.selectbox(
            "Paperless Billing", ["Yes", "No"]
        )
        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )
        monthly = st.number_input(
            "Monthly Charges", min_value=0.0, value=75.5
        )
        total = st.number_input(
            "Total Charges", min_value=0.0, value=906.0
        )

    submitted = st.form_submit_button(
        "Predict Customer Churn",
        use_container_width=True
    )

if submitted:

    customer = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": protection,
        "TechSupport": support,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }

    try:
        with st.spinner("Predicting customer churn..."):

            response = requests.post(
                API_URL,
                json=customer,
                timeout=30
            )

        if response.status_code == 200:

            result = response.json()

            st.divider()
            st.subheader("Prediction Result")

            if result["prediction"] == "Churn":
                st.error("Customer is likely to churn.")
            else:
                st.success("Customer is unlikely to churn.")

            probability = result["churn_probability"]

            st.metric(
                "Churn Probability",
                f"{probability * 100:.2f}%"
            )

            st.progress(probability)

            st.info(result["message"])

        else:
            st.error(
                f"API Error {response.status_code}: "
                f"{response.text}"
            )

    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot connect to FastAPI. "
            "Please start the backend server first."
        )

    except requests.exceptions.RequestException as error:
        st.error(f"Request failed: {error}")
