import streamlit as st
import pandas as pd
import joblib

# Load model and features
model = joblib.load("logistic_churn_model.pkl")
model_features = joblib.load("model_features.pkl")

st.title("📊 Customer Churn Prediction App (Logistic Regression)")

# Collect user inputs (key features)
gender = st.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.selectbox("Partner", ["No", "Yes"])
dependents = st.selectbox("Dependents", ["No", "Yes"])
tenure = st.number_input("Tenure (months)", min_value=0, max_value=72)
phone_service = st.selectbox("Phone Service", ["No", "Yes"])
multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["No", "Yes"])
online_backup = st.selectbox("Online Backup", ["No", "Yes"])
tech_support = st.selectbox("Tech Support", ["No", "Yes"])
streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
payment_method = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])
monthly_charges = st.number_input("Monthly Charges", min_value=0.0)
total_charges = st.number_input("Total Charges", min_value=0.0)

# Convert user input to dataframe
input_dict = {
    "tenure": [tenure],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges],
    "gender_Male": [1 if gender == "Male" else 0],
    "SeniorCitizen": [1 if senior_citizen == "Yes" else 0],
    "Partner_Yes": [1 if partner == "Yes" else 0],
    "Dependents_Yes": [1 if dependents == "Yes" else 0],
    "PhoneService_Yes": [1 if phone_service == "Yes" else 0],
    "MultipleLines_Yes": [1 if multiple_lines == "Yes" else 0],
    "InternetService_Fiber optic": [1 if internet_service == "Fiber optic" else 0],
    "InternetService_No": [1 if internet_service == "No" else 0],
    "OnlineSecurity_Yes": [1 if online_security == "Yes" else 0],
    "OnlineBackup_Yes": [1 if online_backup == "Yes" else 0],
    "TechSupport_Yes": [1 if tech_support == "Yes" else 0],
    "StreamingTV_Yes": [1 if streaming_tv == "Yes" else 0],
    "StreamingMovies_Yes": [1 if streaming_movies == "Yes" else 0],
    "Contract_One year": [1 if contract == "One year" else 0],
    "Contract_Two year": [1 if contract == "Two year" else 0],
    "PaperlessBilling_Yes": [1 if paperless_billing == "Yes" else 0],
    "PaymentMethod_Credit card (automatic)": [1 if payment_method == "Credit card (automatic)" else 0],
    "PaymentMethod_Electronic check": [1 if payment_method == "Electronic check" else 0],
    "PaymentMethod_Mailed check": [1 if payment_method == "Mailed check" else 0]
}

input_df = pd.DataFrame(input_dict)

# Align input with training features
input_df = input_df.reindex(columns=model_features, fill_value=0)

# Prediction
if st.button("Predict"):
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]
    st.write("### 🔮 Prediction:", "Churn" if pred == 1 else "No Churn")
    st.write(f"### 📈 Probability of Churn: {prob*100:.2f}%")
