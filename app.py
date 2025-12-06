import streamlit as st
import pickle
import numpy as np

# Load trained model and scaler
model = pickle.load(open("churn_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Customer Churn Prediction System")
st.write("Enter customer details to predict whether the customer will churn")

# User inputs
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", ["Yes", "No"])
tenure = st.number_input("Tenure (Months)", 0, 100)
monthly = st.number_input("Monthly Charges", 0.0, 200.0)
total = st.number_input("Total Charges", 0.0, 10000.0)
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment = st.selectbox("Payment Method", ["Credit Card", "Bank Transfer", "Electronic Check"])

# Encoding dictionary
encode = {
    "Male": 1, "Female": 0,
    "Yes": 1, "No": 0,
    "Month-to-month": 0, "One year": 1, "Two year": 2,
    "DSL": 0, "Fiber optic": 1, "No": 2,
    "Credit Card": 0, "Bank Transfer": 1, "Electronic Check": 2
}

# Prepare input data
input_data = np.array([
    encode[gender],
    encode[senior],
    tenure,
    monthly,
    total,
    encode[contract],
    encode[internet],
    encode[payment]
]).reshape(1, -1)

# Scale input
input_scaled = scaler.transform(input_data)

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1] * 100

    if prediction == 1:
        st.error(f"⚠️ Customer is likely to CHURN ({probability:.2f}%)")
    else:
        st.success(f"✅ Customer is NOT likely to churn ({probability:.2f}%)")
