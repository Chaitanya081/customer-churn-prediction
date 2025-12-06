import os
import streamlit as st
import pickle
import numpy as np

st.title("Customer Churn Prediction System")

# ✅ Check if model files exist
if not os.path.exists("churn_model.pkl") or not os.path.exists("scaler.pkl"):
    st.error("❌ Model files not found. Please upload churn_model.pkl and scaler.pkl")
    st.stop()

# Load model and scaler
model = pickle.load(open("churn_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.write("Enter customer details to predict churn")

# User inputs
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", ["Yes", "No"])
tenure = st.number_input("Tenure (Months)", 0, 100)
monthly = st.number_input("Monthly Charges", 0.0, 200.0)
total = st.number_input("Total Charges", 0.0, 10000.0)
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment = st.selectbox("Payment Method", ["Credit Card", "Bank Transfer", "Electronic Check"])

encode = {
    "Male": 1, "Female": 0,
    "Yes": 1, "No": 0,
    "Month-to-month": 0, "One year": 1, "Two year": 2,
    "DSL": 0, "Fiber optic": 1, "No": 2,
    "Credit Card": 0, "Bank Transfer": 1, "Electronic Check": 2
}

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

input_scaled = scaler.transform(input_data)

if st.button("Predict"):
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1] * 100

    if pred == 1:
        st.error(f"⚠️ Customer is likely to CHURN ({prob:.2f}%)")
    else:
        st.success(f"✅ Customer will NOT churn ({prob:.2f}%)")
