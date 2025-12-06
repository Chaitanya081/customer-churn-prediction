import streamlit as st
import pickle
import numpy as np

# Load saved model and scaler
model = pickle.load(open("churn_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Customer Churn Prediction System")
st.write("Enter customer details to predict churn")

# User inputs
tenure = st.number_input("Tenure (Months)", 0, 100)
monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0)
total_charges = st.number_input("Total Charges", 0.0, 10000.0)

gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", ["Yes", "No"])
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment = st.selectbox("Payment Method", ["Credit Card", "Bank Transfer", "Electronic Check"])

# Encoding values
encode = {
    "Male": 1, "Female": 0,
    "Yes": 1, "No": 0,
    "Month-to-month": 0, "One year": 1, "Two year": 2,
    "DSL": 0, "Fiber optic": 1, "No": 2,
    "Credit Card": 0, "Bank Transfer": 1, "Electronic Check": 2
}

# Prepare input data
input_data = np.array([
    tenure,
    monthly_charges,
    total_charges,
    encode[gender],
    encode[senior],
    encode[contract],
    encode[internet],
    encode[payment]
]).reshape(1, -1)

# Scale input
input_scaled = scaler.transform(input_data)

# Prediction
if st.button("Predict Churn"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1] * 100

    if prediction == 1:
        st.error(f"⚠️ Customer is likely to CHURN ({probability:.2f}%)")
    else:
        st.success(f"✅ Customer is NOT likely to churn ({probability:.2f}%)")
