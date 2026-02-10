

%%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Fraud Detection App",
    layout="centered"
)

# --------------------------------------------------
# Title & description
# --------------------------------------------------
st.title("💳 Fraud Detection System")
st.write(
    """
    This application predicts whether a transaction is **fraudulent** or **legitimate**
    using a trained machine learning model.
    """
)

# --------------------------------------------------
# Load model (safe loading)
# --------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("fraud_model.pkl")

try:
    model = load_model()
    st.success("Model loaded successfully.")
except Exception as e:
    st.error("❌ Model could not be loaded.")
    st.stop()

# --------------------------------------------------
# User input section
# --------------------------------------------------
st.subheader("Enter Transaction Details")

# ⚠️ IMPORTANT:
# These inputs MUST match the features used when training your model.
# Adjust names/order if your notebook used different columns.

amount = st.number_input("Transaction Amount", min_value=0.0, value=100.0)
time = st.number_input("Transaction Time", min_value=0.0, value=0.0)

# Example placeholder features (edit if your model uses different ones)
v1 = st.number_input("V1", value=0.0)
v2 = st.number_input("V2", value=0.0)
v3 = st.number_input("V3", value=0.0)

# --------------------------------------------------
# Create input dataframe (order matters!)
# --------------------------------------------------
input_data = pd.DataFrame(
    [[amount, time, v1, v2, v3]],
    columns=["Amount", "Time", "V1", "V2", "V3"]
)

# --------------------------------------------------
# Prediction button
# --------------------------------------------------
if st.button("🔍 Detect Fraud"):
    try:
        prediction = model.predict(input_data)[0]

        # Probability (if model supports it)
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_data)[0][1]
        else:
            probability = None

        if prediction == 1:
            st.error("⚠️ Fraudulent Transaction Detected")
        else:
            st.success("✅ Legitimate Transaction")

        if probability is not None:
            st.write(f"**Fraud Probability:** {probability:.2f}")

    except Exception as e:
        st.error("❌ Prediction failed.")
        st.write("Error details:", e)

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption("Fraud Detection App • Streamlit Cloud Deployment")

