
import streamlit as st
import pandas as pd
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
# User input section (MATCHES TRAINING FEATURES)
# --------------------------------------------------
st.subheader("Enter Transaction Details")

bin_country = st.selectbox(
    "Country Risk Category",
    options=[0, 1],
    help="0 = Low-risk country, 1 = High-risk country"
)

email = st.selectbox(
    "Email Type",
    options=[0, 1],
    help="0 = Free email provider, 1 = Corporate email"
)

tx_type = st.selectbox(
    "Transaction Type",
    options=[0, 1],
    help="0 = Normal transaction, 1 = High-risk transaction type"
)

# --------------------------------------------------
# Create input dataframe (EXACT column names & order)
# --------------------------------------------------
input_data = pd.DataFrame({
    "type": [tx_type],
    "email": [email],
    "bin country": [bin_country]
})

# FORCE the training order
input_data = input_data[["type", "email", "bin country"]]

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
