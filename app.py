import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ----------------------------------------------------------------------
# Page Config
# ----------------------------------------------------------------------
st.set_page_config(page_title="Flight Delay Predictor", layout="centered")

# ----------------------------------------------------------------------
# Load Model Artifacts
# ----------------------------------------------------------------------
model = joblib.load("logistic_model.pkl")
scaler = joblib.load("scaler.pkl")
model_columns = joblib.load("model_columns.pkl")

# Extract valid dropdown options directly from the trained model's columns.
# This guarantees the dropdowns always match exactly what the model was trained on.
carrier_options = sorted([c.replace("Operating Carrier_", "") for c in model_columns if c.startswith("Operating Carrier_")])
origin_options = sorted([c.replace("ORIGIN_", "") for c in model_columns if c.startswith("ORIGIN_")])
dest_options = sorted([c.replace("DEST_", "") for c in model_columns if c.startswith("DEST_")])

st.title("✈️ Flight Arrival Delay Predictor")
st.write(
    "Predict whether a flight will arrive **15 or more minutes late**, "
    "based on information available around departure."
)

tab1, tab2 = st.tabs(["🔮 Prediction", "📊 Insights"])

# ----------------------------------------------------------------------
# TAB 1: Prediction
# ----------------------------------------------------------------------
with tab1:
    st.subheader("Flight Details")

    col1, col2 = st.columns(2)

    with col1:
        carrier = st.selectbox("Operating Carrier", carrier_options)
        origin = st.selectbox("Origin Airport", origin_options)
        dest = st.selectbox("Destination Airport", dest_options)

    with col2:
        distance = st.number_input("Distance (miles)", min_value=1, max_value=5000, value=800)
        departure_hour = st.slider("Scheduled Departure Hour", 0, 23, 12)
        dep_delay = st.number_input(
            "Departure Delay so far (minutes)",
            min_value=-60, max_value=500, value=0,
            help="Enter 0 if the flight has not departed yet, or the current departure delay if known."
        )

    predict_btn = st.button("Predict Delay", type="primary")

    if predict_btn:
        # Build a single-row dataframe with all model columns, initialized to 0
        input_df = pd.DataFrame(0, index=[0], columns=model_columns)

        # Numeric / derived features
        input_df["DEP_DELAY"] = dep_delay
        input_df["Departure Delay ≥ 15 Minutes"] = 1 if dep_delay >= 15 else 0
        input_df["DISTANCE"] = distance
        input_df["Departure Hour"] = departure_hour

        # One-hot categorical features
        carrier_col = f"Operating Carrier_{carrier}"
        origin_col = f"ORIGIN_{origin}"
        dest_col = f"DEST_{dest}"

        if carrier_col in input_df.columns:
            input_df[carrier_col] = 1
        if origin_col in input_df.columns:
            input_df[origin_col] = 1
        if dest_col in input_df.columns:
            input_df[dest_col] = 1

        # Scale exactly as done during training
        input_scaled = scaler.transform(input_df)

        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        st.divider()
        if prediction == 1:
            st.error(f"⚠️ Likely to be DELAYED (15+ minutes)")
        else:
            st.success(f"✅ Likely to arrive ON TIME (< 15 minutes late)")

        st.metric("Delay Probability", f"{probability * 100:.1f}%")
        st.progress(min(float(probability), 1.0))

# ----------------------------------------------------------------------
# TAB 2: Insights (reusing Phase 4/5 EDA findings)
# ----------------------------------------------------------------------
with tab2:
    st.subheader("Key Findings from Exploratory Data Analysis")

    try:
        delay_by_carrier = pd.read_csv("delay_rate_by_carrier.csv", index_col=0)
        st.write("**Arrival Delay Rate by Carrier (%)**")
        st.bar_chart(delay_by_carrier)
    except FileNotFoundError:
        st.info("delay_rate_by_carrier.csv not found. Export it from the modeling notebook.")

    try:
        delay_by_hour = pd.read_csv("delay_rate_by_hour.csv", index_col=0)
        st.write("**Arrival Delay Rate by Departure Hour (%)**")
        st.line_chart(delay_by_hour)
    except FileNotFoundError:
        st.info("delay_rate_by_hour.csv not found. Export it from the modeling notebook.")

    st.divider()
    st.subheader("Model Performance")
    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
        "Value": [0.9219, 0.8903, 0.7110, 0.7906]
    })
    st.table(metrics_df)
    st.caption("Logistic Regression — evaluated on a held-out 20% test set.")
