import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# ----------------------------------------------------------------------
# Page Config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Flight Delay Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ----------------------------------------------------------------------
# Custom Styling
# ----------------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1e3c72, #2a5298);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header {
        color: #6b7280;
        font-size: 1.05rem;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }
    div[data-testid="stMetric"] {
        background: #f8fafc;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 14px 16px;
    }
    .result-card-delay {
        background: linear-gradient(135deg, #fee2e2, #fecaca);
        border-left: 6px solid #dc2626;
        border-radius: 12px;
        padding: 22px 26px;
        margin-top: 10px;
    }
    .result-card-ontime {
        background: linear-gradient(135deg, #dcfce7, #bbf7d0);
        border-left: 6px solid #16a34a;
        border-radius: 12px;
        padding: 22px 26px;
        margin-top: 10px;
    }
    .result-title { font-size: 1.4rem; font-weight: 800; margin-bottom: 4px; }
    .result-sub { font-size: 0.95rem; color: #374151; }
    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #1e3c72;
        margin-top: 6px;
        margin-bottom: 8px;
        border-bottom: 2px solid #e5e7eb;
        padding-bottom: 6px;
    }
    footer, #MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Load Model Artifacts (cached so the app stays fast during a live demo)
# ----------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("logistic_model.pkl")
    scaler = joblib.load("scaler.pkl")
    model_columns = joblib.load("model_columns.pkl")
    return model, scaler, model_columns

model, scaler, model_columns = load_artifacts()

carrier_options = sorted([c.replace("Operating Carrier_", "") for c in model_columns if c.startswith("Operating Carrier_")])
origin_options = sorted([c.replace("ORIGIN_", "") for c in model_columns if c.startswith("ORIGIN_")])
dest_options = sorted([c.replace("DEST_", "") for c in model_columns if c.startswith("DEST_")])

# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.markdown('<p class="main-header">✈️ Flight Arrival Delay Predictor</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Predicting whether a flight will arrive 15+ minutes late, '
    'using information available around departure — Logistic Regression, F1-Score 0.79</p>',
    unsafe_allow_html=True
)

tab1, tab2, tab3 = st.tabs(["🔮  Live Prediction", "📊  Data Insights", "🧠  Model Performance"])

# ----------------------------------------------------------------------
# TAB 1: Prediction
# ----------------------------------------------------------------------
with tab1:
    st.markdown('<p class="section-title">Enter Flight Details</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        carrier = st.selectbox("✈️ Operating Carrier", carrier_options)
        origin = st.selectbox("🛫 Origin Airport", origin_options)

    with col2:
        dest = st.selectbox("🛬 Destination Airport", dest_options)
        distance = st.slider("📏 Distance (miles)", min_value=50, max_value=5000, value=800, step=50)

    with col3:
        departure_hour = st.slider("🕒 Scheduled Departure Hour", 0, 23, 12)
        dep_delay = st.number_input(
            "⏱️ Current Departure Delay (minutes)",
            min_value=-60, max_value=500, value=0, step=1,
            help="Enter 0 if the flight has not departed yet, or the current known departure delay."
        )

    st.write("")
    predict_btn = st.button("🔮  Predict Delay Risk", type="primary", use_container_width=True)

    if predict_btn:
        with st.spinner("Analyzing flight data..."):
            time.sleep(0.4)

            input_df = pd.DataFrame(0, index=[0], columns=model_columns)
            input_df["DEP_DELAY"] = dep_delay
            input_df["Departure Delay ≥ 15 Minutes"] = 1 if dep_delay >= 15 else 0
            input_df["DISTANCE"] = distance
            input_df["Departure Hour"] = departure_hour

            for col, val in [("Operating Carrier_", carrier), ("ORIGIN_", origin), ("DEST_", dest)]:
                full_col = f"{col}{val}"
                if full_col in input_df.columns:
                    input_df[full_col] = 1

            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0][1]

        st.write("")
        if prediction == 1:
            st.markdown(f"""
            <div class="result-card-delay">
                <div class="result-title">⚠️ Likely to be DELAYED (15+ minutes)</div>
                <div class="result-sub">Estimated delay probability: <b>{probability*100:.1f}%</b></div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card-ontime">
                <div class="result-title">✅ Likely to arrive ON TIME</div>
                <div class="result-sub">Estimated delay probability: <b>{probability*100:.1f}%</b></div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")
        c1, c2, c3 = st.columns(3)
        c1.metric("Delay Probability", f"{probability*100:.1f}%")
        c2.metric("On-Time Probability", f"{(1-probability)*100:.1f}%")
        c3.metric("Model Confidence", "High" if abs(probability-0.5) > 0.25 else "Moderate")
        st.progress(min(float(probability), 1.0))

# ----------------------------------------------------------------------
# TAB 2: Insights (reusing Phase 4/5 EDA findings)
# ----------------------------------------------------------------------
with tab2:
    st.markdown('<p class="section-title">Key Findings from Exploratory Data Analysis</p>', unsafe_allow_html=True)

    colA, colB = st.columns(2)

    with colA:
        try:
            delay_by_carrier = pd.read_csv("delay_rate_by_carrier.csv", index_col=0)
            st.write("**Arrival Delay Rate by Carrier (%)**")
            st.bar_chart(delay_by_carrier, color="#1e3c72")
        except FileNotFoundError:
            st.info("Export delay_rate_by_carrier.csv from the modeling notebook to show this chart.")

    with colB:
        try:
            delay_by_hour = pd.read_csv("delay_rate_by_hour.csv", index_col=0)
            st.write("**Arrival Delay Rate by Departure Hour (%)**")
            st.line_chart(delay_by_hour, color="#dc2626")
        except FileNotFoundError:
            st.info("Export delay_rate_by_hour.csv from the modeling notebook to show this chart.")

    st.write("")
    st.markdown('<p class="section-title">Business Takeaways</p>', unsafe_allow_html=True)
    st.markdown("""
    - Departure delay is by far the strongest signal for arrival delay.
    - Delay rates rise steadily through the day — a pattern consistent with **delay propagation**.
    - Airline and destination airport also meaningfully affect delay risk.
    """)

# ----------------------------------------------------------------------
# TAB 3: Model Performance
# ----------------------------------------------------------------------
with tab3:
    st.markdown('<p class="section-title">Final Model: Logistic Regression</p>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", "92.19%")
    m2.metric("Precision", "89.03%")
    m3.metric("Recall", "71.10%")
    m4.metric("F1-Score", "79.06%", "Target ≥ 70%")

    st.write("")
    st.markdown('<p class="section-title">Model Comparison</p>', unsafe_allow_html=True)
    comparison = pd.DataFrame({
        "Model": ["Logistic Regression ⭐", "Decision Tree"],
        "Accuracy": ["92.19%", "92.22%"],
        "Precision": ["89.03%", "90.57%"],
        "Recall": ["71.10%", "69.74%"],
        "F1-Score": ["79.06%", "78.80%"],
    })
    st.dataframe(comparison, hide_index=True, use_container_width=True)
    st.caption("Logistic Regression was selected as the final model for its higher F1-score and Recall.")

    st.write("")
    st.markdown('<p class="section-title">Top Predictive Features</p>', unsafe_allow_html=True)
    top_features = pd.DataFrame({
        "Feature": ["DEP_DELAY", "DEST_ORD", "Carrier: Southwest Airlines", "DEST_DCA", "ORIGIN_DTW"],
        "Impact": [6.46, 0.23, -0.18, 0.14, 0.10],
    })
    st.bar_chart(top_features.set_index("Feature"), color="#1e3c72")
