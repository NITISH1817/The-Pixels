"""
monitoring/dashboard.py
-----------------------
Streamlit Monitoring Dashboard for Student Performance Prediction & MLOps Pipeline.

Run with:
    streamlit run monitoring/dashboard.py
"""

import json
from pathlib import Path
import pandas as pd
import streamlit as st

MONITORING_DIR = Path(__file__).resolve().parent
LOGS_DIR = MONITORING_DIR / "logs"
CSV_LOG_PATH = LOGS_DIR / "predictions.csv"
JSONL_LOG_PATH = LOGS_DIR / "predictions.jsonl"

st.set_page_config(
    page_title="MLOps Prediction & Performance Monitoring",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Student Performance MLOps Monitoring Dashboard")
st.markdown("Real-time monitoring of model predictions, risk distributions, and prediction trends.")

# Helper to load logs
def load_prediction_logs() -> pd.DataFrame:
    if not CSV_LOG_PATH.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(CSV_LOG_PATH)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    except Exception:
        return pd.DataFrame()

df_logs = load_prediction_logs()

if df_logs.empty:
    st.info("ℹ️ No prediction logs found yet. Make predictions via Web App or API (`/predict`) to view live metrics here.")
    st.write("Log File Location:", str(CSV_LOG_PATH))
else:
    # Summary Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    total_preds = len(df_logs)
    pass_count = len(df_logs[df_logs["prediction"] == "PASS"])
    fail_count = len(df_logs[df_logs["prediction"] == "FAIL"])
    avg_success_prob = df_logs["success_probability"].mean() if "success_probability" in df_logs.columns else 0.0
    latest_model_ver = df_logs["model_version"].iloc[-1] if "model_version" in df_logs.columns else "N/A"

    col1.metric("Total Predictions", total_preds)
    col2.metric("Pass / Fail Count", f"🟢 {pass_count} / 🔴 {fail_count}")
    col3.metric("Avg Success Probability", f"{avg_success_prob:.1f}%")
    col4.metric("Active Model Version", latest_model_ver)

    st.markdown("---")

    # Visualizations Row
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("📈 Prediction & Risk Level Distribution")
        if "risk_level" in df_logs.columns:
            risk_counts = df_logs["risk_level"].value_counts()
            st.bar_chart(risk_counts)
        else:
            st.write("Risk level column not present.")

    with col_right:
        st.subheader("🗓️ Daily Prediction Count")
        if "timestamp" in df_logs.columns:
            df_logs["date"] = df_logs["timestamp"].dt.date
            daily_counts = df_logs.groupby("date").size()
            st.line_chart(daily_counts)
        else:
            st.write("Timestamp column not present.")

    st.markdown("---")

    # Recent Predictions Log Table
    st.subheader("📜 Recent Prediction Logs")
    st.dataframe(df_logs.sort_values(by="timestamp", ascending=False).head(20), use_container_width=True)
