import sys
import os
import logging

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.render_anomalies import render_anomalies
from src.detect_anomalies import detect_anomalies
from src.explain_anomalies import AnomalyExplainer

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import matplotlib.pyplot as plt
import shap

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()]
)

explainer = AnomalyExplainer(
    model_path="models/autoencoder_model.keras",
    scaler_path="models/scaler.pkl",
    feature_names_path="models/feature_names.pkl"
)

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height or 500, scrolling=True)

# App Title
st.title("📘 Journal Anomaly Detector")
logging.info("App started successfully.")

# Sidebar Upload
st.sidebar.header("📁 Upload Data")
uploaded_file = st.sidebar.file_uploader("📄 Choose a CSV file", type="csv")

# Sidebar Controls
threshold = st.sidebar.slider("Anomaly Detection Threshold (percentile)", 0, 100, 95)
confidence_limit = st.sidebar.slider("⚠️ Confidence Threshold (flag if below)", 0, 100, 60)
confidence_cutoff = confidence_limit / 100.0

show_suspicious = st.sidebar.checkbox("🔍 Include low-confidence entries", value=True)
explain_suspicious = st.sidebar.checkbox("🔎 Explain suspicious entries", value=True)
top_k = st.sidebar.slider("🔬 Top anomalies to explain", 1, 100, 10)

logging.info(f"Threshold: {threshold}, Confidence Limit: {confidence_limit}%")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        logging.info(f"Uploaded file: {uploaded_file.name} with shape {df.shape}")
        st.success(f"✅ Loaded {df.shape[0]} entries.")
    except Exception as e:
        logging.error(f"Error reading uploaded file: {e}")
        st.error(f"❌ Could not read uploaded file: {e}")
        st.stop()

    st.subheader("📊 Data Overview")
    st.write(df.describe())

    # Detect anomalies
    st.subheader("🚨 Detecting Anomalies...")
    detected_df, actual_threshold = detect_anomalies(df, threshold)

    if detected_df is None:
        st.error("Anomaly detection failed.")
        st.stop()

    detected_df["confidence"] = 1.0 - detected_df["anomaly_score"]
    st.success(f"Anomalies detected using threshold: {threshold} percentile")

    # Filter entries
    flagged_anomalies = detected_df[detected_df['is_anomaly'] == True]
    suspicious_entries = detected_df[detected_df["confidence"] < confidence_cutoff]

    explanation_df = pd.concat([flagged_anomalies, suspicious_entries]) if show_suspicious else flagged_anomalies
    explanation_df = explanation_df.drop_duplicates()

    # Display
    st.subheader("🧾 Highlighted Entries")
    if not explanation_df.empty:
        render_anomalies(explanation_df)
    else:
        st.info("No anomalies or suspicious entries found.")
        st.stop()

    if explain_suspicious and not suspicious_entries.empty:
        st.subheader("⚠️ Suspicious Entries (Low Confidence)")
        st.write(suspicious_entries)

    # Confidence Histogram
    st.subheader("📉 Model Confidence Distribution")
    fig_conf, ax_conf = plt.subplots()
    ax_conf.hist(detected_df["confidence"], bins=50, color='skyblue')
    ax_conf.axvline(confidence_cutoff, color='red', linestyle='--', label=f'Confidence Threshold ({confidence_limit}%)')
    ax_conf.set_title("Model Confidence per Entry")
    ax_conf.set_xlabel("Confidence")
    ax_conf.set_ylabel("Count")
    ax_conf.legend()
    st.pyplot(fig_conf)

    # Export Results
    st.subheader("📁 Export Results")
    csv = explanation_df.to_csv(index=False)
    st.download_button(
        label="📤 Download anomalies/suspicious as CSV",
        data=csv,
        file_name='anomalies_detected.csv',
        mime='text/csv'
    )

    # SHAP Explanation
    st.subheader("🧠 SHAP Explanations")
    try:
        with st.spinner("🔍 Computing SHAP values..."):
            shap_output = explainer.explain(explanation_df, top_k=top_k)
        logging.info("SHAP values generated.")
    except Exception as e:
        logging.error(f"SHAP explanation failed: {e}")
        st.error("Failed to generate SHAP values.")
        shap_output = None

    if shap_output is not None:
        shap.initjs()
        shap_values = shap_output.values
        base_values = shap_output.base_values
        features_only = shap_output.features
        sample_indices = features_only.index.tolist()

        if sample_indices:
            selected_index = st.selectbox("Choose an entry to explain", sample_indices)
            sample_index = features_only.index.get_loc(selected_index)

            try:
                base_val = base_values[sample_index] if hasattr(base_values, "__len__") else base_values
                force_plot = shap.force_plot(
                    base_val,
                    shap_values[sample_index],
                    features_only.iloc[sample_index]
                )
                st_shap(force_plot)
                logging.info(f"Displayed SHAP force plot for index {sample_index}")
            except Exception as e:
                logging.error(f"SHAP force plot error: {e}")
                st.error("SHAP force plot failed.")


        st.subheader("📈 SHAP Summary Plot")
        try:
            fig_summary = plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, features_only, show=False)
            st.pyplot(fig_summary)
            logging.info("Displayed SHAP summary plot.")
        except Exception as e:
            logging.error(f"SHAP summary plot failed: {e}")
            st.error("SHAP summary plot failed.")
    else:
        st.warning("No SHAP values available.")


    # SHAP Explanation
    # st.subheader("🧠 SHAP Explanations")
    # try:
    #     with st.spinner("🔍 Computing SHAP values..."):
    #         shap_values = explainer.explain(explanation_df, top_k=top_k)
    #     logging.info("SHAP values generated.")
    # except Exception as e:
    #     logging.error(f"SHAP explanation failed: {e}")
    #     st.error("Failed to generate SHAP values.")
    #     shap_values = None

    # if shap_values is not None:
    #     shap.initjs()

    #     # Use only the features used in SHAP (top_k rows only)
    #     features_only = explanation_df.drop(columns=["anomaly_score", "is_anomaly", "confidence"], errors='ignore')
    #     features_only = features_only.iloc[:shap_values.values.shape[0]]

    #     sample_indices = features_only.index.tolist()

    #     if sample_indices:
    #         selected_index = st.selectbox("Choose an entry to explain", sample_indices)
    #         sample_index = features_only.index.get_loc(selected_index)

    #         if shap_values.values.ndim != 2 or sample_index >= shap_values.values.shape[0]:
    #             st.warning("Selected index is out of bounds for SHAP values.")
    #         else:
    #             try:
    #                 force_plot = shap.force_plot(
    #                     shap_values.base_values[sample_index],
    #                     shap_values.values[sample_index],
    #                     features_only.iloc[sample_index]
    #                 )
    #                 st_shap(force_plot)
    #                 logging.info(f"Displayed SHAP force plot for index {sample_index}")
    #             except Exception as e:
    #                 logging.error(f"SHAP force plot error: {e}")
    #                 st.error("SHAP force plot failed.")

    #     st.subheader("📈 SHAP Summary Plot")
    #     try:
    #         fig_summary = plt.figure(figsize=(10, 6))
    #         shap.summary_plot(shap_values.values, features_only, show=False)
    #         st.pyplot(fig_summary)
    #         logging.info("Displayed SHAP summary plot.")
    #     except Exception as e:
    #         logging.error(f"SHAP summary plot failed: {e}")
    #         st.error("SHAP summary plot failed.")
    # else:
    #     st.warning("No SHAP values available.")

# Cleanup logging
for handler in logging.root.handlers[:]:
    handler.close()
    logging.root.removeHandler(handler)
