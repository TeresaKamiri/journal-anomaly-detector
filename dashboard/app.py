import sys
import os
import logging
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.render_anomalies import render_anomalies
from src.detect_anomalies import detect_anomalies
from src.explain_anomalies import AnomalyExplainer
from src.preprocess import preprocess_real_journals
from src.benchmark_models import run_live_benchmark
from src.train_on_upload import train_on_upload
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import shap
import joblib

shap.initjs()

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()]
)

MODEL_PATH = MODEL_DIR / "autoencoder_model.keras"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
FEATURES_PATH = MODEL_DIR / "feature_names.pkl"

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height or 500, scrolling=True)

# App Title
st.title("📘 Journal Anomaly Detector")
logging.info("App started successfully.")

# Sidebar Upload
st.sidebar.header("📁 Upload Data")
uploaded_file = st.sidebar.file_uploader("📄 Choose a CSV file", type="csv")

retrain_model = st.sidebar.checkbox("📚 Retrain autoencoder on this dataset")

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
        has_labels = "label" in df.columns

        if retrain_model:
            st.warning("⚠️ Retraining model on uploaded data... may take a few seconds.")
            feature_names, scaler, model = train_on_upload(df, MODEL_PATH, SCALER_PATH, FEATURES_PATH)
            explainer = AnomalyExplainer(str(MODEL_PATH), str(SCALER_PATH), str(FEATURES_PATH))
            st.success("✅ Autoencoder retrained and updated.")
        else:
            feature_names = joblib.load(FEATURES_PATH)
            scaler = joblib.load(SCALER_PATH)
            explainer = AnomalyExplainer(str(MODEL_PATH), str(SCALER_PATH), str(FEATURES_PATH))

        feature_names = joblib.load(MODEL_DIR / "feature_names.pkl")
        scaler = joblib.load(MODEL_DIR / "scaler.pkl")

        X_input = preprocess_real_journals(df, feature_names)
        X_scaled = scaler.transform(X_input)

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

    if has_labels:
        from sklearn.metrics import classification_report
        y_true = df["label"].astype(int)
        y_pred = detected_df["is_anomaly"].astype(int)
        st.subheader("📋 Evaluation on Labeled Data")
        report = classification_report(y_true, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).T)

        st.subheader("📊 Live Benchmark on Uploaded Labeled Data")
        y_true = df["label"].astype(int)
        live_results = run_live_benchmark(X_scaled, y_true, explainer.model, scaler, threshold)
        st.dataframe(live_results.style.highlight_max(axis=0, color="lightgreen").format("{:.2f}"))

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

    # Cap top_k to available rows to avoid index errors
    top_k = min(top_k, explanation_df.shape[0])

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
            # Align sample index to SHAP shape
            sample_index = features_only.index.get_loc(selected_index)

            if sample_index >= shap_values.shape[0]:
                st.warning("Selected entry is out of bounds for SHAP values.")
                logging.warning(f"Sample index {sample_index} exceeds SHAP values shape {shap_values.shape}")
            else:
                try:
                    base_val = base_values[sample_index] if hasattr(base_values, "__len__") else base_values

                    force_plot = shap.force_plot(
                        base_val,
                        shap_values[sample_index],
                        features_only.iloc[sample_index]
                    )
                    _ = force_plot  # prevent IPython from trying to auto-display
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


    # =============================
    # 📊 Benchmark Model Evaluation
    # =============================
    st.subheader("📊 Benchmark Model Evaluation (Autoencoder vs KNN vs RF)")

    benchmark_path = BASE_DIR / "evaluation" / "batch_benchmark_results.csv"
    if benchmark_path.exists():
        benchmark_df = pd.read_csv(benchmark_path, index_col=[0, 1])

        # Display metrics table
        st.markdown("### 🔍 Performance Summary Table")

        def highlight_best(df):
            return df.style.highlight_max(axis=0, color="lightgreen").format("{:.2f}")

        st.dataframe(highlight_best(benchmark_df))

        # Interactive bar plot by metric
        metric_to_plot = st.selectbox("📈 Choose metric to visualize", ["f1", "precision", "recall", "roc_auc"])
        st.markdown(f"### 📊 {metric_to_plot.upper()} Score Comparison")

        try:
            import seaborn as sns
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(
                data=benchmark_df.reset_index(),
                x="Dataset", y=metric_to_plot, hue="Model", ax=ax
            )
            ax.set_title(f"{metric_to_plot.upper()} Score per Model per Dataset")
            ax.set_ylabel(metric_to_plot.upper())
            ax.set_xlabel("Dataset")
            plt.xticks(rotation=45)
            st.pyplot(fig)

            # Save figure to disk
            chart_path = BASE_DIR / "evaluation" / f"{metric_to_plot}_score_comparison.png"
            fig.savefig(chart_path, bbox_inches="tight")
            st.caption(f"📁 Saved to: `{chart_path.name}`")
        except Exception as e:
            st.warning(f"Could not display {metric_to_plot} chart: {e}")

    else:
        st.info("Benchmark results not found. Please run the benchmark_models.py script.")


# Cleanup logging
for handler in logging.root.handlers[:]:
    handler.close()
    logging.root.removeHandler(handler)
