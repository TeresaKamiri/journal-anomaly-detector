import shap
import pandas as pd
import joblib
import numpy as np
import os
import sys
import logging
from tensorflow.keras.models import load_model
from collections import namedtuple

# Ensure root path is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Named tuple for output
# SHAPOutput = namedtuple("SHAPOutput", ["values", "base_values"])

# class AnomalyExplainer:
#     def __init__(self, model_path, scaler_path, feature_names_path):
#         """
#         Initialize explainer by loading model and preprocessing tools.
#         """
#         try:
#             self.model = load_model(model_path)
#             self.scaler = joblib.load(scaler_path)
#             self.expected_features = joblib.load(feature_names_path)
#             logger.info("Model, scaler, and feature names loaded successfully.")
#         except Exception as e:
#             logger.error(f"Failed to load model or preprocessing tools: {e}")
#             raise

#     def explain(self, df, top_k=5):
#         """
#         Generate SHAP explanations for top_k anomalies in the DataFrame.

#         Args:
#             df (pd.DataFrame): Input data including anomaly_score or confidence columns.
#             top_k (int): Number of top anomalies to explain.

#         Returns:
#             SHAPOutput: Contains SHAP values and base values.
#         """
#         try:
#             # Drop non-feature columns
#             df_clean = df.drop(columns=["anomaly_score", "is_anomaly", "confidence"], errors='ignore')
            
#             # Sort by anomaly score if present
#             if "anomaly_score" in df.columns:
#                 df = df.sort_values("anomaly_score", ascending=False)
#                 df_clean = df.drop(columns=["anomaly_score", "is_anomaly", "confidence"], errors='ignore')

#             # One-hot encode and align            
#             X = pd.get_dummies(df_clean)
#             X = X.reindex(columns=self.expected_features, fill_value=0)

#             # Scale features
#             X_scaled = self.scaler.transform(X)

#             # Limit to top_k
#             top_k = min(top_k, X_scaled.shape[0])
#             X_top = X_scaled[:top_k]

#             # Define custom prediction function
#             def reconstruction_error(x):
#                 pred = self.model.predict(x)
#                 return np.mean((x - pred) ** 2, axis=1)

#             # Use custom prediction
#             explainer = shap.KernelExplainer(reconstruction_error, X_top)

#             # Compute SHAP values
#             shap_values = explainer.shap_values(X_top)
#             logger.info(f"SHAP values computed for top {top_k} samples.")

#             return SHAPOutput(values=np.array(shap_values), base_values=np.array(explainer.expected_value))

#         except Exception as e:
#             logger.error(f"[SHAP explain error] {e}")
#             return None

# Named tuple for output
SHAPOutput = namedtuple("SHAPOutput", ["values", "base_values", "features"])

class AnomalyExplainer:
    def __init__(self, model_path, scaler_path, feature_names_path):
        try:
            self.model = load_model(model_path)
            self.scaler = joblib.load(scaler_path)
            self.expected_features = joblib.load(feature_names_path)
            logger.info("Model, scaler, and feature names loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model or preprocessing tools: {e}")
            raise

    def explain(self, df, top_k=5):
        try:
            df_clean = df.drop(columns=["anomaly_score", "is_anomaly", "confidence"], errors='ignore')
            
            if "anomaly_score" in df.columns:
                df = df.sort_values("anomaly_score", ascending=False)
                df_clean = df.drop(columns=["anomaly_score", "is_anomaly", "confidence"], errors='ignore')

            X = pd.get_dummies(df_clean)
            X = X.reindex(columns=self.expected_features, fill_value=0)
            X_scaled = self.scaler.transform(X)

            top_k = min(top_k, X_scaled.shape[0])
            X_top = X_scaled[:top_k]
            X_top_df = X.iloc[:top_k]  # Keep the unscaled version for SHAP visualization

            def reconstruction_error(x):
                pred = self.model.predict(x)
                return np.mean((x - pred) ** 2, axis=1)

            explainer = shap.KernelExplainer(reconstruction_error, X_top)
            shap_vals = explainer.shap_values(X_top)

            # Handle single-output case (common for anomaly scores)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[0]

            base_value = explainer.expected_value
            base_values = np.array(base_value if isinstance(base_value, (list, np.ndarray)) else [base_value])

            logger.info(f"SHAP values computed for top {top_k} samples.")

            return SHAPOutput(
                values=np.array(shap_vals),
                base_values=base_values,
                features=X_top_df
            )
        except Exception as e:
            logger.error(f"[SHAP explain error] {e}")
            return None

