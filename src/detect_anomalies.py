import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import numpy as np
import sys
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
anomalies_file = BASE_DIR / "anomalies" / "anomalies_found.csv"
expected_features_file = BASE_DIR / "models" / "feature_names.pkl"
scaler_pkl_file = BASE_DIR / "models" / "scaler.pkl"
keras_model = BASE_DIR / "models" / "autoencoder_model.keras"

def detect_anomalies(df, threshold=None, save_path=anomalies_file):
    try:
        print("Starting anomaly detection...")

        # Extract timestamp features
        if 'posting_date' in df.columns:
            df['posting_date'] = pd.to_datetime(df['posting_date'], errors='coerce')
            df['day_of_week'] = df['posting_date'].dt.weekday
            df['is_weekend'] = df['day_of_week'] >= 5

        # Preprocess the data
        X = pd.get_dummies(df)

        # Align with expected features
        expected_features = joblib.load(expected_features_file)
        X = X.reindex(columns=expected_features, fill_value=0)

        # Load scaler and transform
        scaler = joblib.load(scaler_pkl_file)
        X_scaled = scaler.transform(X)

        # Load model
        model = load_model(keras_model)

        # Predict and calculate anomaly scores
        X_pred = model.predict(X_scaled)
        mse = np.mean(np.power(X_scaled - X_pred, 2), axis=1)
        df['anomaly_score'] = mse

        # Set dynamic threshold if not provided
        if threshold is None:
            threshold = np.percentile(mse, 95)

        df['is_anomaly'] = df['anomaly_score'] > threshold

        # Sort by severity
        df = df.sort_values(by="anomaly_score", ascending=False)

        # Save only anomalies to CSV
        anomalies = df[df['is_anomaly']]
        anomalies.to_csv(save_path, index=False)
        print(f"🔍 {len(anomalies)} anomalies saved to {save_path}")

        return df, threshold

    except Exception as e:
        print(f"❌ Error in anomaly detection: {e}")
        return None
        