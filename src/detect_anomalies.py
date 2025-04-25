import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#def detect_anomalies(df, threshold=None, save_path="C:/Users/hp/Desktop/Peppper ML/audit_anomaly_detector_project_v2/anomalies/anomalies_found.csv"): # you can use hardcoded paths

def detect_anomalies(df, threshold=None, save_path="../anomalies/anomalies_found.csv"):
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
        expected_features = joblib.load("C:/Users/hp/Desktop/Peppper ML/audit_anomaly_detector_project_v2/models/feature_names.pkl")
        X = X.reindex(columns=expected_features, fill_value=0)

        # Load scaler and transform
        scaler = joblib.load("C:/Users/hp/Desktop/Peppper ML/audit_anomaly_detector_project_v2/models/scaler.pkl")
        X_scaled = scaler.transform(X)

        # Load model
        model = load_model("C:/Users/hp/Desktop/Peppper ML/audit_anomaly_detector_project_v2/models/autoencoder_model.keras")

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


# file_path = "C:/Users/hp/Desktop/Peppper ML/audit_anomaly_detector_project_v2/data/journal_entries.csv"
# df = pd.read_csv(file_path)

# # Define threshold
# threshold = 0.95  # or any value you'd like to use

# # Call the anomaly detection function
# detected_df, threshold = detect_anomalies(df, threshold)
