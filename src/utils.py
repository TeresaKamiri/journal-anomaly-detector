import joblib
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

def load_scaler(scaler_path):
    try:
        scaler = joblib.load(scaler_path)
        return scaler
    except Exception as e:
        print(f"Error loading scaler: {e}")
        return None

def load_model_from_file(model_path):
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def scale_data(data, scaler):
    try:
        if scaler:
            return scaler.transform(data)
        else:
            print("Scaler is not available.")
            return None
    except Exception as e:
        print(f"Error in scaling data: {e}")
        return None

def inverse_scale_data(scaled_data, scaler):
    try:
        if scaler:
            return scaler.inverse_transform(scaled_data)
        else:
            print("Scaler is not available.")
            return None
    except Exception as e:
        print(f"Error in inverse scaling: {e}")
        return None
