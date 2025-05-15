import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
import joblib
import sys
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
data_file = BASE_DIR / "data" / "synthetic_versions" / "synthetic_labeled_v1.csv"
preprocesed_file = BASE_DIR / "data" / "preprocessed_journal_entries.csv"
scaler_pkl = BASE_DIR / "models" / "scaler.pkl"
keras_autoencoder = BASE_DIR / "models" / "autoencoder_model.keras"
feature_names = BASE_DIR / "models" / "feature_names.pkl"

try:
    df = pd.read_csv(data_file)

    # df = pd.read_csv("../data/journal_entries.csv")

    # Preprocess data (one-hot encoding)
    X = pd.get_dummies(df.select_dtypes(include=[int, float, object]))  # Handle categoricals
    
    # Save the preprocessed data to CSV
    X.to_csv(preprocesed_file, index=False)
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save the scaler to disk
    joblib.dump(scaler, scaler_pkl)

    # Build the autoencoder model
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_scaled.shape[1],)),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(32, activation='relu'),
        Dense(64, activation='relu'),
        Dense(X_scaled.shape[1], activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='mse')

    # Fit the model
    model.fit(X_scaled, X_scaled, epochs=50, batch_size=32, validation_split=0.1)

    # Save the model in .keras format
    model.save(keras_autoencoder)
    print("Model saved successfully as .keras file")

    # Verify the saved model by loading it back
    loaded_model = load_model(keras_autoencoder)
    print("Model loaded successfully")

    with open(feature_names, "wb") as f:
        joblib.dump(X.columns.tolist(), f)
    print("Feature names saved.")

except Exception as e:
    print(f"An error occurred: {e}")
