import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import os

def train_on_upload(df, model_path, scaler_path, feature_names_path):
    import logging
    logging.info("🔁 Training autoencoder on uploaded data")

    df = df.drop(columns=["label"], errors="ignore")
    df_encoded = pd.get_dummies(df)
    feature_names = df_encoded.columns.tolist()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_encoded)

    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_scaled.shape[1],)),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(32, activation='relu'),
        Dense(64, activation='relu'),
        Dense(X_scaled.shape[1], activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(
        X_scaled, X_scaled,
        epochs=100, batch_size=32, validation_split=0.1,
        callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
        verbose=0
    )

    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(feature_names, feature_names_path)

    logging.info("✅ Autoencoder trained and artifacts saved.")
    return feature_names, scaler, model
