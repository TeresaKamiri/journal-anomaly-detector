import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
import joblib
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    df = pd.read_csv("../data/journal_entries.csv")

    # Preprocess data (one-hot encoding)
    X = pd.get_dummies(df.select_dtypes(include=[int, float, object]))  # Handle categoricals
    
    # Save the preprocessed data to CSV
    X.to_csv("../data/preprocessed_journal_entries.csv", index=False)
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save the scaler to disk
    joblib.dump(scaler, "../models/scaler.pkl")

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
    model.save("../models/autoencoder_model.keras")
    print("Model saved successfully as .keras file")

    # Verify the saved model by loading it back
    loaded_model = load_model("../models/autoencoder_model.keras")
    print("Model loaded successfully")

    with open("../models/feature_names.pkl", "wb") as f:
        joblib.dump(X.columns.tolist(), f)
    print("Feature names saved.")

except Exception as e:
    print(f"An error occurred: {e}")
