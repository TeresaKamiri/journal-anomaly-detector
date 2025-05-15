import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from collections import defaultdict
import joblib
import os
from pathlib import Path
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent.parent
batch_files = BASE_DIR / "data" / "synthetic_versions"
eval_csv = BASE_DIR / "evaluation" / "batch_benchmark_results.csv"

# Load autoencoder model and supporting objects once
autoencoder = load_model(BASE_DIR / "models" / "autoencoder_model.keras")
scaler = joblib.load(BASE_DIR / "models" / "scaler.pkl")
feature_names = joblib.load(BASE_DIR / "models" / "feature_names.pkl")

def generate_knn_scores(X, n_neighbors=5):
    knn = NearestNeighbors(n_neighbors=n_neighbors)
    knn.fit(X)
    distances, _ = knn.kneighbors(X)
    scores = distances.mean(axis=1)
    return scores

def evaluate_supervised_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    }

def evaluate_autoencoder(df, threshold_percentile=95):
    labels = df['label'].astype(int)
    X = pd.get_dummies(df.drop(columns=["label"]))
    X = X.reindex(columns=feature_names, fill_value=0)
    X_scaled = scaler.transform(X)

    X_pred = autoencoder.predict(X_scaled)
    mse = np.mean(np.square(X_scaled - X_pred), axis=1)

    threshold = np.percentile(mse, threshold_percentile)
    y_pred = (mse > threshold).astype(int)

    return {
        "precision": precision_score(labels, y_pred, zero_division=0),
        "recall": recall_score(labels, y_pred, zero_division=0),
        "f1": f1_score(labels, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(labels, mse)
    }

def run_benchmarks(data_path, label_column="label"):
    results = defaultdict(dict)

    df = pd.read_csv(data_path)
    if label_column not in df.columns:
        raise ValueError("Dataset must contain a 'label' column for evaluation.")

    y = df[label_column].astype(int)
    X = df.drop(columns=[label_column])

    X = pd.get_dummies(X)
    scaler_bench = StandardScaler()
    X_scaled = scaler_bench.fit_transform(X)

    # --- KNN ---
    knn_scores = generate_knn_scores(X_scaled)
    threshold = np.percentile(knn_scores, 95)
    y_knn_pred = (knn_scores > threshold).astype(int)

    results["KNN"] = {
        "precision": precision_score(y, y_knn_pred),
        "recall": recall_score(y, y_knn_pred),
        "f1": f1_score(y, y_knn_pred),
        "roc_auc": roc_auc_score(y, knn_scores)
    }

    # --- Random Forest ---
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    results["Random Forest"] = evaluate_supervised_model(rf, X_train, y_train, X_test, y_test)

    # --- Autoencoder ---
    auto_results = evaluate_autoencoder(df.copy())
    results["Autoencoder"] = auto_results

    return pd.DataFrame(results).T

def run_batch(directory=batch_files):
    files = sorted([f for f in os.listdir(directory) if f.endswith(".csv")])
    combined_results = {}

    for file in files:
        path = os.path.join(directory, file)
        print(f"Running benchmarks on {file}")
        try:
            results_df = run_benchmarks(path)
            combined_results[file] = results_df
        except Exception as e:
            print(f"❌ Failed on {file}: {e}")

    if combined_results:
        final_df = pd.concat(combined_results, names=["Dataset", "Model"])
        final_df.to_csv(eval_csv)
        print("✅ All results saved to", eval_csv)

        # Optional enhancement: highlight top F1 scores
        pivoted = final_df.reset_index().pivot(index="Dataset", columns="Model", values="f1")
        fig, ax = plt.subplots(figsize=(10, 5))
        pivoted.plot(kind="bar", ax=ax)
        ax.set_title("F1 Score Comparison Across Models")
        ax.set_ylabel("F1 Score")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        fig.savefig(BASE_DIR / "evaluation" / "f1_comparison_chart.png")
        print("✅ Bar chart saved.")
    else:
        print("No results generated.")

def run_live_benchmark(X_scaled, y_true, autoencoder, scaler, threshold=95):
    results = {}

    # --- Autoencoder ---
    X_pred = autoencoder.predict(X_scaled)
    mse = np.mean(np.square(X_scaled - X_pred), axis=1)
    ae_thresh = np.percentile(mse, threshold)
    y_ae_pred = (mse > ae_thresh).astype(int)

    results["Autoencoder"] = {
        "precision": precision_score(y_true, y_ae_pred, zero_division=0),
        "recall": recall_score(y_true, y_ae_pred, zero_division=0),
        "f1": f1_score(y_true, y_ae_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, mse)
    }

    # --- KNN ---
    knn = NearestNeighbors(n_neighbors=5)
    knn.fit(X_scaled)
    distances, _ = knn.kneighbors(X_scaled)
    knn_scores = distances.mean(axis=1)
    knn_thresh = np.percentile(knn_scores, threshold)
    y_knn_pred = (knn_scores > knn_thresh).astype(int)

    results["KNN"] = {
        "precision": precision_score(y_true, y_knn_pred, zero_division=0),
        "recall": recall_score(y_true, y_knn_pred, zero_division=0),
        "f1": f1_score(y_true, y_knn_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, knn_scores)
    }

    # --- Random Forest ---
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_scaled, y_true)
    y_rf_pred = rf.predict(X_scaled)
    y_rf_prob = rf.predict_proba(X_scaled)[:, 1]

    results["Random Forest"] = {
        "precision": precision_score(y_true, y_rf_pred, zero_division=0),
        "recall": recall_score(y_true, y_rf_pred, zero_division=0),
        "f1": f1_score(y_true, y_rf_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_rf_prob)
    }

    return pd.DataFrame(results).T

if __name__ == "__main__":
    run_batch()
