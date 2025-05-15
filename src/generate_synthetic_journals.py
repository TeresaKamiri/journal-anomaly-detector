import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
batch_synth_dir = BASE_DIR / "data" / "synthetic_versions"

def generate_synthetic_data(n_normal=950, n_fraud=50, seed=42, fraud_types=None):
    np.random.seed(seed)
    random.seed(seed)

    if fraud_types is None:
        fraud_types = ["round", "duplicate_vendor", "weekend"]

    vendors = [f"Vendor_{i}" for i in range(1, 21)]
    accounts = ["Sales", "Marketing", "IT", "HR", "Finance", "Legal", "Admin"]

    def random_date():
        start = datetime(2023, 1, 1)
        return start + timedelta(days=np.random.randint(0, 365))

    # Normal entries
    data = []
    for _ in range(n_normal):
        data.append({
            "amount": round(np.random.normal(1000, 300), 2),
            "vendor": np.random.choice(vendors),
            "account": np.random.choice(accounts),
            "posting_date": random_date(),
            "user_id": np.random.randint(1000, 1100),
            "description": "Regular business transaction",
            "label": 0
        })

    # Anomalous entries
    for _ in range(n_fraud):
        fraud_type = np.random.choice(fraud_types)
        entry = {
            "amount": round(np.random.normal(1000, 300), 2),
            "vendor": np.random.choice(vendors),
            "account": np.random.choice(accounts),
            "posting_date": random_date(),
            "user_id": np.random.randint(1000, 1100),
            "description": "Suspicious transaction",
            "label": 1
        }

        if fraud_type == "round":
            entry["amount"] = np.random.choice([5000, 10000, 20000])
        elif fraud_type == "duplicate_vendor":
            entry["vendor"] = "Vendor_5"
            entry["account"] = "Legal" if np.random.rand() > 0.5 else "HR"
        elif fraud_type == "weekend":
            while True:
                d = random_date()
                if d.weekday() >= 5:
                    entry["posting_date"] = d
                    break

        data.append(entry)

    df = pd.DataFrame(data)
    df["posting_date"] = pd.to_datetime(df["posting_date"])
    return df


def save_multiple_versions(output_dir=batch_synth_dir):
    os.makedirs(output_dir, exist_ok=True)
    configs = [
        {"n_normal": 950, "n_fraud": 50, "seed": 1},
        {"n_normal": 900, "n_fraud": 100, "seed": 2},
        {"n_normal": 800, "n_fraud": 200, "seed": 3},
        {"n_normal": 950, "n_fraud": 50, "seed": 4, "fraud_types": ["round"]},
        {"n_normal": 950, "n_fraud": 50, "seed": 5, "fraud_types": ["duplicate_vendor"]},
    ]

    for i, config in enumerate(configs):
        df = generate_synthetic_data(**config)
        filename = os.path.join(output_dir, f"synthetic_labeled_v{i+1}.csv")
        df.to_csv(filename, index=False)
        print(f"✅ Saved: {filename}")


if __name__ == "__main__":
    save_multiple_versions()
