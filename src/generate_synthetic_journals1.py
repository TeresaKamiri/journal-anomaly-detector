import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_synthetic_data(n_normal=950, n_fraud=50, seed=42):
    np.random.seed(seed)
    vendors = [f"Vendor_{i}" for i in range(1, 21)]
    accounts = ["Sales", "Marketing", "IT", "HR", "Finance", "Legal", "Admin"]

    def random_date():
        start = datetime(2023, 1, 1)
        return start + timedelta(days=np.random.randint(0, 365))

    # Generate normal entries
    data = []
    for _ in range(n_normal):
        data.append({
            "amount": round(np.random.normal(1000, 300), 2),
            "vendor": np.random.choice(vendors),
            "account": np.random.choice(accounts),
            "posting_date": random_date(),
            "user_id": np.random.randint(1000, 1100),
            "description": "Regular business transaction"
        })

    # Injected fraud entries
    for _ in range(n_fraud):
        fraud_type = np.random.choice(["round", "duplicate_vendor", "weekend"])
        entry = {
            "amount": round(np.random.normal(1000, 300), 2),
            "vendor": np.random.choice(vendors),
            "account": np.random.choice(accounts),
            "posting_date": random_date(),
            "user_id": np.random.randint(1000, 1100),
            "description": "Suspicious transaction"
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

# Save to CSV
df = generate_synthetic_data()
df.to_csv("../data/journal_entries.csv", index=False)
print("✅ Synthetic data with anomalies saved.")
