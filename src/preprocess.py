import pandas as pd

def preprocess_real_journals(df, feature_names=None):
    df_clean = df.drop(columns=["label"], errors="ignore")  # preserve label for eval
    X = pd.get_dummies(df_clean)
    if feature_names:
        X = X.reindex(columns=feature_names, fill_value=0)
    return X
