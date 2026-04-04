from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


DATA_PATH = "data/processed/features.csv"
MODEL_PATH = "models/baseline/random_forest.joblib"
METRICS_PATH = "models/baseline/metrics.json"


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def prepare_features(df: pd.DataFrame):
    target = "weekly_sales"
    drop_cols = ["Date", target]
    if "Store" in df.columns:
        drop_cols.append("Store")

    X = df.drop(columns=drop_cols)
    y = df[target]
    return X, y


def mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate():
    df = load_data(DATA_PATH)
    X, y = prepare_features(df)

    split_idx = int(len(df) * 0.8)
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]

    model = joblib.load(MODEL_PATH)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    mape_value = mape(y_test, preds)

    metrics = {
    "MAE": float(round(mae, 4)),
    "RMSE": float(round(rmse, 4)),
    "MAPE": float(round(mape_value, 4)),
    }

    print(metrics)

    Path(METRICS_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    evaluate()