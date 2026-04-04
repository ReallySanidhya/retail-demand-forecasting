from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

DATA_PATH = "data/processed/features.csv"
MODEL_PATH = "models/baseline/random_forest.joblib"


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


def train():
    df = load_data(DATA_PATH)

    X, y = prepare_features(df)

    # Time-aware split: keep last 20% for testing
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5

    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")

    Path("models/baseline").mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    train()