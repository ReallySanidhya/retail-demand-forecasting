import pandas as pd
from pathlib import Path


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Store", "Date"]).reset_index(drop=True)
    return df


def create_lag_features(df: pd.DataFrame, lags=None) -> pd.DataFrame:
    if lags is None:
        lags = [1, 2, 4, 8]

    df = df.copy()
    for lag in lags:
        df[f"lag_{lag}"] = df.groupby("Store", group_keys=False)["weekly_sales"].shift(lag)
    return df


def create_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    grouped = df.groupby("Store", group_keys=False)["weekly_sales"]

    df["rolling_mean_4"] = grouped.transform(lambda x: x.shift(1).rolling(window=4).mean())
    df["rolling_std_4"] = grouped.transform(lambda x: x.shift(1).rolling(window=4).std())
    return df


def create_growth_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["growth_1"] = df["weekly_sales"] / df["lag_1"]
    df["growth_4"] = df["weekly_sales"] / df["lag_4"]

    df["growth_1"] = df["growth_1"].replace([float("inf"), float("-inf")], pd.NA)
    df["growth_4"] = df["growth_4"].replace([float("inf"), float("-inf")], pd.NA)
    return df


def clean_final(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna().reset_index(drop=True)

    # Clip after making a fresh copy to avoid SettingWithCopyWarning.
    df.loc[:, "growth_1"] = df["growth_1"].clip(0, 5)
    df.loc[:, "growth_4"] = df["growth_4"].clip(0, 5)

    return df


def save_data(df: pd.DataFrame, output_path: str):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def run_pipeline():
    input_path = "data/processed/weekly_sales.csv"
    output_path = "data/processed/features.csv"

    df = load_data(input_path)
    df = create_lag_features(df)
    df = create_rolling_features(df)
    df = create_growth_features(df)
    df = clean_final(df)

    save_data(df, output_path)

    print("Feature dataset saved!")
    print("Shape:", df.shape)
    print(df.head())


if __name__ == "__main__":
    run_pipeline()
