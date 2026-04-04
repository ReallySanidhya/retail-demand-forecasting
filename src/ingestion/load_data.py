from __future__ import annotations

import argparse
import os
from pathlib import Path
import pandas as pd 


def read_raw_data(input_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    df = df.dropna(subset=["Date", "Store"])

    # Fill numeric columns
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # Fill categorical
    if "Type" in df.columns:
        df["Type"] = df["Type"].fillna("Unknown")

    return df


def aggregate_to_store_week(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["Store", "Date"]).agg({
        "Weekly_Sales": "sum",
        "IsHoliday": "max",
        "Temperature": "mean",
        "Fuel_Price": "mean",
        "MarkDown1": "mean",
        "MarkDown2": "mean",
        "MarkDown3": "mean",
        "MarkDown4": "mean",
        "MarkDown5": "mean",
        "CPI": "mean",
        "Unemployment": "mean",
        "Type": "first",
        "Size": "first"
    }).reset_index()

    grouped = grouped.rename(columns={"Weekly_Sales": "weekly_sales"})

    # Time features
    grouped["year"] = grouped["Date"].dt.year
    grouped["month"] = grouped["Date"].dt.month
    grouped["week_of_year"] = grouped["Date"].dt.isocalendar().week
    grouped["day_of_week"] = grouped["Date"].dt.dayofweek

    grouped = grouped.sort_values(["Store", "Date"])

    return grouped


def write_processed_data(df: pd.DataFrame, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def run_pipeline(input_path: str, output_path: str) -> pd.DataFrame:
    df = read_raw_data(input_path)
    df = clean_data(df)
    df = aggregate_to_store_week(df)
    write_processed_data(df, output_path)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/walmart_cleaned.csv")
    parser.add_argument("--output", default="data/processed/weekly_sales.csv")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"File not found: {args.input}")

    df = run_pipeline(args.input, args.output)

    print("Saved to:", args.output)
    print("Shape:", df.shape)
    print(df.head())
