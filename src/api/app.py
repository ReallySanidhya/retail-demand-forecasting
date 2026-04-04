from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List

import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

MODEL_PATH = "models/baseline/random_forest.joblib"
model = joblib.load(MODEL_PATH)

# Keep this in the same order as the features used during training.
FEATURE_COLUMNS: List[str] = [
    "IsHoliday",
    "Temperature",
    "Fuel_Price",
    "MarkDown1",
    "MarkDown2",
    "MarkDown3",
    "MarkDown4",
    "MarkDown5",
    "CPI",
    "Unemployment",
    "Type",
    "Size",
    "year",
    "month",
    "week_of_year",
    "day_of_week",
    "lag_1",
    "lag_2",
    "lag_4",
    "lag_8",
    "rolling_mean_4",
    "rolling_std_4",
    "growth_1",
    "growth_4",
]


def _as_float_list(values: List[Any]) -> List[float]:
    return [float(v) for v in values]


def _make_feature_row(
    *,
    lag_1: float,
    lag_2: float,
    lag_4: float,
    lag_8: float,
    temperature: float,
    fuel_price: float,
    cpi: float,
    unemployment: float,
    is_holiday: int,
    store_type: float,
    size: float,
    markdowns: Dict[str, float],
    forecast_date: pd.Timestamp,
) -> pd.DataFrame:
    recent = _as_float_list([lag_1, lag_2, lag_4, lag_8])
    rolling_mean_4 = float(pd.Series(recent).mean())
    rolling_std_4 = float(pd.Series(recent).std(ddof=1)) if len(recent) > 1 else 0.0
    growth_1 = float(lag_1 / lag_2) if lag_2 not in (0, 0.0) else 1.0
    growth_4 = float(lag_1 / lag_4) if lag_4 not in (0, 0.0) else 1.0

    row = {
        "IsHoliday": int(is_holiday),
        "Temperature": float(temperature),
        "Fuel_Price": float(fuel_price),
        "MarkDown1": float(markdowns.get("MarkDown1", 0.0)),
        "MarkDown2": float(markdowns.get("MarkDown2", 0.0)),
        "MarkDown3": float(markdowns.get("MarkDown3", 0.0)),
        "MarkDown4": float(markdowns.get("MarkDown4", 0.0)),
        "MarkDown5": float(markdowns.get("MarkDown5", 0.0)),
        "CPI": float(cpi),
        "Unemployment": float(unemployment),
        "Type": float(store_type),
        "Size": float(size),
        "year": int(forecast_date.year),
        "month": int(forecast_date.month),
        "week_of_year": int(forecast_date.isocalendar().week),
        "day_of_week": int(forecast_date.dayofweek),
        "lag_1": float(lag_1),
        "lag_2": float(lag_2),
        "lag_4": float(lag_4),
        "lag_8": float(lag_8),
        "rolling_mean_4": rolling_mean_4,
        "rolling_std_4": rolling_std_4,
        "growth_1": growth_1,
        "growth_4": growth_4,
    }

    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


@app.route("/", methods=["GET"])
def home():
    return jsonify(
        {
            "status": "ok",
            "message": "Retail demand forecasting API is running",
            "endpoints": ["/health", "/predict", "/forecast"],
        }
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        recent_sales = data.get("recent_sales")
        if not recent_sales or len(recent_sales) < 8:
            return jsonify({"error": "recent_sales must contain at least 8 values"}), 400

        recent_sales = _as_float_list(recent_sales)
        forecast_date = pd.to_datetime(data.get("date", datetime.utcnow().date()))

        df = _make_feature_row(
            lag_1=recent_sales[-1],
            lag_2=recent_sales[-2],
            lag_4=recent_sales[-4],
            lag_8=recent_sales[-8],
            temperature=float(data.get("temperature", 50.0)),
            fuel_price=float(data.get("fuel_price", 3.0)),
            cpi=float(data.get("cpi", 200.0)),
            unemployment=float(data.get("unemployment", 7.0)),
            is_holiday=int(data.get("is_holiday", 0)),
            store_type=float(data.get("store_type", 1.0)),
            size=float(data.get("size", 150000)),
            markdowns=data.get("markdowns", {}),
            forecast_date=forecast_date,
        )

        prediction = float(model.predict(df)[0])
        return jsonify({"prediction": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/forecast", methods=["POST"])
def forecast():
    try:
        data = request.get_json(force=True)
        recent_sales = data.get("recent_sales")
        horizon = int(data.get("horizon", 4))

        if not recent_sales or len(recent_sales) < 8:
            return jsonify({"error": "recent_sales must contain at least 8 values"}), 400
        if horizon < 1:
            return jsonify({"error": "horizon must be at least 1"}), 400

        recent_sales = _as_float_list(recent_sales)
        temperature = float(data.get("temperature", 50.0))
        fuel_price = float(data.get("fuel_price", 3.0))
        cpi = float(data.get("cpi", 200.0))
        unemployment = float(data.get("unemployment", 7.0))
        is_holiday = int(data.get("is_holiday", 0))
        store_type = float(data.get("store_type", 1.0))
        size = float(data.get("size", 150000))
        markdowns = data.get("markdowns", {})

        start_date = pd.to_datetime(data.get("start_date", datetime.utcnow().date()))
        freq = data.get("freq", "W-FRI")

        predictions: List[Dict[str, Any]] = []
        history = recent_sales.copy()

        for step in range(1, horizon + 1):
            forecast_date = pd.date_range(start=start_date, periods=step, freq=freq)[-1]

            row = _make_feature_row(
                lag_1=history[-1],
                lag_2=history[-2],
                lag_4=history[-4],
                lag_8=history[-8],
                temperature=temperature,
                fuel_price=fuel_price,
                cpi=cpi,
                unemployment=unemployment,
                is_holiday=is_holiday,
                store_type=store_type,
                size=size,
                markdowns=markdowns,
                forecast_date=forecast_date,
            )

            pred = float(model.predict(row)[0])
            predictions.append(
                {
                    "step": step,
                    "date": forecast_date.strftime("%Y-%m-%d"),
                    "prediction": pred,
                }
            )
            history.append(pred)

        return jsonify(
            {
                "horizon": horizon,
                "predictions": predictions,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
