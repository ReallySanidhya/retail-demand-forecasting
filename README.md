<img width="2690" height="1684" alt="image" src="https://github.com/user-attachments/assets/3d09590f-047c-4b48-9c60-447553738678" />

link to dashboard: https://public.tableau.com/app/profile/sanidhya.sanidhya/viz/RetailDemandForecastingDashboard/RetailDemandForecastingDashboard?publish=yes

📦 Retail Demand Forecasting System

An end-to-end machine learning system that predicts weekly retail sales, serves predictions via a deployed API, and visualizes results through an interactive dashboard.

🚀 Project Overview

This project builds a complete forecasting pipeline for retail sales using historical Walmart data.

It includes:

Data ingestion and preprocessing
Feature engineering for time series forecasting
Machine learning model training (Random Forest baseline)
Model evaluation using standard metrics
Deployment of the model as a live API (Render)
Dashboard visualization (Tableau Public)
🧠 Problem Statement

Retail businesses need accurate demand forecasts to:

optimize inventory
reduce stockouts
improve pricing and promotions

This project predicts weekly sales by store using historical trends and external factors.

🏗️ Architecture
Raw Data → Feature Engineering → Model Training → API (Render)
                                   ↓
                              Predictions
                                   ↓
                           CSV Export Script
                                   ↓
                          Tableau Dashboard
📊 Dataset

Walmart retail dataset with features:

Store, Date, Dept
Weekly_Sales (target)
Temperature, Fuel Price
CPI, Unemployment
MarkDown features
Holiday indicator
⚙️ Features Engineered
Lag features (1, 2, 4, 8 weeks)
Rolling mean and standard deviation
Growth rates
Time features (week, month, year)
🤖 Model

Baseline Model: Random Forest Regressor

Handles non-linearity
No scaling required
Strong performance baseline
📈 Evaluation Metrics
MAE (Mean Absolute Error)
RMSE (Root Mean Squared Error)
MAPE (Mean Absolute Percentage Error)

Example results:

MAE: ~12,709
RMSE: ~22,071
MAPE: ~1.86%
🌐 API (Deployed)

The model is deployed as a Flask API on Render.

🔗 Live API:

👉 https://retail-demand-forecasting-pmnm.onrender.com/

Endpoints:
/health

Check API status

/predict

Single-step prediction

/forecast

Multi-step future forecasting

📊 Dashboard

Built using Tableau Public.

🔗 Dashboard Link:

👉 (paste your Tableau Public link here)

Features:
Actual vs Forecast trend line
Future forecast visualization
Clean, interactive layout
🔄 Data Flow

The dashboard is powered by predictions generated from the deployed API:

API → export_dashboard_data.py → dashboard_data.csv → Tableau
🛠️ Tech Stack
Python (Pandas, Scikit-learn)
Flask (API)
Render (deployment)
Tableau Public (visualization)
📁 Project Structure
src/
  ├── ingestion/
  ├── features/
  ├── train/
  ├── evaluate/
  ├── api/

data/
  ├── raw/
  ├── processed/

models/
  ├── baseline/
  ├── lstm/
  ├── artifacts/

scripts/
  └── export_dashboard_data.py
▶️ How to Run Locally
1. Install dependencies
pip install -r requirements.txt
2. Run pipeline
python src/ingestion/load_data.py
python src/features/feature_engineering.py
python src/train/train_baseline.py
3. Run API
python src/api/app.py
4. Export dashboard data
python scripts/export_dashboard_data.py
💡 Key Learnings
Building end-to-end ML pipelines
Feature engineering for time series
Model deployment using Flask + Render
API-driven data workflows
Dashboard design and storytelling
🚀 Future Improvements
LSTM / deep learning models
Real-time dashboard integration
Hyperparameter tuning
Multi-store forecasting
📸 Screenshots

(Add your dashboard screenshot here)

👤 Author

Sanidhya

⭐ Summary

This project demonstrates a complete machine learning system, from raw data to deployed predictions and visualization — showcasing both data science and ML engineering skills.
