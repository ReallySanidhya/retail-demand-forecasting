<img width="2690" height="1684" alt="image" src="https://github.com/user-attachments/assets/3d09590f-047c-4b48-9c60-447553738678" />

link to dashboard: https://public.tableau.com/app/profile/sanidhya.sanidhya/viz/RetailDemandForecastingDashboard/RetailDemandForecastingDashboard?publish=yes

# 📦 Retail Demand Forecasting System

I built an end-to-end machine learning system that predicts weekly retail sales, serves predictions via a deployed API, and visualizes results through an interactive dashboard.

**Project Overview**

This project builds a complete forecasting pipeline for retail sales using historical Walmart data.

It includes:

1. Data ingestion and preprocessing
2. Feature engineering for time series forecasting
3. Machine learning model training (Random Forest baseline)
4. Model evaluation using standard metrics
5. Deployment of the model as a live API (Render)
6. Dashboard visualization (Tableau Public)
7. 

**Problem Statement**

Retail businesses need accurate demand forecasts to optimize inventory, reduce stockouts and improve pricing and promotions.

This project predicts weekly sales by store using historical trends and external factors.

🏗️ Architecture

<img width="786" height="226" alt="image" src="https://github.com/user-attachments/assets/061f8490-da11-4d60-adb6-eeb3f04bd83c" />

📊 Dataset

Walmart retail dataset with features:

Store, Date, Dept
Weekly_Sales (target)
Temperature, Fuel Price
CPI, Unemployment
MarkDown features
Holiday indicator
Features Engineered
Lag features (1, 2, 4, 8 weeks)
Rolling mean and standard deviation
Growth rates
Time features (week, month, year)

Mode 🤖 

Baseline Model: Random Forest Regressor

Handles non-linearity
No scaling required
Strong performance baseline
Evaluation Metrics 📈
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

 https://retail-demand-forecasting-pmnm.onrender.com/

Endpoints:
/health

Check API status

/predict

Single-step prediction

/forecast

Multi-step future forecasting

📊 Dashboard

Built using Tableau Public.

🔗 My dashboard Link:

https://public.tableau.com/app/profile/sanidhya.sanidhya/viz/RetailDemandForecastingDashboard/RetailDemandForecastingDashboard?publish=yes

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

<img width="530" height="604" alt="image" src="https://github.com/user-attachments/assets/3a1df4ee-0a6b-4f90-8b84-7105a36c3fdf" />

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

**Why I think this proeject is special**

This project demonstrates a complete machine learning system, from raw data to deployed predictions and visualization, showcasing both data science and ML engineering skills.

Need any help, contact me :) 
                          - Sanidhya.
