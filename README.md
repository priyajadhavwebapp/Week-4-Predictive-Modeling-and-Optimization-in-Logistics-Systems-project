# Week 4 — Predictive Modeling and Optimization in Logistics Systems

This project demonstrates predicting shipment delivery time and using the model as a planning aid. All data is synthetic and intended for learning.

## Files
- `Week4_Predictive_Modeling_Logistics_Report.docx`: report with methodology, evaluation metrics, charts, results, and recommendations.
- `data/logistics_predictive_dataset.csv`: 650 simulated shipments.
- `src/train_model.py`: preprocessing, model comparison, cross-validation, hyperparameter tuning, evaluation, charts, predictions, and model export.
- `dashboard.py`: Streamlit dashboard with filters, KPI cards, mode trade-offs, and an interactive delivery-time prediction form.
- `outputs/charts/`: evaluation visualizations.
- `outputs/tables/`: metrics, CV scores, feature importance, test predictions, mode summaries.
- `models/delivery_time_model.joblib`: tuned model artifact.
- `requirements.txt`: dependencies.

## Setup (Windows / VS Code)
Open this folder in VS Code, then in Terminal run:
```powershell
python -m pip install -r requirements.txt
```
Run model training first to create/update the saved model:
```powershell
python src/train_model.py
```
Launch the dashboard:
```powershell
streamlit run dashboard.py
```
Open the local URL printed by Streamlit (usually `http://localhost:8501`).

## Modeling approach
- Target: `actual_delivery_days` (regression).
- Features: region, mode, shipment type, distance, volume, weight, traffic, weather.
- Models compared: Linear Regression, Decision Tree, Random Forest, Gradient Boosting.
- Metrics: MAE, RMSE, R².
- Validation: 80/20 holdout, five-fold cross-validation on training data, GridSearchCV tuning for Random Forest.
- Optimization concepts: high-duration risk flagging, mode/time/cost trade-offs, buffers, and targeted monitoring.

## Disclaimer
This is synthetic data. Performance scores are not evidence of real-world accuracy. Validate on real time-stamped operational data before deployment.
