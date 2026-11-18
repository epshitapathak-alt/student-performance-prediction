# Student Performance Prediction System Using Machine Learning

A decision-support / early-warning web app that estimates a student's final exam marks, percentage and performance category (Excellent, Good, Average, At Risk), with a risk probability, explanation of influencing factors and personalised recommendations. **All outputs are ML-based estimates, not guarantees.**

## Problem statement and objectives
Struggling students are often identified only after results. This project uses pre-exam signals (attendance, marks, study habits, engagement) to flag risk early. Objectives: regression + classification, explainability, personalised advice, validated inputs, stored history, analytics dashboard.

## Features
Prediction form with client and server validation; result page with gauge, risk bar, positive/negative factors and recommendations; analytics dashboard (Chart.js); SQLite history; About & Model page with comparison tables and plots; JSON API `POST /api/predict`.

## Technologies
Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn, joblib, Flask, SQLite, Bootstrap 5, Chart.js.

## Architecture
Offline training (`src/train_model.py`) saves full sklearn Pipelines (preprocessing + model) to `models/`. Flask loads them, validates input (`src/validation.py`), predicts and explains (`src/prediction.py`), recommends (`src/recommendations.py`) and stores results via `database/database.py` (parameterised SQL only).

## ML methodology
Clean (duplicates, clipping impossible values, median/mode imputation inside the Pipeline) -> feature engineering (`assignment_completion_rate`, `academic_consistency`, `engagement_index`) -> 80/20 split (stratified) -> scale/one-hot encode -> compare models with 5-fold CV -> grid-search tuning -> test evaluation -> permutation importance. Regression models: Linear, Decision Tree, Random Forest, Gradient Boosting (best chosen by CV RMSE). Classifiers: Logistic Regression, Random Forest, Gradient Boosting (best by CV macro-F1). **Leakage prevention:** `final_exam_score`, `final_percentage`, `performance_category`, `student_id` are never inputs (asserted in code).
Per-prediction explanation: each input is swapped for the training median/mode and the change in predicted marks is reported (model-agnostic). SHAP can be added later in `Predictor.explain`.

## Dataset
`dataset/student_performance.csv` is **synthetic** (default 3,000 rows, ~2% missing values in some columns) with correlated variables driven by a latent ability factor. To use real data, export anonymised records with the same column names (no names or direct identifiers) to that path and retrain. Features: all columns in `config.py::RANGES` plus gender, department, internet_access. Targets: `final_exam_score` (regression), `performance_category` (classification, from `final_percentage = 0.7*exam + 0.3*internal`).

## Install and run
```bash
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m src.generate_dataset 3000     # generate dataset (any size)
python -m src.train_model               # train, compare, save models, metrics, figures
python app.py                           # open http://127.0.0.1:5000
python -m pytest tests                  # run tests
```
Pretrained models and a dataset are already included; re-run the two commands to regenerate.

## Results (synthetic test set, 600 records)
See `models/metrics.json` and the About & Model page. Linear Regression won on cross-validated RMSE (R² ≈ 0.88, MAE ≈ 3.7 marks), which is expected because the synthetic relationships are mostly linear. Gradient Boosting was best classifier (accuracy ≈ 0.81, macro-F1 ≈ 0.81, ROC-AUC ≈ 0.96). These numbers describe synthetic data only and will differ on real data.

## Screenshots
`[placeholder: home]  [placeholder: prediction form]  [placeholder: result]  [placeholder: dashboard]`

## Limitations and ethics
Synthetic data; estimates can be wrong; potential bias and fairness issues across groups; privacy and consent needed for real student data; avoid over-reliance and never use as a permanent label. Keep a human in the loop.

## Common errors
`ModuleNotFoundError: config` -> run from the project root. `FileNotFoundError models/...` -> run `python -m src.train_model`. Charts blank -> the dashboard loads Chart.js/Bootstrap from a CDN, so it needs internet.

## Future scope
Deep learning, SHAP, LMS/college integration, early-warning notifications, mobile app, learning paths, MySQL/PostgreSQL.

## License
MIT
