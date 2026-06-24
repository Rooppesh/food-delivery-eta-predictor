# food-delivery-eta-predictor

Predict DoorDash delivery ETA (seconds from order creation to drop-off) from historical
order, store, and marketplace features.

## Problem
Given an order at time *t*, predict `delivery_duration = actual_delivery_time - created_at`.
Success metric: minimise **MAE** on a time-based held-out test set.

## Dataset
DoorDash ETA prediction dataset (Kaggle: `dharun4772/doordash-eta-prediction`).
~197k orders, 16 raw columns including order composition, store, and live marketplace
load (busy / on-shift dashers, outstanding orders).

Place the file at `data/historical_data.csv` before running anything.

## Project structure
```
food-delivery-eta-predictor/
├── notebooks/        # 01_eda → 05_error_analysis  (self-contained, no src/)
├── data/             # historical_data.csv
├── models/           # trained .joblib files (created by notebook 04)
├── reports/          # intermediate train/test parquets + metrics + predictions
├── requirements.txt
├── README.md
└── report.md
```

All logic — preprocessing, feature engineering, training, evaluation — lives directly
in the notebooks. There is no `src/` package to import from.

The project covers two framings: **regression** (predict delivery seconds — primary)
and **classification** (predict the `<30 / 30–60 / >60 min` bucket, evaluated with
PR-AUC — diagnostic / risk-flagging).

## Setup
```bash
pip install -r requirements.txt
```
On macOS, XGBoost also needs `brew install libomp`.

## Run the notebooks (in order)
```bash
jupyter notebook notebooks/
```
1. `01_eda.ipynb` — load, target, missingness, distributions, correlations.
2. `02_baseline_model.ipynb` — time-based split, baseline Linear Regression.
3. `03_feature_engineering.ipynb` — time / marketplace / price / complexity / categorical features.
4. `04_advanced_models.ipynb` — Decision Tree, Random Forest, XGBoost + feature importance.
5. `05_error_analysis.ipynb` — residuals, worst predictions, segment MAE, long-tail.
6. `06_classification.ipynb` — 3-bucket classifier (`<30 / 30–60 / >60 min`),
   PR-AUC headline + regress-then-bin comparison. Depends on 03 and 04.

Notebooks 02 → 06 read intermediate CSVs from `reports/`, so run them in order the
first time.

## Results summary
See [report.md](report.md) for the full write-up and metric tables.
