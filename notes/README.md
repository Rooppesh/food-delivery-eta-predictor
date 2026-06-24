# Speaker Notes

Presenter narration for each notebook — what we did at each step and, more importantly,
**why**. Read each file alongside the matching notebook; the section headers line up 1:1
with the notebook's `##` sections so you can talk through it cell-by-cell.

## The one-line story

> **raw orders → EDA → time-based split → baseline → engineered features → model ladder → error analysis**

We predict `delivery_duration` (seconds from order creation to drop-off) for ~197k
historical DoorDash orders. The thread running through every decision is **honest
evaluation**: a time-based split that mimics deployment, a dumb baseline every model must
beat, and an error analysis that asks *where* and *why* we're wrong rather than just
reporting one headline number.

## How the notebooks chain

The notebooks are self-contained (no `src/` package). They hand off through CSVs in
`reports/`, so the order matters on a cold run:

| # | Notebook | Reads | Writes | One-line job |
|---|---|---|---|---|
| 01 | `01_eda.ipynb` | `data/historical_data.csv` | — | Understand the data before touching a model |
| 02 | `02_baseline_model.ipynb` | raw CSV | `train.csv`, `test.csv`, `metrics_baseline.csv` | Lock the split + a baseline floor |
| 03 | `03_feature_engineering.ipynb` | `train/test.csv` | `train_eng.csv`, `test_eng.csv`, `feature_lists.json` | Build features, prove they help |
| 04 | `04_regression_models.ipynb` | `*_eng.csv`, `feature_lists.json` | `model_metrics.csv`, `test_predictions.csv`, `models/*.joblib` | Climb the model ladder |
| 05 | `05_error_analysis.ipynb` | `test_predictions.csv` | — | Find *where* the best model breaks |

## Notes files

- [`01_eda.md`](01_eda.md) — exploratory data analysis
- [`02_baseline_model.md`](02_baseline_model.md) — baseline linear regression
- [`03_feature_engineering.md`](03_feature_engineering.md) — feature engineering
- [`04_regression_models.md`](04_regression_models.md) — tree models + XGBoost
- [`05_error_analysis.md`](05_error_analysis.md) — residual & segment analysis

## The recurring "whys" (themes to hit no matter which notebook you're presenting)

1. **Time, not randomness.** Orders carry strong time-of-day and day-of-week seasonality.
   A random split lets the model peek at the future; a time-based split forces it to
   generalise forward, which is what production actually requires.
2. **Always have a floor.** A model that can't beat "predict the average every time" isn't
   learning anything. We compute that floor (834.8 s MAE) before celebrating any model.
3. **MAE in seconds.** We optimise and report a metric a non-ML stakeholder can read: "off
   by ~11 minutes on average," not an abstract loss.
4. **Fit on train only.** Every imputation value, encoding, and median is learned from the
   training slice and applied to test — otherwise the test score is optimistic fiction.
5. **Marketplace state beats order details.** The single strongest predictor is
   `orders_per_dasher` (load), not anything about the basket — that's the project's
   headline business insight and it recurs from EDA through error analysis.
