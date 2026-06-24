# Speaker Notes — 04 Regression Models

> **The point of this notebook:** climb a model ladder of rising capacity on the *same* 29
> engineered features and the *same* time-based split, so every improvement is attributable
> to the model, not the data. Then read what the best model learned.

---

## Setup

- **Why load features from `feature_lists.json` instead of re-listing them.** The feature set
  is defined once in nb 03 and serialized. Reading it here guarantees nb 04 trains on exactly
  what nb 03 built — no copy-paste drift. Change the features in one place, the whole pipeline
  follows.
- **Why the ±inf / NaN cleanup on load.** The engineered CSVs can carry stray ratio/log edge
  cases. We scrub ±inf→NaN and median-fill *after reading*, so this notebook is robust no
  matter what's in the file. We print the residual count (0/0) as proof. (Tree models tolerate
  NaN, but LinearRegression doesn't — so we clean unconditionally.)
- **Why `try/except` around the XGBoost import.** On macOS XGBoost needs `libomp`. Catching a
  failed import lets the rest of the pipeline run (and fall back to Random Forest as "best")
  instead of crashing the whole notebook on a missing system library.
- **The `fit_eval` helper.** One function that fits, predicts, scores (MAE/RMSE/R²), saves the
  model, and prints a line. Why: uniform treatment of every model — no chance of scoring two
  models differently by accident.

---

## 1. Linear Regression (baseline + engineered)

- **Why include both LR variants here too.** It re-anchors the ladder. Baseline LR (779.8 s,
  7 raw features) and engineered LR (~706 s, 29 features) are the "no real model capacity"
  rungs. Everything below has to beat 706 s to justify its added complexity.

---

## 2. Decision Tree

- **Why a single tree, depth-capped at 10.** It's the simplest *non-linear* model — it can
  capture interactions linear regression can't (e.g. high load *and* late hour). The depth
  cap prevents it from memorising the training set. It lands ~706 s — basically tied with
  engineered LR, which tells us a *single* tree's variance eats its non-linear advantage.
  That motivates ensembling.

---

## 3. Random Forest

- **Why bag many trees (120, depth 18).** Averaging many decorrelated deep trees cuts the
  variance that hurt the single tree, while keeping the non-linearity. MAE drops to ~678 s —
  the first clear step below the linear models. The cost is interpretability and model size.

---

## 4. XGBoost

- **Why boosting after bagging.** Gradient boosting fits trees sequentially to each other's
  residuals — typically the strongest tabular model. It wins the ladder at **~652 s** MAE,
  R² ~0.34.
- **Why this specific, *light* config** (`max_depth=7`, `learning_rate=0.05`,
  `n_estimators=600`, `subsample=0.9`, `colsample_bytree=0.9`, `min_child_weight=5`). The
  target is noisy, so we regularise deliberately: a moderate depth, a slow learning rate with
  many trees (smooth fitting), row/column subsampling and a min-child-weight floor to resist
  overfitting. We're buying generalisation, not chasing the training score.

---

## 5. Final model comparison

- **Why one consolidated table → `model_metrics.csv`.** Side-by-side MAE/RMSE/R² makes the
  ladder legible and feeds the report. The narrative it supports: features bought ~73 s
  (nb 03), model capacity bought another ~54 s on top — both matter, neither alone is enough.

---

## 6. Feature importance

- **Why inspect importance on the winning model.** This is where the project's headline
  insight surfaces: **`orders_per_dasher` is the single most important feature** — marketplace
  *load* outranks the platform's own `estimated_*_duration` fields and everything about the
  basket. The business takeaway we voice here: to improve ETAs, invest in modelling live
  supply/demand, not order details. We save the ranking to CSV for the report.
- **Caveat to state aloud.** Tree "gain" importance is directional, not causal — it tells us
  what the model leaned on, which nb 05's ablation then stress-tests by actually removing
  feature groups.

---

## Hand-off

- **Why save `test_predictions.csv` with every model's predictions.** nb 05 does error
  analysis on the *best* model's predictions. Persisting all of them (plus actuals and a few
  slice columns like `market_id`, `store_primary_category`) means the error analysis can run
  standalone without retraining anything.

---

## Closing narration

The ladder did its job: each rung beat the last on a fixed yardstick, XGBoost won, and the
importance ranking handed us a business story (load > basket). But a single MAE hides *where*
the model fails — which is exactly what nb 05 exists to expose.
