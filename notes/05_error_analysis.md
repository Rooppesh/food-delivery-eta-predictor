# Speaker Notes — 05 Error Analysis

> **The point of this notebook:** a single MAE is a summary, not an understanding. Here we ask
> *where*, *when*, and *for whom* the best model is wrong — because that's what tells you
> whether to ship it, recalibrate it, or collect new data. We analyse the winning model's
> predictions (XGBoost, MAE ~652 s) loaded from `test_predictions.csv`.

---

## Pick best model column

- **Why re-select the best model from the saved predictions.** The notebook stays decoupled
  from nb 04 — it reads `test_predictions.csv`, recomputes MAE per model column, and picks the
  winner programmatically (`pred_XGBoost`). No retraining, no hard-coded model name to go stale.

---

## 1. Residual analysis

- **Why signed residuals (`actual − predicted`), not just absolute error.** Direction is the
  business signal. A **positive** residual means we *under*-predicted — we quoted too short
  and the food is late, the painful customer outcome. Negative means we over-quoted (mere
  slack). Lumping them into `|error|` would hide which way we fail.
- **Why the histogram + predicted-vs-actual scatter.** The histogram shows the residual shape
  (and any systematic bias — the mean residual is positive, i.e. we lean toward
  under-prediction). The scatter against the y=x line shows the model compressing toward the
  mean: it under-predicts the long deliveries and over-predicts the short ones — classic
  regression-to-the-mean, and a direct preview of the long-tail problem below.

---

## 2. Worst 100 predictions

- **Why zoom in on the absolute worst cases.** Aggregate metrics can't tell you if errors are
  uniform noise or a structured failure mode. Sorting by `abs_error` and reading the top 100
  shows they're **not random**: the worst misses are all huge *actual* durations
  (8,000–10,000 s) that the model pulled down toward ~2,000–3,000 s. Mean abs-error in the top
  100 is ~5,600 s — these few catastrophic under-predictions are where the pain concentrates.
- **Why the "what's distinctive" comparison.** Contrasting the worst-100's averages against
  the full test set confirms the pattern quantitatively: their mean actual duration (~8,600 s)
  is ~3× the overall mean (~2,960 s). The failure is "very long deliveries," full stop.

---

## 3. Segment MAE — by hour, day-of-week, market, store category

- **Why slice the error the same way EDA sliced the target.** To find *operationally
  actionable* weak spots, not just a global number. The findings:
  - **By hour:** hour 14 is worst (~1,170 s) — lunch-rush congestion the features don't fully
    capture.
  - **By day-of-week:** Monday (dow 0) leads — start-of-week dynamics.
  - **By market:** market 1 is worst (~765 s), and **rows with missing `market_id` are worst
    of all** (~806 s) — which retro-justifies the missingness flags from nb 03.
  - **By store category:** rare cuisines (lebanese, belgian, spanish…) top the error table —
    small sample, high prep-time variance. The honest caveat: thin segments are noisy, so we
    treat these as "needs more data" rather than "model is broken."
- **Why we fixed the pandas `groupby.apply` call** (`include_groups=False`). A deprecation
  warning was firing; the flag silences it and future-proofs the cell without changing results.

---

## 4. Long-tail performance

- **Why bucket by the same `<30 / 30–60 / >60 min` bands.** This quantifies the
  regression-to-the-mean story in business terms:
  - `<30 min`: MAE ~697 s
  - `30–60 min` (the dominant 67%): MAE ~450 s — the model is sharpest where most volume is.
  - `>60 min`: MAE **~1,193 s** — roughly **2.6× worse** than the middle band.
- **Why this is the most important slide.** The >60 min tail is both the *hardest to predict*
  and the *most expensive to get wrong* (a very late delivery is the worst customer
  experience). It's where extra signal — live traffic, real per-store prep times — would pay
  off most, and it's the natural next investment.

---

## Closing narration

The error analysis turns one number into a to-do list: the model is reliable in the common
30–60 min band, biased toward under-prediction, and weakest exactly where it hurts most — the
long tail, the lunch rush, rare cuisines, and missing-data rows. That's an honest account of
*when to trust it* and *where to improve it*, which is the real deliverable of the project.
