# Speaker Notes — 02 Baseline Linear Regression

> **The point of this notebook:** establish the rules of the game — the split, the metric,
> and a floor — and ship the simplest honest model. We are *not* trying to win here. We're
> trying to create a fair, reproducible yardstick that every later model is measured against.

---

## 1. Load, build target, time-based split

- **Why rebuild the target here** (not reuse EDA). Each notebook is self-contained, so we
  recompute `delivery_duration = actual_delivery_time − created_at` from the raw CSV. No
  hidden state carried between notebooks.
- **Why the outlier filter (60 s ≤ ETA ≤ 3 h).** Straight out of EDA: the impossible max of
  8.5M s and sub-minute deliveries are data-quality artifacts. Keeping them would let a
  handful of rows dominate MAE and poison every model. We drop the target-null rows too —
  no target, no training signal.
- **Why sort by time, then cut 80/20.** This is the crucial move. We sort by `created_at`
  and take the first 80% as train, last 20% as test. The test set is literally *the future*
  relative to train (train ends 2015-02-13, test runs 02-13 → 02-18). That mirrors
  deployment: we always predict forward in time. We print both date ranges so anyone can
  verify there's no overlap — no future leaking backward.

---

## 2. Select baseline features

- **Why only the 7 "given" numeric columns.** The baseline deliberately uses just the raw,
  always-present order fields (items, subtotal, distinct items, price min/max, the two
  platform duration estimates). No engineering yet. The whole purpose of a baseline is to
  show what the *raw* signal is worth, so nb 03's feature work has something to prove value
  against.

---

## 3. Median imputation (computed on train only)

- **Why median, not mean.** Several of these are skewed (subtotal, prices); the median is
  robust to the long right tail we saw in EDA.
- **Why "train only" is non-negotiable.** We learn the median from the **training** slice and
  apply those same values to test. If we computed medians over the full dataset, test-set
  statistics would bleed into training — a subtle leak that inflates the score. Same
  discipline we'll repeat for every encoder and transform downstream.

### The mean-prediction floor
- **Why we compute a "dumb" baseline.** `mean_pred = train mean for every row` gives
  **MAE = 834.8 s**. This is the floor: a model that predicts the average and ignores all
  features. Any model that can't beat 834.8 s has learned nothing. It reframes every later
  result — the engineered linear model at ~706 s and XGBoost at ~652 s are "X seconds better
  than knowing nothing," which is a far more honest claim than a raw MAE in isolation.

---

## 4. Train Linear Regression

- **Why linear regression first.** It's transparent, fast, and has no hyperparameters to
  fuss over — the perfect reference point. Baseline MAE lands at **779.8 s**: better than the
  834.8 s floor (so the raw features *do* carry some signal), but only by ~55 s (so there's a
  lot of headroom — which is the motivation for nbs 03–04).
- **Why report MAE, RMSE, and R² together but lead with MAE.** MAE is our headline because
  it's directly interpretable in seconds ("off by ~13 min on average") and treats over- and
  under-prediction symmetrically. RMSE rides along to flag sensitivity to big misses; R²
  contextualises how much variance we explain (a thin 0.09 here — honest about how hard this
  is with raw features). Accuracy is meaningless on a continuous target, so we don't pretend.

---

## 5. Coefficient interpretation

- **Why bother reading coefficients on a baseline.** Two reasons. (1) Sanity check: the signs
  and magnitudes should make sense — `num_distinct_items` carrying the largest positive
  weight (~30 s per distinct item) is plausible (more variety = more prep). (2) It sets a
  baseline mental model of which raw fields matter, so when nb 04's feature importance later
  crowns `orders_per_dasher`, we can say "the *engineered* signal dwarfs the raw fields" with
  evidence.

---

## Closing narration

This notebook's deliverables aren't really the model — they're the **contract**: a
time-based split saved to `train.csv` / `test.csv`, a metric everyone agreed on, and a
834.8 s floor. Everything downstream reuses that split and is judged against that floor.
