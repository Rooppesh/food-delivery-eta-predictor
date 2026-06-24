# Speaker Notes — 03 Feature Engineering

> **The point of this notebook:** turn weak raw columns into features that encode *domain
> knowledge* — time rhythms, marketplace load, price structure — and prove they help by
> re-running the same linear model on the same split. Same yardstick, better inputs.

---

## 1. Time features

- **Why these four (`hour`, `day_of_week`, `month`, `is_weekend`).** EDA showed clear diurnal
  and weekly patterns in ETA. A raw timestamp is useless to a model; decomposing it exposes
  the cyclical demand structure (lunch/dinner rushes, weekend behaviour) as columns a tree
  can split on.
- **Why a quick weekend-vs-weekday table.** A sanity check that the feature carries signal
  before we commit to it — weekends do run a bit longer (~2,903 s vs ~2,788 s mean).

---

## 2. Marketplace features

- **Why ratios instead of raw counts.** `busy_ratio = busy / onshift` and
  `orders_per_dasher = outstanding / onshift` capture **supply-vs-demand**, which is what
  actually drives delays. 20 busy dashers is great with 100 on shift and a disaster with 22.
  The ratio encodes that; the raw count can't.
- **Why replace 0 on-shift with NaN before dividing.** Straight from the EDA landmine: ~1.8%
  of rows have zero on-shift dashers, which would make the ratio `inf`. We convert 0→NaN
  first, divide, then fill the resulting NaN with the column median. This is the clean way to
  say "we don't know the load here" rather than emitting infinities.
- **Why the busy_ratio decile plot.** Confirms the feature is monotonic-ish with ETA before
  we trust it — higher load, longer deliveries.

---

## 3. Price features

- **Why `avg_item_price` and `price_range`.** `subtotal / total_items` is a proxy for cuisine
  tier / order type; `max − min` price hints at basket heterogeneity. Both are cheap signals
  that raw columns don't express directly. We guard the division (`total_items` 0→NaN) the
  same way as the marketplace ratios.

---

## 4. Order-complexity features

- **Why `items_per_distinct_item`.** Distinguishes "10 of the same drink" (fast) from "10
  different dishes" (slow prep). It's a complexity signal orthogonal to raw counts. We fill
  the degenerate case (0 distinct) with 1.0.

---

## 5. Categorical encoding

- **Why frequency encoding, not one-hot.** `store_primary_category` has dozens of values;
  one-hot would explode the feature space and create sparse, noisy columns for rare cuisines.
  Frequency encoding maps each category to how often it appears — one dense column that
  preserves the common-vs-rare distinction trees can exploit.
- **Why fit the encoder on train only and `fillna('missing')`.** Same anti-leakage rule:
  frequencies are learned from train. Unseen test categories map to 0.0. NA becomes an
  explicit `'missing'` bucket rather than silently dropping rows.

---

## 6. Missingness as signal & log transforms

**Two ideas in one section, both about respecting the data's structure.**

- **Why missingness flags.** EDA proved the dasher columns are NA *together* on 16,262 rows —
  the marketplace snapshot failed, which itself may correlate with unusual conditions. So
  before we impute those values away, we stamp `*_is_missing` flags (for the three dasher
  columns + `store_primary_category`). We let the model decide whether "we didn't have load
  data" is predictive, instead of erasing that fact. **Missingness is information, not just
  absence.**
- **Why log-transform `subtotal`, `min_item_price`, `max_item_price`.** These are
  right-skewed (long tail of expensive orders). `log1p` compresses the tail so a *linear*
  model sees a roughly linear relationship instead of being dragged by outliers. We keep the
  originals too — tree models are invariant to monotonic transforms, so this only helps the
  linear side and costs the trees nothing.
- **Why `clip(lower=0)` before `log1p`.** A subtle but important guard: a few price values
  are below 0 (data noise), and `log1p(x < −1)` is undefined → `NaN`/warnings. Clipping
  negatives to 0 keeps us in-domain. Prices can't truly be negative, so this is also the
  *correct* fix, not just a silencer.

---

## 7. Rebuild linear regression with engineered features

- **Why re-run the exact same model.** Holding the model and split fixed isolates the
  contribution of the *features*. Engineered LR drops MAE from **779.8 → ~706 s** — a real,
  attributable ~73 s win from features alone, and R² roughly triples (0.09 → 0.26).
- **Why the ±inf / NaN safety fill right before fitting.** Belt-and-suspenders: even after
  careful guarding, a ratio or log edge case could slip through, and LinearRegression refuses
  NaN/inf. We replace ±inf→NaN and fill with train medians so the fit can never crash on a
  stray value. We print the remaining NaN/inf count (0/0) to prove it.
- **Why persist `train_eng.csv`, `test_eng.csv`, and `feature_lists.json`.** This is the
  hand-off to nb 04. `feature_lists.json` is the **single source of truth** for the 29-feature
  set — nb 04 and nb 05 read it rather than re-listing features, so the feature definition
  lives in exactly one place and can't drift between notebooks.

---

## Closing narration

We converted weak raw columns into domain-aware features and *proved* the lift on a fixed
yardstick. The headline isn't the number — it's that load and time features (not basket
details) did the heavy lifting, foreshadowing what nb 04's importance ranking confirms.
