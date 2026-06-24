# Speaker Notes — 01 Exploratory Data Analysis

> **The point of this notebook:** understand the data well enough that none of our later
> modelling choices are guesses. Every check here exists to *feed a downstream decision* —
> we call out which one as we go. We deliberately do EDA *before* building the target or
> splitting, so we look at the data with fresh eyes.

---

## 1. Sanity checks

**Why we start here, not with plots.** Before any distribution is interesting, we need to
trust the table: right grain, no silent duplication, and a clear picture of what's missing.

- **Grain & shape.** One row = one order at creation time, 197,428 rows × 16 columns. We
  state the grain out loud because every later aggregation (per-hour, per-market) assumes it.
- **`df.info()`** is our missingness preview and dtype check in one — note `created_at` /
  `actual_delivery_time` parsed as datetimes (we need that for the target and for time
  features), and the dasher columns sitting at ~181k non-null versus 197k total.

### Duplicates
- **Why check a composite key when there's no ID column.** This dataset has no real primary
  key, so we use `(store_id, created_at)` as the best-available identity. **0 exact
  duplicate rows** (good — no copy-paste ingestion bug) but **173 duplicate
  `(store_id, created_at)` keys**. We *don't* drop these: a busy store genuinely can take
  two orders in the same second. We flag it so we're not surprised later, and move on.

### Missingness
- **Why a percentage table, sorted.** It tells us where to spend effort. The story is
  concentrated: the three dasher columns are each **8.24%** missing,
  `store_primary_category` **2.41%**, and everything else is a rounding error.

### Joint missingness
- **This is the most important cell in the section.** The three dasher columns aren't
  missing independently — they're NA *together* on **16,262 rows** and present together on
  the other 181,166. That single fact drives two later decisions: (1) it's **one**
  missingness mechanism (the marketplace snapshot failed), so **one** `*_is_missing` flag
  captures it, not three; (2) it justifies imputing them as a group.

### Alternate missing indicators
- **Why eyeball `store_primary_category` value counts.** Missingness sometimes hides inside
  a category rather than as NaN. We confirm there's an explicit `"other"` bucket (3,988) and
  a separate `NaN` (4,760) — they mean different things, so we keep them distinct rather
  than merging.

---

## 2. Distributions & outliers

**Why build the target here.** `delivery_duration = actual_delivery_time − created_at` in
seconds. We compute it inside EDA so we can inspect it before it ever touches a model.

- **`describe()` is alarming on purpose.** Median 2,660 s (~44 min) but **max 8,516,859 s**
  (~98 days) and a std of 19,229. That impossible max is the whole reason the baseline
  notebook clips to 60 s ≤ ETA ≤ 3 h — we're seeing the data-quality artifacts that justify
  that rule.
- **Extended quantiles** make the cut defensible: the 99.9th percentile is ~9,935 s but the
  max is 8.5M. The damage is in the top 0.1%, so a 3-hour ceiling removes artifacts without
  touching real long deliveries.
- **Zeros in the marketplace counters.** ~1.8–2.1% of rows have **zero** on-shift / busy
  dashers. This is *the* reason `busy_ratio = busy / onshift` must guard against
  divide-by-zero in nb 03 — we found the landmine here so we can defuse it there.
- **Target histogram + boxplot** (clipped to 10k s for legibility) show the right-skew we
  expect for durations — a fat right tail. That skew is why we'll later add log-transformed
  price features and why MAE (robust) is a saner objective than something that chases the tail.
- **Feature distributions + correlation heatmap.** The linear correlations with the target
  are *weak* (`total_outstanding_orders` highest at 0.12). The honest takeaway we voice here:
  no single raw feature is a silver bullet → value will come from **engineered ratios** and
  **non-linear models**, which is exactly the path nbs 03–04 take.

---

## 3. Slice analysis

**Why slice instead of trusting the global average.** Averages hide structure (Simpson's
paradox). We look at the target across the dimensions we'll actually model on.

- **Bucket preview (13.2% / 67.0% / 19.9%).** We pre-compute the `<30 / 30–60 / >60 min`
  split that a classification framing would use. The imbalance is the point: a 67%-majority
  class means **accuracy is a trap** and PR-AUC would be the right headline if we go that route.
- **By `market_id` / `order_protocol`.** Means do move across markets (market 1 ~3,312 s vs
  market 2 ~2,764 s) and protocols — evidence these categoricals carry signal worth encoding.
- **By store category.** Italian (~4,212 s) sits far above fast food (~2,631 s) — prep time
  is real and category-dependent. We filter to n ≥ 100 so we don't over-read tiny cuisines
  (a caveat that returns in error analysis, where rare cuisines top the error table).
- **By hour / day-of-week.** Clear diurnal and weekly rhythm → justifies the time features
  in nb 03. **Important caveat to say aloud:** the hour-8 mean (~203,284 s) is an *outlier
  artifact*, not a real lunch effect — a handful of absurd-duration rows dominate a thin
  bucket. It's a reminder that we must clip outliers before trusting any hourly average.

---

## 4. Leakage & time discipline

**Why this section guards the whole project.**

- **Time span: 122 days** (Oct 2014 → Feb 2015), contiguous. Because the data is a time
  series of orders, a random train/test split would let the model learn from future days to
  predict past ones — leaking seasonality. So every downstream split is **time-based**: train
  on the earlier portion, test on the most recent. This single decision is the backbone of
  honest evaluation in nbs 02–05.
- **Leakage audit mindset.** `actual_delivery_time` is post-outcome (it *is* the target, can
  never be a feature). The dasher counts and `estimated_*_duration` are snapshots/estimates
  available at order-creation time, so they're fair game. We reason about each column's
  availability *at prediction time*, not just whether it correlates.

---

## Closing narration

EDA didn't just describe the data — it pre-decided our pipeline: clip outliers to 3 h, split
on time, treat dasher-missingness as one signal, lean on engineered ratios over weak raw
correlations, and watch the >60 min tail. Everything after this is executing on what we
learned here.
