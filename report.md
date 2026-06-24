# DoorDash ETA Prediction — Report

## Executive Summary
We built a model to predict total delivery duration (seconds from order creation to drop-off)
on ~197k historical DoorDash orders. After feature engineering and time-based evaluation,
the best model — XGBoost — reached **MAE ≈ 654.7s (~10.9 min)** and **R² ≈ 0.34** on
a held-out 20% time slice. The biggest gains came from (1) marketplace-load features
(`orders_per_dasher`, `busy_ratio`), (2) time-of-day signal (`hour`, `is_weekend`),
and (3) categorical encodings of `market_id` / `order_protocol` / `store_primary_category`.

## Dataset Overview
| | |
|---|---|
| Rows (raw) | 197,428 |
| Rows (after clean: 60s ≤ ETA ≤ 3h, target non-null) | 197,283 |
| Raw columns | 16 |
| Engineered features | 22 |
| Target | `delivery_duration = actual_delivery_time − created_at` (seconds) |
| Split | Time-based, 80% train (157,826) / 20% test (39,457) |

## EDA — Key Findings
- Target is right-skewed; most deliveries fall in 30–60 min, with a long >60 min tail.
- Missingness is concentrated in the dasher columns (`total_*_dashers`) and `store_primary_category`.
- Strongest single-feature correlate of ETA is `estimated_store_to_consumer_driving_duration`;
  marketplace load (`total_outstanding_orders`, busy/onshift ratio) is close behind.
- Outlier policy: drop rows with ETA < 60s or > 3h — clearly data-quality artefacts.

## Feature Engineering
| Family | Features | Motivation |
|---|---|---|
| Time | `hour`, `day_of_week`, `month`, `is_weekend` | Diurnal & weekly demand patterns |
| Marketplace | `busy_ratio = busy/onshift`, `orders_per_dasher = outstanding/onshift` | Capture supply/demand imbalance |
| Price | `avg_item_price = subtotal/total_items`, `price_range = max − min` | Proxy for cuisine / order complexity |
| Complexity | `items_per_distinct_item` | Bulk-of-same-item vs varied basket |
| Categorical | Frequency-encoded `market_id`, `order_protocol`, `store_primary_category` | Avoid blow-up from one-hot; preserves rare-vs-common signal |

## Modelling — Test-set metrics
| Model | MAE (s) | RMSE (s) | R² |
|---|---:|---:|---:|
| Linear Regression (baseline, 7 feats) | 779.8 | 1061.8 | 0.09 |
| Linear Regression (engineered, 22 feats) | 705.9 | 961.6 | 0.26 |
| Decision Tree (depth=10) | 708.4 | 990.7 | 0.21 |
| Random Forest (120 trees, depth=18) | 677.5 | 934.1 | 0.30 |
| **XGBoost (600 trees, lr=0.05, depth=7)** | **654.7** | **909.0** | **0.34** |

Feature engineering alone cut linear-regression MAE by ~74s (~9%). Moving from linear
to XGBoost cut another ~51s on top.

### Top 10 features (XGBoost gain)
1. `orders_per_dasher` (0.17)
2. `hour` (0.10)
3. `estimated_store_to_consumer_driving_duration` (0.08)
4. `estimated_order_place_duration` (0.08)
5. `is_weekend` (0.07)
6. `month` (0.06)
7. `market_id_freq` (0.05)
8. `subtotal` (0.05)
9. `day_of_week` (0.05)
10. `order_protocol_freq` (0.04)

The marketplace-load feature `orders_per_dasher` is the single biggest predictor — more
than the platform's own duration estimates.

## Error Analysis

Notebook `05_error_analysis.ipynb` follows the 7-section framework from
[*AI/ML Learning Notes — Day 23: Model Error Analysis*](https://jaspinders30.substack.com/p/ai-ml-learning-notes-day-23-model),
adapted for regression (FP/FN → signed residuals; threshold sweep → ETA-quote slack δ; ROC-AUC vs decision metrics → Spearman ρ vs MAE).

### Error decomposition (signed)
| Direction | Share | Mean \|err\| (s) | p90 \|err\| (s) |
|---|---:|---:|---:|
| Under-predicted (actual > quote — costly) | 47% | 787 | 1,732 |
| Over-predicted (actual < quote — slack) | 53% | 532 | 1,042 |

Under-predictions are fewer but **~50% larger** on average — the model is more wrong in the more painful direction.

### Magnitude concentration
| \|error\| bucket (s) | Share of rows | Share of total |error| |
|---|---:|---:|
| 0–300 | 31% | 7% |
| 300–600 | 27% | 18% |
| 600–1,200 | 29% | 37% |
| 1,200–3,000 | 12% | 31% |
| 3,000+ | 1% | 7% |

The top 13% of orders by error account for **~37%** of total absolute error — that's where there's leverage.

### Long-tail breakdown
| Bucket | n | MAE (s) |
|---|---:|---:|
| < 30 min | 4,399 | 699 |
| 30–60 min | 25,789 | 450 |
| > 60 min | 9,269 | 1,193 |

Model is sharpest in the dominant 30–60 min band and degrades ~2.6× on > 60 min orders.

### Worst segments
- **By hour**: 14h–16h is the worst (MAE 1,188 s at 14h, 940 s at 15h) — lunch-rush congestion.
- **By store category**: rare cuisines (lebanese, belgian, spanish, …) — small sample, high prep-time variance.
- **By market**: market 1 worst (MAE 766 s), market 2 best (MAE 591 s); rows with missing market_id also high-error (821 s).

### Feature ablation (XGBoost, ΔMAE on test)
| Group removed | n features kept | MAE (s) | ΔMAE vs full |
|---|---:|---:|---:|
| marketplace | 17 | 697.0 | **+44.7** |
| platform estimates | 20 | 682.6 | +30.3 |
| time | 18 | 671.7 | +19.3 |
| categorical (freq) | 19 | 667.4 | +15.0 |
| price | 17 | 660.2 | +7.9 |
| **order composition** | 19 | 652.1 | **−0.2** |

`marketplace` is the heaviest hitter; `order composition` is **redundant** — removing it slightly improves the model. That's exactly the article's red-flag pattern: candidate for retirement.

### Ranking vs decision (metric separation)
| Metric | Value |
|---|---:|
| MAE (s) | 652.1 |
| RMSE (s) | 909.3 |
| Spearman ρ | 0.62 |
| **Mean residual (s)** | **+89.6** |
| Median residual (s) | −53.2 |

The model **ranks** deliveries reasonably (ρ = 0.62) but systematically **under-predicts** by ~90 s. A constant `+90` recalibration is a free win on the decision metric without retraining.

### Cost-aware ETA quote (decision-threshold analogue)
Sweeping an additive slack `δ` on the quote and minimising `C_under·P(under) + C_over·P(over)`:

| Cost ratio (under : over) | Optimal δ (s) |
|---|---:|
| 1 : 1 (symmetric) | ≈ 0 |
| 3 : 1 | ~+200 |
| 5 : 1 (article default) | ~+400 |

Realistic asymmetry pushes the customer-facing quote a few minutes higher than the point estimate.

## Classification framing — predicting the bucket

To answer *“is this delivery going to be late?”* directly, we also reframed the task as
a 3-class problem on the same buckets used in the long-tail analysis (`<30 / 30–60 / >60 min`).
Test-set class balance is 11% / 65% / 23% — imbalanced enough that **PR-AUC**, not
accuracy, is the right headline.

### PR-AUC by class (one-vs-rest)
| Model | <30min | 30-60min | **>60min** | macro | weighted | ROC-AUC (macro) |
|---|---:|---:|---:|---:|---:|---:|
| LogisticRegression | 0.367 | 0.752 | 0.503 | 0.541 | 0.651 | 0.745 |
| RandomForestClassifier | 0.391 | 0.744 | 0.521 | 0.552 | 0.652 | 0.751 |
| **XGBClassifier** | **0.417** | **0.766** | **0.555** | **0.580** | **0.678** | **0.774** |

The `>60min` class is the operationally critical one (an undetected late delivery is the
worst customer outcome). XGBClassifier's PR-AUC on it (0.555) is ~5× the random-baseline
prevalence (~0.23), and meaningfully above the other two classifiers.

### Is a dedicated classifier worth it vs binning the regressor?
Take the XGBoost **regressor**'s seconds output and bucket it. Result:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| <30min | 0.615 | 0.120 | 0.201 | 4,399 |
| 30-60min | 0.705 | 0.913 | 0.795 | 25,789 |
| **>60min** | **0.628** | **0.351** | **0.450** | **9,269** |
| accuracy | | | **0.693** | 39,457 |

The regressor-then-bin baseline only catches **35% of >60min deliveries** — most long
deliveries get pulled toward the dominant 30-60min bucket (classic regression-toward-
the-mean). The dedicated classifier is the right tool when the *signal* you want is the
late-delivery probability — and it gives you a tunable score, not just a hard label, so
you can choose your operating point on the PR curve.

**Recommendation:** keep XGBoost regression as the primary model (customer-facing ETA
quote needs seconds), and use the XGBoost classifier's `P(>60 min)` as a parallel
risk-flag for dispatcher / operations dashboards.

## Conclusions
- **Marketplace state matters more than order details.** The strongest features were
  marketplace load and time-of-day, not order composition.
- **The platform's own estimates (`estimated_*_duration`) are useful but not dominant** —
  the model meaningfully improves on them by conditioning on live load.
- **Long-tail (>60 min) deliveries are the hardest** and would benefit most from extra
  signal (live traffic, real per-store prep times).

### Lessons learned
- Time-based splits are essential here — random splits leak hour-of-day seasonality.
- Frequency encoding for categoricals worked well and avoided high-cardinality blow-up.
- Tree models gained more from engineered features than linear models, because the
  ratios (`busy_ratio`, `orders_per_dasher`) interact non-linearly with everything else.

### Future improvements
- Store-level historical-average ETA (target encoding with time-aware folds).
- Rolling marketplace-load features (1h, 24h windows).
- SHAP analysis + partial-dependence plots for stakeholder explainability.
- Quantile regression (p50 / p90) — DoorDash actually cares about the upper-bound ETA
  it quotes the customer, not just the point estimate.
- Deployment: persist the XGBoost model + a thin FastAPI / Streamlit prediction service.
