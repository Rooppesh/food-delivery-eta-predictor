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
### Residuals
- Mean residual ≈ +78s (slight under-prediction)
- Median residual ≈ −63s (model slightly over-predicts the median order)
- Std ≈ 906s — wide tails.

### Long-tail breakdown
| Bucket | n | MAE (s) |
|---|---:|---:|
| < 30 min | 4,399 | 707 |
| 30–60 min | 25,789 | 455 |
| > 60 min | 9,269 | 1,184 |

Model is sharpest in the dominant 30–60 min band and degrades hard on > 60 min orders —
typical regression-toward-the-mean behaviour amplified by sparse training signal in the tail.

### Worst segments
- **By hour**: 2pm–4pm is the worst (MAE up to ~1,211s at 14h) — peak lunch-wave congestion.
- **By store category**: rare cuisines (lebanese, belgian, spanish, tapas) — small sample,
  high variance in prep times.
- **By market**: market 1 worst (MAE 766s), market 2 best (MAE 591s); rows with missing
  market_id are also high-error (MAE 823s).

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
