# Disaggregating Data Center Load Growth in AEP Ohio via Residual Analysis

**EE P 555 — Winter 2026 Final Project**  
Haoran Xu · Holly Wang · Zixiang Xu

---

## Project Overview

AEP Ohio (Columbus area) is experiencing a massive surge in data center construction (AWS, Google, etc.). Traditional load forecasting models assume electricity demand is primarily driven by weather. However, data centers introduce a large, flat "base load" that does not correlate linearly with weather.

This project **separates weather-sensitive load from structural load growth** caused by data center additions. By training models on 2021–2024 data and testing on 2025–2026, we quantify the "data center effect" through residual analysis.

---

## Repository Structure

```
555-Final-Project/
│
├── data/                          # Raw input data
│   ├── pjm_load/                  # PJM hourly metered load (AEP zone)
│   │   ├── 2021-2022.csv
│   │   ├── 2022-2023.csv
│   │   ├── 2023-2024.csv
│   │   ├── 2024-2025.csv
│   │   └── 2025-2026.csv
│   └── weather/                   # NOAA LCD hourly weather (Columbus, OH)
│       ├── LCD_USW00014821_2021.csv
│       ├── LCD_USW00014821_2022.csv
│       ├── LCD_USW00014821_2023.csv
│       ├── LCD_USW00014821_2024.csv
│       └── LCD_USW00014821_2025.csv
│
├── src/                           # Source code (run in order)
│   ├── data_prep.py               # Step 1: Data loading, merging, feature engineering
│   ├── eda.py                     # Step 2: Exploratory Data Analysis (14 plots)
│   ├── ridge_model.py             # Step 3: Ridge / Lasso / ElasticNet baselines
│   ├── xgboost_model.py           # Step 4: XGBoost with 2-stage hyperparameter tuning
│   ├── transformer_model.py       # Step 5: PatchTST (Transformer-based forecasting)
│   └── residual_analysis.py       # Step 6: Model comparison & data center effect analysis
│
├── output/                        # Generated outputs (created by pipeline)
│   ├── processed_data.csv         # Merged & feature-engineered dataset
│   ├── plots/                     # All visualization PNGs (34 plots)
│   ├── predictions/               # Model predictions CSVs + comparison table
│   └── models/                    # Saved model files (pkl, json, pt)
│
├── run.sh                         # One-command pipeline runner
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## Data Sources

| Dataset | Source | Records | Time Span | Resolution |
|---------|--------|---------|-----------|------------|
| PJM Metered Load (AEPOPT) | [PJM Data Miner 2](https://dataminer2.pjm.com/) | ~43,800 | Jan 2021 – Feb 2026 | Hourly |
| NOAA LCD Weather (KCMH) | [NOAA NCEI](https://www.ncei.noaa.gov/) | ~43,500 (FM-15) | Jan 2021 – Dec 2025 | Hourly |

**Weather station:** Columbus John Glenn International Airport (USW00014821)

---

## Features Engineered

| Category | Features |
|----------|----------|
| **Weather** | Temperature, Dew Point, Humidity, Wind Speed, Precipitation, Pressure |
| **Temporal** | Hour, Day of week, Month, Year, Day of year, Weekend flag, Holiday flag |
| **Cyclical encoding** | sin/cos transformations of hour, month, day of week |
| **Weather-derived** | CDD, HDD, Heat Index, Temp², Temp³, Wind Chill |
| **Rolling statistics** | 6h/12h/24h/48h/168h rolling mean & std of temperature and load |
| **Lag features** | Load at t-1, t-2, t-3, t-6, t-12, t-24, t-48, t-168, t-336 |
| **Differenced** | Load diff (1h, 24h), Temperature diff (1h, 24h) |

**Total features:** 60+

---

## Models

### 1. Ridge / Lasso / ElasticNet (Baseline)

- **Weather-only Ridge:** Trained without lag features to isolate concept drift (the "data center detector").
- **Full Ridge:** All features with L2 regularization.
- **Lasso:** L1 regularization for feature selection.
- **ElasticNet:** Combined L1+L2 with grid search over alpha & l1_ratio.
- **CV:** TimeSeriesSplit (5 folds), alpha sweep over 12 values.

### 2. XGBoost (Gradient Boosting)

- **2-stage hyperparameter search:**
  - Stage 1: Grid over n_estimators × max_depth × learning_rate (36 combos × 5-fold CV)
  - Stage 2: Grid over subsample × colsample × min_child_weight × reg_alpha × reg_lambda (540 combos × 5-fold CV)
- Total configurations evaluated: **576 × 5 = 2,880 model fits**
- Rolling window CV reported with mean ± std.

### 3. PatchTST (Transformer)

- **Architecture:** Patch-based Transformer encoder, aligned with Week 10 lecture topics.
  - Lookback window: 336 hours (2 weeks)
  - Patch length: 24 hours (1 day) → 14 patches
  - Model: 4-layer Transformer encoder, d_model=128, 8 attention heads
  - GELU activation, LayerNorm, mean pooling, AdamW optimizer
- **Training:** 80 epochs, cosine annealing with warmup, early stopping (patience=15)
- **Parameters:** ~300K+

---

## Evaluation Strategy

| Aspect | Detail |
|--------|--------|
| **Split** | Chronological — Train: Feb 2021 – Dec 2024, Test: Jan 2025 – Jan 2026 |
| **Primary metric** | RMSE (penalizes peak errors, critical for grid stability) |
| **Supporting metrics** | MAE, R², MAPE |
| **Cross-validation** | TimeSeriesSplit (5-fold, time-aware, no data leakage) |
| **Uncertainty** | CV results reported as mean ± std |

---

## Key Analysis: Data Center Effect

The core insight uses **residual analysis** of the weather-only Ridge model:

1. Train a Ridge model using **only weather + calendar features** (no load history).
2. This model captures the **normal weather-load relationship** from 2021–2024.
3. On the 2025 test set, the model **systematically under-predicts** — the residuals are persistently positive.
4. A **linear trend** in the residuals quantifies the structural load growth rate.
5. Monthly residual bars show the under-prediction growing from ~200 MW (Jan 2025) to ~1200 MW (Dec 2025).

This non-weather-driven load growth is consistent with **data center capacity additions** in the AEP Ohio footprint.

---

## Quick Start

### Prerequisites

- Python 3.10+
- ~8 GB RAM (XGBoost grid search can be memory-intensive)
- GPU optional (PatchTST benefits from CUDA but runs on CPU)

### Run Complete Pipeline

```bash
# Install dependencies + run all 6 steps
bash run.sh
```

### Run Individually

```bash
pip install -r requirements.txt

python src/data_prep.py          # Step 1: merge & feature engineer
python src/eda.py                # Step 2: EDA plots
python src/ridge_model.py        # Step 3: linear baselines
python src/xgboost_model.py      # Step 4: XGBoost
python src/transformer_model.py  # Step 5: PatchTST
python src/residual_analysis.py  # Step 6: comparison & residual analysis
```

### Output

All results go to `output/`:
- `output/plots/` — 34 PNG visualizations (numbered for ordering)
- `output/predictions/` — CSV files with predictions + model comparison table
- `output/models/` — Saved models (Ridge pkl, XGBoost json, PatchTST pt checkpoint)

---

## Plot Index

| # | Filename | Description |
|---|----------|-------------|
| 01 | `load_timeseries.png` | Full 5-year hourly load time series |
| 02 | `monthly_avg_by_year.png` | Monthly avg load overlay by year |
| 03 | `load_vs_temp.png` | Load–temperature scatter (U-shape) by year |
| 04 | `correlation_heatmap.png` | Feature correlation matrix |
| 05 | `hourly_pattern.png` | Avg load by hour of day |
| 06 | `weekly_pattern.png` | Avg load by day of week |
| 07 | `load_distribution.png` | Histogram of MW |
| 08 | `yearly_avg_load.png` | Annual avg load growth |
| 09 | `load_boxplot_yearly.png` | Box plot by year |
| 10 | `temp_dist_by_year.png` | Temperature distribution comparison |
| 11 | `load_heatmap_hour_month.png` | Load heatmap (hour × month) |
| 12 | `cdd_hdd_vs_load.png` | CDD/HDD vs load |
| 13 | `lag_scatter.png` | Lag autocorrelation scatter |
| 14 | `rolling_load_by_year.png` | Weekly rolling avg by year |
| 15 | `ridge_pred_vs_actual.png` | Ridge prediction overlay |
| 16 | `ridge_wx_residuals.png` | Weather-only Ridge residuals |
| 17 | `ridge_full_residuals.png` | Full Ridge residuals |
| 18 | `ridge_coefficients.png` | Top Ridge coefficients |
| 19 | `ridge_cv_curve.png` | CV RMSE vs alpha curve |
| 20 | `xgb_feature_importance.png` | XGBoost feature importance |
| 21 | `xgb_pred_vs_actual.png` | XGBoost prediction overlay |
| 22 | `xgb_residuals.png` | XGBoost residuals |
| 23 | `xgb_scatter_actual_pred.png` | XGBoost actual vs predicted scatter |
| 24 | `patchtst_loss.png` | PatchTST training curve |
| 25 | `patchtst_pred_vs_actual.png` | PatchTST prediction overlay |
| 26 | `patchtst_residuals.png` | PatchTST residuals |
| 27 | `patchtst_scatter.png` | PatchTST actual vs predicted scatter |
| 28 | `monthly_residual_bars.png` | Monthly avg residual bar chart |
| 29 | `residual_trend_line.png` | Residual trend (data center signal) |
| 30 | `residual_comparison_smooth.png` | Smoothed residual comparison |
| 31 | `full_residual_timeline.png` | Full-period residual timeline |
| 32 | `yearly_load_vs_residual.png` | Year-over-year load vs residual |
| 33 | `residual_distributions.png` | Residual histograms (all models) |
| 34 | `model_comparison_bars.png` | RMSE/MAE/R²/MAPE comparison chart |

---

## Team Responsibilities

| Member | Responsibilities |
|--------|-----------------|
| **Haoran Xu** | Data collection, preprocessing, feature engineering, PatchTST implementation |
| **Holly Wang** | EDA, Ridge/Lasso baselines, residual analysis, visualization |
| **Zixiang Xu** | XGBoost model & hyperparameter tuning, model evaluation, report write-up |

---

## References

- Weng, Y., Xie, L., & Rajagopal, R. (2023). *Data Science and Applications for Modern Power Systems.* Springer.
- Nie, Y., et al. (2023). "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers." *ICLR 2023.* (PatchTST)
- Chen, T. & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System." *KDD 2016.*
