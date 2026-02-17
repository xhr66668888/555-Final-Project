"""
SARIMAX baseline — classical, interpretable forecasting model.

Serves as a sanity-check baseline alongside Ridge, so that
the "residual growth / data-center effect" story is validated
across multiple model families (ML + classical statistical).

Aggregates hourly data to daily, fits SARIMAX with
weekly seasonality (s=7) and temperature as exogenous regressor.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import warnings
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE, 'output')
PLOT_DIR = os.path.join(OUT_DIR, 'plots')
PRED_DIR = os.path.join(OUT_DIR, 'predictions')
MODEL_DIR = os.path.join(OUT_DIR, 'models')
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ================================================================
# Load and aggregate to daily
# ================================================================
df = pd.read_csv(os.path.join(OUT_DIR, 'processed_data.csv'))
df['datetime'] = pd.to_datetime(df['datetime'])
print(f"loaded {len(df)} hourly rows")

daily = df.set_index('datetime').resample('D').agg({
    'mw': 'mean',
    'temp': 'mean',
    'humidity': 'mean',
    'cdd': 'sum',
    'hdd': 'sum',
}).dropna()

daily['dayofweek'] = daily.index.dayofweek
daily['is_weekend'] = (daily['dayofweek'] >= 5).astype(int)
daily['month'] = daily.index.month
daily['temp_sq'] = daily['temp'] ** 2

print(f"daily rows: {len(daily)}")
print(f"date range: {daily.index.min()} to {daily.index.max()}")

# ================================================================
# Train / Test split — same cutoff as other models
# ================================================================
cutoff = pd.Timestamp('2025-01-01')
train = daily[daily.index < cutoff]
test  = daily[daily.index >= cutoff]
print(f"train: {len(train)} days, test: {len(test)} days")

# ================================================================
# Exogenous variables
# ================================================================
exog_cols = ['temp', 'temp_sq', 'is_weekend']
exog_train = train[exog_cols]
exog_test  = test[exog_cols]

# ================================================================
# SARIMAX order selection via AIC
# ================================================================
print("\n" + "=" * 60)
print("SARIMAX Order Selection (AIC-based)")
print("=" * 60)

# candidate orders — keep tractable
orders = [
    (1, 1, 1),
    (2, 1, 1),
    (1, 1, 2),
    (2, 1, 2),
    (3, 1, 1),
]
seasonal_orders = [
    (1, 0, 1, 7),
    (1, 1, 1, 7),
    (0, 1, 1, 7),
]

best_aic = np.inf
best_order = None
best_sorder = None
aic_results = []

for order in orders:
    for sorder in seasonal_orders:
        try:
            mdl = SARIMAX(
                train['mw'],
                exog=exog_train,
                order=order,
                seasonal_order=sorder,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            res = mdl.fit(disp=False, maxiter=500)
            aic = res.aic
            aic_results.append({
                'order': order, 'seasonal': sorder, 'aic': aic
            })
            tag = ""
            if aic < best_aic:
                best_aic = aic
                best_order = order
                best_sorder = sorder
                tag = " << best"
            print(f"  SARIMAX{order}x{sorder}  AIC={aic:.1f}{tag}")
        except Exception as e:
            print(f"  SARIMAX{order}x{sorder}  FAILED: {e}")

print(f"\n>> Best: SARIMAX{best_order}x{best_sorder}  AIC={best_aic:.1f}")

# ================================================================
# Fit final model
# ================================================================
print("\n" + "=" * 60)
print("Fitting Final SARIMAX Model")
print("=" * 60)

final_model = SARIMAX(
    train['mw'],
    exog=exog_train,
    order=best_order,
    seasonal_order=best_sorder,
    enforce_stationarity=False,
    enforce_invertibility=False,
)
result = final_model.fit(disp=True, maxiter=1000)
print(result.summary())

# ================================================================
# In-sample (train) predictions
# ================================================================
train_pred = result.fittedvalues
train_pred = train_pred.reindex(train.index)

# ================================================================
# Out-of-sample (test) forecast
# ================================================================
forecast = result.get_forecast(steps=len(test), exog=exog_test)
test_pred = forecast.predicted_mean
test_ci   = forecast.conf_int(alpha=0.05)

# align indices
test_pred.index = test.index
test_ci.index   = test.index

# ================================================================
# Metrics
# ================================================================
def calc_metrics(actual, pred):
    mask = ~(np.isnan(actual) | np.isnan(pred))
    a, p = actual[mask], pred[mask]
    rmse = np.sqrt(mean_squared_error(a, p))
    mae  = mean_absolute_error(a, p)
    r2   = r2_score(a, p)
    mape = np.mean(np.abs((a - p) / a)) * 100
    return rmse, mae, r2, mape

rmse, mae, r2, mape = calc_metrics(test['mw'].values, test_pred.values)
print(f"\nSARIMAX TEST (daily):  RMSE={rmse:.2f}  MAE={mae:.2f}  R2={r2:.4f}  MAPE={mape:.2f}%")

tr_rmse, tr_mae, tr_r2, tr_mape = calc_metrics(train['mw'].values, train_pred.values)
print(f"SARIMAX TRAIN (daily): RMSE={tr_rmse:.2f}  MAE={tr_mae:.2f}  R2={tr_r2:.4f}  MAPE={tr_mape:.2f}%")

# ================================================================
# Residual trend — same analysis as ridge weather-only
# ================================================================
resid_test = test['mw'].values - test_pred.values
resid_train = train['mw'].values - train_pred.values

from scipy import stats as sp_stats

x_days_test = np.arange(len(resid_test))
slope, intercept, r_val, p_val, std_err = sp_stats.linregress(x_days_test, resid_test)
growth_per_month = slope * 30
growth_per_year  = slope * 365

print(f"\nSARIMAX residual trend (test period):")
print(f"  slope = {slope:.4f} MW/day = {growth_per_month:.1f} MW/month = {growth_per_year:.0f} MW/year")
print(f"  R² of trend = {r_val**2:.4f},  p-value = {p_val:.2e}")

if p_val < 0.05 and slope > 0:
    print(f"  >>> SIGNIFICANT POSITIVE DRIFT — corroborates Ridge findings <<<")

# ================================================================
# PLOTS
# ================================================================

# 1. Forecast vs actual with confidence interval
plt.figure(figsize=(16, 6))
plt.plot(test.index, test['mw'], label='Actual (daily avg)', linewidth=1, alpha=0.8)
plt.plot(test.index, test_pred, label='SARIMAX forecast', linewidth=1, alpha=0.8, color='orange')
plt.fill_between(test.index,
                 test_ci.iloc[:, 0], test_ci.iloc[:, 1],
                 alpha=0.15, color='orange', label='95% CI')
plt.title('SARIMAX: Daily Average Load Forecast (Test Set)', fontsize=14)
plt.xlabel('Date')
plt.ylabel('MW (daily avg)')
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '35_sarimax_forecast.png'), dpi=200)
plt.close()

# 2. Residual scatter
plt.figure(figsize=(16, 6))
plt.scatter(test.index, resid_test, s=8, alpha=0.5, color='darkorange')
trend_line = slope * x_days_test + intercept
plt.plot(test.index, trend_line, 'r-', linewidth=2.5,
         label=f'Trend: {growth_per_month:+.1f} MW/month (p={p_val:.2e})')
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.title('SARIMAX: Test Residuals — Data Center Effect', fontsize=14)
plt.xlabel('Date')
plt.ylabel('Residual (MW)')
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '36_sarimax_residuals_trend.png'), dpi=200)
plt.close()

# 3. Monthly residual bars
test_monthly = pd.DataFrame({
    'datetime': test.index,
    'residual': resid_test
})
test_monthly['month_str'] = test_monthly['datetime'].dt.to_period('M').astype(str)
monthly_stats = test_monthly.groupby('month_str')['residual'].agg(['mean', 'std', 'count']).reset_index()

plt.figure(figsize=(14, 6))
x_pos = range(len(monthly_stats))
plt.bar(x_pos, monthly_stats['mean'].values,
        yerr=monthly_stats['std'].values,
        capsize=3, alpha=0.7, color='darkorange', edgecolor='black', linewidth=0.5)
plt.xticks(list(x_pos), monthly_stats['month_str'].values, rotation=45, ha='right')
plt.axhline(y=0, color='r', linestyle='--', linewidth=1.5)
plt.title('SARIMAX: Monthly Avg Residuals (Test)', fontsize=14)
plt.ylabel('Avg Residual (MW)')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '37_sarimax_monthly_residual_bars.png'), dpi=200)
plt.close()

print("\nSARIMAX plots saved")

# ================================================================
# Save predictions (daily level)
# ================================================================
# Test predictions
test_out = pd.DataFrame({
    'datetime': test.index,
    'actual':   test['mw'].values,
    'sarimax_pred': test_pred.values,
})
test_out.to_csv(os.path.join(PRED_DIR, 'sarimax_test_preds.csv'), index=False)

# Train predictions
train_out = pd.DataFrame({
    'datetime': train.index,
    'actual':   train['mw'].values,
    'sarimax_pred': train_pred.values,
})
train_out.to_csv(os.path.join(PRED_DIR, 'sarimax_train_preds.csv'), index=False)

print("saved SARIMAX predictions (daily)")
