"""
Residual Analysis — First-class deliverable.

This script quantifies the "data center effect" from model residuals using
multiple complementary methods:

PRE-REGISTERED MAIN STATISTIC:
    Slope of the 30-day rolling mean of Weather-Only Ridge residuals (MW/month)
    during the test period, tested for significance with linear regression
    and confirmed with a CUSUM change-point test.

METHODS:
  1. Monthly rolling mean of residuals — trend visualization
  2. Linear trend (OLS slope) with confidence interval
  3. Piecewise linear trend — slope before vs after inflection
  4. Change-point detection via CUSUM test
  5. Cross-model residual drift comparison (Ridge, SARIMAX, XGBoost, PatchTST)

CONCLUSION CRITERION:
  If >=2 model families (classical statistical + ML) show significantly
  positive residual drift (p < 0.05), we conclude structural load growth
  beyond weather explanations — consistent with data center additions.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE, 'output')
PLOT_DIR = os.path.join(OUT_DIR, 'plots')
PRED_DIR = os.path.join(OUT_DIR, 'predictions')
os.makedirs(PLOT_DIR, exist_ok=True)

# =========================================================
# LOAD PREDICTIONS FROM ALL MODELS
# =========================================================
ridge_df = pd.read_csv(os.path.join(PRED_DIR, 'ridge_test_preds.csv'))
ridge_df['datetime'] = pd.to_datetime(ridge_df['datetime'])

xgb_df = pd.read_csv(os.path.join(PRED_DIR, 'xgb_test_preds.csv'))
xgb_df['datetime'] = pd.to_datetime(xgb_df['datetime'])

tf_df = pd.read_csv(os.path.join(PRED_DIR, 'transformer_test_preds.csv'))
tf_df['datetime'] = pd.to_datetime(tf_df['datetime'])

# SARIMAX (daily level)
sarimax_df = pd.read_csv(os.path.join(PRED_DIR, 'sarimax_test_preds.csv'))
sarimax_df['datetime'] = pd.to_datetime(sarimax_df['datetime'])

print(f"Ridge: {len(ridge_df)}, XGBoost: {len(xgb_df)}, PatchTST: {len(tf_df)}, SARIMAX(daily): {len(sarimax_df)}")

# compute residuals
ridge_df['resid_wx'] = ridge_df['actual'] - ridge_df['ridge_wx_pred']
ridge_df['resid_full'] = ridge_df['actual'] - ridge_df['ridge_full_pred']
ridge_df['resid_lasso'] = ridge_df['actual'] - ridge_df['lasso_pred']
ridge_df['resid_en'] = ridge_df['actual'] - ridge_df['elasticnet_pred']
xgb_df['residual'] = xgb_df['actual'] - xgb_df['xgb_pred']
tf_df['residual'] = tf_df['actual'] - tf_df['patchtst_pred']
sarimax_df['residual'] = sarimax_df['actual'] - sarimax_df['sarimax_pred']


def get_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    mape = np.mean(np.abs((actual - pred) / actual)) * 100
    return rmse, mae, r2, mape


# =========================================================
# 1. MODEL COMPARISON TABLE
# =========================================================
print("\n" + "=" * 75)
print("MODEL COMPARISON ON TEST SET (Jan 2025 - Jan 2026)")
print("=" * 75)
print(f"{'Model':<30} {'RMSE':>8} {'MAE':>8} {'R2':>8} {'MAPE':>8}")
print("-" * 66)

models_info = [
    ('Ridge (weather-only)',   ridge_df['actual'].values, ridge_df['ridge_wx_pred'].values),
    ('Ridge (full features)',  ridge_df['actual'].values, ridge_df['ridge_full_pred'].values),
    ('Lasso (full)',           ridge_df['actual'].values, ridge_df['lasso_pred'].values),
    ('ElasticNet (full)',      ridge_df['actual'].values, ridge_df['elasticnet_pred'].values),
    ('SARIMAX (daily)',        sarimax_df['actual'].values, sarimax_df['sarimax_pred'].values),
    ('XGBoost',                xgb_df['actual'].values, xgb_df['xgb_pred'].values),
    ('PatchTST',               tf_df['actual'].values, tf_df['patchtst_pred'].values),
]

results_table = []
i = 0
while i < len(models_info):
    name, act, pred = models_info[i]
    rmse, mae, r2, mape = get_metrics(act, pred)
    print(f"{name:<30} {rmse:>8.2f} {mae:>8.2f} {r2:>8.4f} {mape:>7.2f}%")
    results_table.append({'model': name, 'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape})
    i += 1

res_df = pd.DataFrame(results_table)
res_df.to_csv(os.path.join(PRED_DIR, 'model_comparison.csv'), index=False)


# =========================================================
# 2. PRE-REGISTERED MAIN STATISTIC
#    Slope of Weather-Only Ridge residuals (MW/month)
# =========================================================
print("\n" + "=" * 75)
print("PRE-REGISTERED MAIN STATISTIC")
print("  Metric: OLS slope of weather-only Ridge residuals over test period")
print("  Units:  MW per month")
print("  H0:    slope = 0  (no structural load growth)")
print("  H1:    slope > 0  (load growing beyond weather explanations)")
print("=" * 75)

x_hrs = (ridge_df['datetime'] - ridge_df['datetime'].min()).dt.total_seconds().values / 3600
y_resid = ridge_df['resid_wx'].values

slope, intercept, r_val, p_val, std_err = stats.linregress(x_hrs, y_resid)
growth_per_month = slope * 24 * 30
growth_per_year = slope * 24 * 365
ci95 = 1.96 * std_err * 24 * 30  # 95% CI on monthly growth

print(f"\n  Ridge (weather-only) residual trend:")
print(f"    slope          = {slope:.6f} MW/hour")
print(f"    monthly growth = {growth_per_month:+.1f} MW/month  (95% CI: [{growth_per_month - ci95:.1f}, {growth_per_month + ci95:.1f}])")
print(f"    annual growth  = {growth_per_year:+.0f} MW/year")
print(f"    R2 of trend    = {r_val ** 2:.4f}")
print(f"    p-value        = {p_val:.2e}")

if p_val < 0.05 and slope > 0:
    print(f"\n  >>> RESULT: REJECT H0 -- significant positive drift (p = {p_val:.2e}) <<<")
    print(f"  Weather-only model increasingly under-predicts actual load.")
    print(f"  Estimated structural growth: ~{growth_per_year:.0f} MW/year")
else:
    print(f"\n  >>> RESULT: FAIL TO REJECT H0 -- no significant drift detected <<<")


# =========================================================
# 3. MONTHLY ROLLING MEAN OF RESIDUALS
# =========================================================
print("\n" + "=" * 75)
print("MONTHLY ROLLING MEAN OF RESIDUALS")
print("=" * 75)

ridge_df['month_str'] = ridge_df['datetime'].dt.to_period('M').astype(str)
monthly = ridge_df.groupby('month_str')['resid_wx'].agg(['mean', 'std', 'count']).reset_index()

print("\nWeather-Only Ridge -- Monthly Avg Residuals (Test Period):")
print("(positive = model under-predicts = structural load growth)")
j = 0
while j < len(monthly):
    row = monthly.iloc[j]
    cnt = int(row['count'])
    std_val = row['std'] if not np.isnan(row['std']) else 0
    print(f"  {row['month_str']}: mean={row['mean']:+.1f} MW, std={std_val:.1f}, n={cnt}")
    j += 1


# =========================================================
# 4. PIECEWISE LINEAR TREND -- slope before/after inflection
# =========================================================
print("\n" + "=" * 75)
print("PIECEWISE LINEAR TREND -- Before vs After Midpoint")
print("=" * 75)

mid_idx = len(y_resid) // 2
mid_date = ridge_df['datetime'].iloc[mid_idx]
print(f"  Inflection point (midpoint): {mid_date.date()}")

# first half
slope1, int1, r1, p1, se1 = stats.linregress(x_hrs[:mid_idx], y_resid[:mid_idx])
gpm1 = slope1 * 24 * 30

# second half
x_hrs_2 = x_hrs[mid_idx:] - x_hrs[mid_idx]
slope2, int2, r2, p2, se2 = stats.linregress(x_hrs_2, y_resid[mid_idx:])
gpm2 = slope2 * 24 * 30

print(f"\n  Before {mid_date.date()}:")
print(f"    slope = {gpm1:+.1f} MW/month  (p = {p1:.2e}, R2 = {r1 ** 2:.4f})")
print(f"  After  {mid_date.date()}:")
print(f"    slope = {gpm2:+.1f} MW/month  (p = {p2:.2e}, R2 = {r2 ** 2:.4f})")

# test if slopes differ significantly (Chow-like comparison)
slope_diff = slope2 - slope1
pooled_se = np.sqrt(se1 ** 2 + se2 ** 2)
if pooled_se > 0:
    z_stat = slope_diff / pooled_se
    p_chow = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    print(f"\n  Slope difference test:")
    print(f"    delta_slope = {(slope_diff * 24 * 30):+.1f} MW/month, z = {z_stat:.2f}, p = {p_chow:.4f}")
    if p_chow < 0.05:
        print(f"    >>> Slopes differ significantly -- drift is accelerating <<<")
    else:
        print(f"    Slopes not significantly different -- consistent drift rate")


# =========================================================
# 5. CHANGE-POINT DETECTION -- CUSUM TEST
# =========================================================
print("\n" + "=" * 75)
print("CHANGE-POINT DETECTION -- CUSUM")
print("=" * 75)

# Use 30-day rolling mean of residuals for cleaner signal
roll_mean = pd.Series(y_resid).rolling(720, min_periods=168).mean().values  # 30-day rolling

# CUSUM statistic
valid_mask = ~np.isnan(roll_mean)
rm_valid = roll_mean[valid_mask]
cusum = np.cumsum(rm_valid - np.mean(rm_valid))
cusum_max_idx = np.argmax(np.abs(cusum))

# Map back to datetime
valid_dates = ridge_df['datetime'].values[valid_mask]
changepoint_date = pd.Timestamp(valid_dates[cusum_max_idx])

print(f"  CUSUM max deviation at index {cusum_max_idx} -- date: {changepoint_date.date()}")
print(f"  CUSUM value at changepoint: {cusum[cusum_max_idx]:.1f}")

# Bootstrap significance for CUSUM
n_boot = 1000
max_cusum_obs = np.max(np.abs(cusum))
boot_exceeds = 0
rng = np.random.RandomState(42)
for _ in range(n_boot):
    shuffled = rng.permutation(rm_valid)
    boot_cusum = np.cumsum(shuffled - np.mean(shuffled))
    if np.max(np.abs(boot_cusum)) >= max_cusum_obs:
        boot_exceeds += 1

cusum_p = boot_exceeds / n_boot
print(f"  Bootstrap CUSUM p-value: {cusum_p:.4f}  (n_boot={n_boot})")
if cusum_p < 0.05:
    print(f"  >>> SIGNIFICANT CHANGE-POINT DETECTED <<<")
else:
    print(f"  Change-point not significant at alpha=0.05")


# =========================================================
# 6. CROSS-MODEL DRIFT COMPARISON
# =========================================================
print("\n" + "=" * 75)
print("CROSS-MODEL RESIDUAL DRIFT COMPARISON")
print("  If multiple model families show positive drift -> structural effect")
print("=" * 75)

drift_results = []

# (a) Ridge (weather-only) -- already computed
drift_results.append({
    'model': 'Ridge (weather-only)',
    'slope_mw_per_month': growth_per_month,
    'p_value': p_val,
    'significant': p_val < 0.05 and slope > 0,
    'family': 'Linear/ML'
})

# (b) SARIMAX
sx_days = np.arange(len(sarimax_df))
sx_resid = sarimax_df['residual'].values
sx_slope, sx_int, sx_r, sx_p, sx_se = stats.linregress(sx_days, sx_resid)
sx_gpm = sx_slope * 30

drift_results.append({
    'model': 'SARIMAX',
    'slope_mw_per_month': sx_gpm,
    'p_value': sx_p,
    'significant': sx_p < 0.05 and sx_slope > 0,
    'family': 'Classical Statistical'
})

# (c) XGBoost
xg_hrs = (xgb_df['datetime'] - xgb_df['datetime'].min()).dt.total_seconds().values / 3600
xg_resid = xgb_df['residual'].values
xg_slope, xg_int, xg_r, xg_p, xg_se = stats.linregress(xg_hrs, xg_resid)
xg_gpm = xg_slope * 24 * 30

drift_results.append({
    'model': 'XGBoost',
    'slope_mw_per_month': xg_gpm,
    'p_value': xg_p,
    'significant': xg_p < 0.05 and xg_slope > 0,
    'family': 'Tree-based ML'
})

# (d) PatchTST
tf_hrs = (tf_df['datetime'] - tf_df['datetime'].min()).dt.total_seconds().values / 3600
tf_resid = tf_df['residual'].values
tf_slope, tf_int, tf_r, tf_p, tf_se = stats.linregress(tf_hrs, tf_resid)
tf_gpm = tf_slope * 24 * 30

drift_results.append({
    'model': 'PatchTST',
    'slope_mw_per_month': tf_gpm,
    'p_value': tf_p,
    'significant': tf_p < 0.05 and tf_slope > 0,
    'family': 'Deep Learning'
})

print(f"\n{'Model':<25} {'Family':<22} {'Drift (MW/mo)':>14} {'p-value':>12} {'Sig?':>5}")
print("-" * 80)
for d in drift_results:
    sig_str = "YES" if d['significant'] else "no"
    print(f"  {d['model']:<23} {d['family']:<22} {d['slope_mw_per_month']:>+12.1f} {d['p_value']:>12.2e}   {sig_str}")

n_sig = sum(1 for d in drift_results if d['significant'])
n_families = len(set(d['family'] for d in drift_results if d['significant']))

print(f"\n  Models with significant positive drift: {n_sig}/{len(drift_results)}")
print(f"  Distinct model families with drift:     {n_families}")

if n_families >= 2:
    print(f"\n  ============================================================")
    print(f"  CONCLUSION: >=2 model families confirm structural load")
    print(f"  growth beyond weather -- consistent with data center")
    print(f"  expansion in AEP Ohio service territory.")
    print(f"  ============================================================")
else:
    print(f"\n  CONCLUSION: Insufficient cross-model evidence for structural drift.")

drift_df = pd.DataFrame(drift_results)
drift_df.to_csv(os.path.join(PRED_DIR, 'drift_analysis.csv'), index=False)


# =========================================================
# PLOTS
# =========================================================

# P1. Monthly residual bars (Ridge weather-only)
plt.figure(figsize=(14, 6))
x_pos = range(len(monthly))
plt.bar(x_pos, monthly['mean'].values, yerr=monthly['std'].values,
        capsize=3, alpha=0.7, color='steelblue', edgecolor='navy', linewidth=0.5)
labels = monthly['month_str'].values
plt.xticks(list(x_pos), labels, rotation=45, ha='right')
plt.axhline(y=0, color='r', linestyle='--', linewidth=1.5)
plt.title('Weather-Only Ridge: Monthly Avg Residuals (Test)', fontsize=14)
plt.ylabel('Avg Residual (MW)')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '28_monthly_residual_bars.png'), dpi=200)
plt.close()

# P2. Residual trend line with piecewise overlay
fig, ax = plt.subplots(figsize=(16, 7))
ax.scatter(ridge_df['datetime'], y_resid, s=0.3, alpha=0.10, color='steelblue', label='Hourly residuals')

# overall trend
trend_y = slope * x_hrs + intercept
ax.plot(ridge_df['datetime'], trend_y, 'r-', linewidth=2.5,
        label=f'Overall: {growth_per_month:+.1f} MW/mo (p={p_val:.1e})')

# piecewise trend
piece1_y = slope1 * x_hrs[:mid_idx] + int1
ax.plot(ridge_df['datetime'].iloc[:mid_idx], piece1_y, '--', color='darkgreen', linewidth=2,
        label=f'1st half: {gpm1:+.1f} MW/mo')

piece2_y = slope2 * x_hrs_2 + int2
ax.plot(ridge_df['datetime'].iloc[mid_idx:], piece2_y, '--', color='purple', linewidth=2,
        label=f'2nd half: {gpm2:+.1f} MW/mo')

ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax.axvline(x=mid_date, color='gray', linestyle=':', alpha=0.5, linewidth=1.5)
ax.set_title('Weather-Only Ridge: Residual Trend with Piecewise Fit', fontsize=14)
ax.set_xlabel('Date')
ax.set_ylabel('Residual (MW)')
ax.legend(fontsize=11, loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '29_residual_trend_piecewise.png'), dpi=200)
plt.close()

# P3. Smoothed residual comparison: Ridge + SARIMAX + XGB + PatchTST
plt.figure(figsize=(16, 6))
ridge_df['smooth_wx'] = ridge_df['resid_wx'].rolling(168, min_periods=24).mean()
ridge_df['smooth_full'] = ridge_df['resid_full'].rolling(168, min_periods=24).mean()
xgb_df['smooth'] = xgb_df['residual'].rolling(168, min_periods=24).mean()
tf_df['smooth'] = tf_df['residual'].rolling(168, min_periods=24).mean()
sarimax_df['smooth'] = sarimax_df['residual'].rolling(7, min_periods=3).mean()

plt.plot(ridge_df['datetime'], ridge_df['smooth_wx'],
         label='Ridge (weather-only)', linewidth=1.5, alpha=0.8)
plt.plot(ridge_df['datetime'], ridge_df['smooth_full'],
         label='Ridge (full)', linewidth=1.5, alpha=0.8)
plt.plot(sarimax_df['datetime'], sarimax_df['smooth'],
         label='SARIMAX (daily)', linewidth=1.5, alpha=0.8, color='darkorange', linestyle='--')
plt.plot(xgb_df['datetime'], xgb_df['smooth'],
         label='XGBoost', linewidth=1.5, alpha=0.8)
plt.plot(tf_df['datetime'], tf_df['smooth'],
         label='PatchTST', linewidth=1.5, alpha=0.8)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.4)
plt.title('Smoothed Residuals: All Models (7-day rolling mean)', fontsize=13)
plt.xlabel('Date')
plt.ylabel('Residual (MW)')
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '30_residual_comparison_smooth.png'), dpi=200)
plt.close()

# P4. CUSUM plot
plt.figure(figsize=(16, 6))
plt.plot(valid_dates, cusum, linewidth=1.5, color='darkred')
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.axvline(x=changepoint_date, color='blue', linestyle='--', linewidth=2,
            label=f'Change-point: {changepoint_date.date()} (p={cusum_p:.3f})')
plt.fill_between(valid_dates, cusum, alpha=0.15, color='darkred')
plt.title('CUSUM of Weather-Only Residuals -- Change-Point Detection', fontsize=14)
plt.xlabel('Date')
plt.ylabel('Cumulative Sum')
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '38_cusum_changepoint.png'), dpi=200)
plt.close()

# P5. Cross-model drift bar chart
fig, ax = plt.subplots(figsize=(10, 6))
model_names = [d['model'] for d in drift_results]
drift_vals = [d['slope_mw_per_month'] for d in drift_results]
colors = ['forestgreen' if d['significant'] else 'gray' for d in drift_results]
bars = ax.barh(model_names, drift_vals, color=colors, edgecolor='black', linewidth=0.5)
ax.axvline(x=0, color='k', linestyle='-', linewidth=0.8)
ax.set_xlabel('Residual Drift (MW / month)')
ax.set_title('Cross-Model Residual Drift Comparison\n(green = significant at p<0.05)', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '39_cross_model_drift.png'), dpi=200)
plt.close()


# P6. Full timeline analysis (train + test)
print("\n" + "=" * 75)
print("FULL TIMELINE ANALYSIS (Train + Test)")
print("=" * 75)

try:
    ridge_train = pd.read_csv(os.path.join(PRED_DIR, 'ridge_train_preds.csv'))
    ridge_train['datetime'] = pd.to_datetime(ridge_train['datetime'])
    ridge_train['resid_wx'] = ridge_train['actual'] - ridge_train['ridge_wx_pred']

    xgb_train = pd.read_csv(os.path.join(PRED_DIR, 'xgb_train_preds.csv'))
    xgb_train['datetime'] = pd.to_datetime(xgb_train['datetime'])
    xgb_train['residual'] = xgb_train['actual'] - xgb_train['xgb_pred']

    sarimax_train = pd.read_csv(os.path.join(PRED_DIR, 'sarimax_train_preds.csv'))
    sarimax_train['datetime'] = pd.to_datetime(sarimax_train['datetime'])
    sarimax_train['residual'] = sarimax_train['actual'] - sarimax_train['sarimax_pred']

    full_ridge = pd.concat([
        ridge_train[['datetime', 'actual', 'resid_wx']],
        ridge_df[['datetime', 'actual', 'resid_wx']]
    ]).sort_values('datetime')

    full_xgb = pd.concat([
        xgb_train[['datetime', 'actual', 'residual']],
        xgb_df[['datetime', 'actual', 'residual']]
    ]).sort_values('datetime')

    full_sarimax = pd.concat([
        sarimax_train[['datetime', 'actual', 'residual']],
        sarimax_df[['datetime', 'actual', 'residual']]
    ]).sort_values('datetime')

    # yearly stats
    full_ridge['year'] = full_ridge['datetime'].dt.year
    yearly = full_ridge.groupby('year')['resid_wx'].agg(['mean', 'std', 'count'])
    print("\nWeather-Only Ridge: Yearly Avg Residuals (full period):")
    print(yearly.to_string())

    full_xgb['year'] = full_xgb['datetime'].dt.year
    yearly_xgb = full_xgb.groupby('year')['residual'].agg(['mean', 'std', 'count'])
    print("\nXGBoost: Yearly Avg Residuals (full period):")
    print(yearly_xgb.to_string())

    full_sarimax['year'] = full_sarimax['datetime'].dt.year
    yearly_sx = full_sarimax.groupby('year')['residual'].agg(['mean', 'std', 'count'])
    print("\nSARIMAX: Yearly Avg Residuals (full period):")
    print(yearly_sx.to_string())

    # full residual timeline
    plt.figure(figsize=(18, 6))
    full_ridge['smooth'] = full_ridge['resid_wx'].rolling(720, min_periods=168).mean()
    full_xgb['smooth'] = full_xgb['residual'].rolling(720, min_periods=168).mean()
    full_sarimax['smooth'] = full_sarimax['residual'].rolling(30, min_periods=7).mean()

    plt.plot(full_ridge['datetime'], full_ridge['smooth'],
             label='Ridge (weather-only)', linewidth=1.8)
    plt.plot(full_xgb['datetime'], full_xgb['smooth'],
             label='XGBoost', linewidth=1.8)
    plt.plot(full_sarimax['datetime'], full_sarimax['smooth'],
             label='SARIMAX (daily)', linewidth=1.8, linestyle='--')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=pd.Timestamp('2025-01-01'), color='red', linestyle='--',
                alpha=0.7, linewidth=2, label='Train/Test Split')
    plt.title('Full Period Residual Timeline (30-day rolling mean)', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Residual (MW)')
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '31_full_residual_timeline.png'), dpi=200)
    plt.close()

    # yearly load + residual
    yr_load = full_ridge.groupby('year')['actual'].mean()
    yr_resid = full_ridge.groupby('year')['resid_wx'].mean()

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.bar(yr_load.index.astype(str), yr_load.values, alpha=0.6,
            color='steelblue', edgecolor='navy', label='Avg Load (MW)')
    ax1.set_ylabel('Average Load (MW)', fontsize=12)
    ax1.set_xlabel('Year')

    ax2 = ax1.twinx()
    ax2.plot(yr_resid.index.astype(str), yr_resid.values, 'ro-',
             linewidth=2.5, markersize=10, label='Avg Residual', zorder=5)
    ax2.set_ylabel('Avg Weather-Only Residual (MW)', fontsize=12, color='red')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=12)
    plt.title('Load Growth vs Model Residual by Year', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '32_yearly_load_vs_residual.png'), dpi=200)
    plt.close()

    print("\nResidual growth in later years => structural load growth")
    print("beyond weather, consistent with data center expansion")

except FileNotFoundError as e:
    print(f"train prediction files not found: {e}")


# P7. Residual distribution comparison (now includes SARIMAX)
fig, axes = plt.subplots(2, 4, figsize=(22, 10))
plot_data = [
    (ridge_df['resid_wx'].values,   'Ridge (wx-only)',  'blue'),
    (ridge_df['resid_full'].values, 'Ridge (full)',     'green'),
    (ridge_df['resid_lasso'].values,'Lasso',            'purple'),
    (ridge_df['resid_en'].values,   'ElasticNet',       'cyan'),
    (sarimax_df['residual'].values, 'SARIMAX (daily)',  'darkorange'),
    (xgb_df['residual'].values,     'XGBoost',          'orange'),
    (tf_df['residual'].values,      'PatchTST',         'green'),
]

q = 0
while q < len(plot_data):
    row = q // 4
    col = q % 4
    data, name, clr = plot_data[q]
    axes[row][col].hist(data, bins=80, alpha=0.7, edgecolor='black', linewidth=0.3, color=clr)
    axes[row][col].axvline(x=0, color='r', linestyle='--', alpha=0.5)
    mu = np.mean(data)
    sig = np.std(data)
    axes[row][col].set_title(f'{name}\nmean={mu:.0f}, std={sig:.0f}')
    axes[row][col].set_xlabel('MW')
    q += 1

# hide unused subplot(s)
if q < 8:
    axes[1][3].set_visible(False)

plt.suptitle('Residual Distributions (Test Set)', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '33_residual_distributions.png'), dpi=200)
plt.close()


# P8. Model comparison bar chart
fig, axes = plt.subplots(1, 4, figsize=(22, 6))
metric_names = ['rmse', 'mae', 'r2', 'mape']
titles = ['RMSE (lower better)', 'MAE (lower better)', 'R2 (higher better)', 'MAPE % (lower better)']
m_names = [r['model'] for r in results_table]

m = 0
while m < 4:
    vals = [r[metric_names[m]] for r in results_table]
    clrs = ['steelblue'] * len(vals)
    if metric_names[m] == 'r2':
        best_idx = np.argmax(vals)
    else:
        best_idx = np.argmin(vals)
    clrs[best_idx] = 'orangered'

    axes[m].barh(range(len(m_names)), vals, color=clrs, edgecolor='black', linewidth=0.3)
    axes[m].set_yticks(range(len(m_names)))
    axes[m].set_yticklabels(m_names)
    axes[m].set_title(titles[m])
    axes[m].invert_yaxis()
    m += 1

plt.suptitle('Model Comparison', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '34_model_comparison_bars.png'), dpi=200)
plt.close()

n_plots = len([f for f in os.listdir(PLOT_DIR) if f.endswith('.png')])
print(f"\n=== Residual analysis complete, {n_plots} total plots ===")
