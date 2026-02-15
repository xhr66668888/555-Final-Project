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

# load preds
ridge_df = pd.read_csv(os.path.join(PRED_DIR, 'ridge_test_preds.csv'))
ridge_df['datetime'] = pd.to_datetime(ridge_df['datetime'])

xgb_df = pd.read_csv(os.path.join(PRED_DIR, 'xgb_test_preds.csv'))
xgb_df['datetime'] = pd.to_datetime(xgb_df['datetime'])

tf_df = pd.read_csv(os.path.join(PRED_DIR, 'transformer_test_preds.csv'))
tf_df['datetime'] = pd.to_datetime(tf_df['datetime'])

print(f"Ridge: {len(ridge_df)}, XGBoost: {len(xgb_df)}, PatchTST: {len(tf_df)}")

# residuals
ridge_df['resid_wx'] = ridge_df['actual'] - ridge_df['ridge_wx_pred']
ridge_df['resid_full'] = ridge_df['actual'] - ridge_df['ridge_full_pred']
ridge_df['resid_lasso'] = ridge_df['actual'] - ridge_df['lasso_pred']
ridge_df['resid_en'] = ridge_df['actual'] - ridge_df['elasticnet_pred']
xgb_df['residual'] = xgb_df['actual'] - xgb_df['xgb_pred']
tf_df['residual'] = tf_df['actual'] - tf_df['patchtst_pred']

def get_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    mape = np.mean(np.abs((actual-pred)/actual))*100
    return rmse, mae, r2, mape

# =========================================================
# MODEL COMPARISON TABLE
# =========================================================
print("\n" + "="*70)
print("MODEL COMPARISON ON TEST SET (Jan 2025 - Jan 2026)")
print("="*70)
print(f"{'Model':<25} {'RMSE':>8} {'MAE':>8} {'R2':>8} {'MAPE':>8}")
print("-"*61)

models_info = [
    ('Ridge (weather-only)', ridge_df['actual'].values, ridge_df['ridge_wx_pred'].values),
    ('Ridge (full)', ridge_df['actual'].values, ridge_df['ridge_full_pred'].values),
    ('Lasso (full)', ridge_df['actual'].values, ridge_df['lasso_pred'].values),
    ('ElasticNet (full)', ridge_df['actual'].values, ridge_df['elasticnet_pred'].values),
    ('XGBoost', xgb_df['actual'].values, xgb_df['xgb_pred'].values),
    ('PatchTST', tf_df['actual'].values, tf_df['patchtst_pred'].values),
]

results_table=[]
i=0
while i<len(models_info):
    name, act, pred = models_info[i]
    rmse, mae, r2, mape = get_metrics(act, pred)
    print(f"{name:<25} {rmse:>8.2f} {mae:>8.2f} {r2:>8.4f} {mape:>7.2f}%")
    results_table.append({'model':name, 'rmse':rmse, 'mae':mae, 'r2':r2, 'mape':mape})
    i+=1

res_df = pd.DataFrame(results_table)
res_df.to_csv(os.path.join(PRED_DIR, 'model_comparison.csv'), index=False)

# =========================================================
# DATA CENTER EFFECT: residual trend analysis
# =========================================================
print("\n" + "="*70)
print("RESIDUAL TREND ANALYSIS — Data Center Effect")
print("="*70)

ridge_df['month_str'] = ridge_df['datetime'].dt.to_period('M').astype(str)
monthly = ridge_df.groupby('month_str')['resid_wx'].agg(['mean','std','count']).reset_index()

print("\nWeather-Only Ridge — Monthly Avg Residuals (Test Period):")
print("(positive = model under-predicts = structural load growth)")
j=0
while j<len(monthly):
    row = monthly.iloc[j]
    cnt = int(row['count'])
    std_val = row['std'] if not np.isnan(row['std']) else 0
    print(f"  {row['month_str']}: mean={row['mean']:+.1f} MW, std={std_val:.1f}, n={cnt}")
    j+=1

# linear trend in weather-only residual
x_hrs = (ridge_df['datetime'] - ridge_df['datetime'].min()).dt.total_seconds().values / 3600
y_resid = ridge_df['resid_wx'].values

slope, intercept, r_val, p_val, std_err = stats.linregress(x_hrs, y_resid)

growth_per_month = slope * 24 * 30
growth_per_year = slope * 24 * 365

print(f"\nLinear trend in Weather-Only Ridge residuals:")
print(f"  slope = {slope:.6f} MW/hour")
print(f"  = {growth_per_month:.1f} MW/month")
print(f"  = {growth_per_year:.0f} MW/year")
print(f"  R^2 of trend = {r_val**2:.4f}")
print(f"  p-value = {p_val:.2e}")

if p_val < 0.05 and slope > 0:
    print(f"\n  >>> STATISTICALLY SIGNIFICANT POSITIVE TREND (p < 0.05) <<<")
    print(f"  Weather-only model increasingly under-predicts actual load.")
    avg_resid = np.mean(y_resid)
    print(f"  Average under-prediction: {avg_resid:.0f} MW")
    print(f"  Estimated structural (non-weather) growth: ~{growth_per_year:.0f} MW/year")
    print(f"  This signal is consistent with data center load additions in AEP Ohio.")

# =========================================================
# PLOTS
# =========================================================

# 1 monthly bar chart
plt.figure(figsize=(14,6))
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

# 2 trend line
plt.figure(figsize=(16,6))
plt.scatter(ridge_df['datetime'], y_resid, s=0.3, alpha=0.12, color='steelblue', label='Residuals')
trend_y = slope * x_hrs + intercept
plt.plot(ridge_df['datetime'], trend_y, 'r-', linewidth=3,
         label=f'Trend: {growth_per_month:+.1f} MW/month (R²={r_val**2:.3f})')
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.title('Weather-Only Ridge: Residual Trend — Data Center Effect', fontsize=14)
plt.xlabel('Date')
plt.ylabel('Residual (MW)')
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '29_residual_trend_line.png'), dpi=200)
plt.close()

# 3 smoothed residual comparison
plt.figure(figsize=(16,6))
ridge_df['smooth_wx'] = ridge_df['resid_wx'].rolling(168, min_periods=24).mean()
ridge_df['smooth_full'] = ridge_df['resid_full'].rolling(168, min_periods=24).mean()
xgb_df['smooth'] = xgb_df['residual'].rolling(168, min_periods=24).mean()
tf_df['smooth'] = tf_df['residual'].rolling(168, min_periods=24).mean()

plt.plot(ridge_df['datetime'], ridge_df['smooth_wx'],
         label='Ridge (weather-only)', linewidth=1.5, alpha=0.8)
plt.plot(ridge_df['datetime'], ridge_df['smooth_full'],
         label='Ridge (full)', linewidth=1.5, alpha=0.8)
plt.plot(xgb_df['datetime'], xgb_df['smooth'],
         label='XGBoost', linewidth=1.5, alpha=0.8)
plt.plot(tf_df['datetime'], tf_df['smooth'],
         label='PatchTST', linewidth=1.5, alpha=0.8)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.4)
plt.title('Smoothed Residuals Comparison (168h rolling mean)', fontsize=14)
plt.xlabel('Date')
plt.ylabel('Residual (MW)')
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '30_residual_comparison_smooth.png'), dpi=200)
plt.close()

# 4 full timeline analysis
print("\n" + "="*70)
print("FULL TIMELINE ANALYSIS (Train + Test)")
print("="*70)

try:
    ridge_train = pd.read_csv(os.path.join(PRED_DIR, 'ridge_train_preds.csv'))
    ridge_train['datetime'] = pd.to_datetime(ridge_train['datetime'])
    ridge_train['resid_wx'] = ridge_train['actual'] - ridge_train['ridge_wx_pred']

    xgb_train = pd.read_csv(os.path.join(PRED_DIR, 'xgb_train_preds.csv'))
    xgb_train['datetime'] = pd.to_datetime(xgb_train['datetime'])
    xgb_train['residual'] = xgb_train['actual'] - xgb_train['xgb_pred']

    full_ridge = pd.concat([
        ridge_train[['datetime','actual','resid_wx']],
        ridge_df[['datetime','actual','resid_wx']]
    ]).sort_values('datetime')

    full_xgb = pd.concat([
        xgb_train[['datetime','actual','residual']],
        xgb_df[['datetime','actual','residual']]
    ]).sort_values('datetime')

    # yearly stats
    full_ridge['year'] = full_ridge['datetime'].dt.year
    yearly = full_ridge.groupby('year')['resid_wx'].agg(['mean','std','count'])
    print("\nWeather-Only Ridge: Yearly Avg Residuals (full period):")
    print(yearly.to_string())

    full_xgb['year'] = full_xgb['datetime'].dt.year
    yearly_xgb = full_xgb.groupby('year')['residual'].agg(['mean','std','count'])
    print("\nXGBoost: Yearly Avg Residuals (full period):")
    print(yearly_xgb.to_string())

    # full residual timeline
    plt.figure(figsize=(18,6))
    full_ridge['smooth'] = full_ridge['resid_wx'].rolling(720, min_periods=168).mean()
    full_xgb['smooth'] = full_xgb['residual'].rolling(720, min_periods=168).mean()

    plt.plot(full_ridge['datetime'], full_ridge['smooth'],
             label='Ridge (weather-only)', linewidth=1.8)
    plt.plot(full_xgb['datetime'], full_xgb['smooth'],
             label='XGBoost', linewidth=1.8)
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

    fig, ax1 = plt.subplots(figsize=(12,6))
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
    ax1.legend(lines1+lines2, labels1+labels2, loc='upper left', fontsize=12)
    plt.title('Load Growth vs Model Residual by Year', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '32_yearly_load_vs_residual.png'), dpi=200)
    plt.close()

    print("\nresidual growth in later years => structural load growth")
    print("beyond weather, consistent with data center expansion")

except FileNotFoundError as e:
    print(f"train prediction files not found: {e}")

# 5 residual distribution comparison
fig, axes = plt.subplots(2, 3, figsize=(18,10))
plot_data = [
    (ridge_df['resid_wx'].values, 'Ridge (wx-only)', 'blue'),
    (ridge_df['resid_full'].values, 'Ridge (full)', 'green'),
    (ridge_df['resid_lasso'].values, 'Lasso', 'purple'),
    (ridge_df['resid_en'].values, 'ElasticNet', 'cyan'),
    (xgb_df['residual'].values, 'XGBoost', 'orange'),
    (tf_df['residual'].values, 'PatchTST', 'green'),
]

q=0
while q<len(plot_data):
    row = q//3
    col = q%3
    data, name, clr = plot_data[q]
    axes[row][col].hist(data, bins=80, alpha=0.7, edgecolor='black', linewidth=0.3, color=clr)
    axes[row][col].set_title(f'{name} Residuals')
    axes[row][col].set_xlabel('MW')
    axes[row][col].axvline(x=0, color='r', linestyle='--', alpha=0.5)
    mu = np.mean(data)
    sig = np.std(data)
    axes[row][col].set_title(f'{name}\nμ={mu:.0f}, σ={sig:.0f}')
    q+=1

plt.suptitle('Residual Distributions (Test Set)', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '33_residual_distributions.png'), dpi=200)
plt.close()

# 6 model comparison bar chart
fig, axes = plt.subplots(1, 4, figsize=(20,5))
metric_names = ['rmse','mae','r2','mape']
titles = ['RMSE (lower better)','MAE (lower better)','R² (higher better)','MAPE % (lower better)']
m_names = [r['model'] for r in results_table]
m=0
while m<4:
    vals = [r[metric_names[m]] for r in results_table]
    clrs = ['steelblue' if metric_names[m]!='r2' else 'steelblue'] * len(vals)
    # highlight best
    if metric_names[m]=='r2':
        best_idx = np.argmax(vals)
    else:
        best_idx = np.argmin(vals)
    clrs[best_idx] = 'orangered'

    axes[m].barh(range(len(m_names)), vals, color=clrs, edgecolor='black', linewidth=0.3)
    axes[m].set_yticks(range(len(m_names)))
    axes[m].set_yticklabels(m_names)
    axes[m].set_title(titles[m])
    axes[m].invert_yaxis()
    m+=1

plt.suptitle('Model Comparison', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '34_model_comparison_bars.png'), dpi=200)
plt.close()

n_plots = len([f for f in os.listdir(PLOT_DIR) if f.endswith('.png')])
print(f"\n=== analysis complete, {n_plots} total plots ===")
