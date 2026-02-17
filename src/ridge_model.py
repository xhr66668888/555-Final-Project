import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import pickle

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE, 'output')
PLOT_DIR = os.path.join(OUT_DIR, 'plots')
PRED_DIR = os.path.join(OUT_DIR, 'predictions')
MODEL_DIR = os.path.join(OUT_DIR, 'models')
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(OUT_DIR, 'processed_data.csv'))
df['datetime'] = pd.to_datetime(df['datetime'])
print(f"loaded {len(df)} rows")

# weather only cols (for drift analysis, no lags)
weather_cols = ['temp','dewpoint','humidity','hour','dayofweek','month','year',
                'is_weekend','is_holiday','temp_sq','temp_cb',
                'hour_sin','hour_cos','month_sin','month_cos','dow_sin','dow_cos',
                'cdd','hdd','heat_idx','temp_diff1','temp_diff24']
# add rolling if exist
roll_wx = [c for c in df.columns if 'temp_roll' in c or 'temp_std' in c]
weather_cols = weather_cols + roll_wx
weather_cols = [c for c in weather_cols if c in df.columns]

# full features
lag_cols = [c for c in df.columns if 'load_lag' in c or 'load_roll' in c or 'load_diff' in c]
full_cols = weather_cols + lag_cols
# add extra weather
extra = [c for c in ['windspeed','precip','pressure','wind_chill'] if c in df.columns]
full_cols = full_cols + extra
full_cols = list(dict.fromkeys(full_cols))  # dedup

print(f"weather features: {len(weather_cols)}")
print(f"full features: {len(full_cols)}")

y = df['mw'].values
dates = df['datetime'].values

cutoff = pd.Timestamp('2025-01-01')
train_mask = df['datetime'] < cutoff
test_mask = df['datetime'] >= cutoff

y_train = y[train_mask]
y_test = y[test_mask]
dates_train = dates[train_mask]
dates_test = dates[test_mask]
print(f"train: {len(y_train)}, test: {len(y_test)}")

def calc_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    mape = np.mean(np.abs((actual-pred)/actual))*100
    return rmse, mae, r2, mape

# ================================================================
# PART 1: weather-only Ridge with thorough alpha sweep
# ================================================================
print("\n" + "="*60)
print("Weather-Only Ridge (concept drift detector)")
print("="*60)

X_wx = df[weather_cols].values
X_wx_train = X_wx[train_mask]
X_wx_test = X_wx[test_mask]

# LEAKAGE CHECK: scaler is fit on TRAIN only, then applied to test.
# This prevents any test-set statistics from leaking into training.
scaler_wx = StandardScaler()
X_wx_train_s = scaler_wx.fit_transform(X_wx_train)   # fit on train
X_wx_test_s = scaler_wx.transform(X_wx_test)          # transform only

alphas = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
tscv = TimeSeriesSplit(n_splits=5)

best_a = None
best_rmse = 999999
cv_results_wx=[]

i=0
while i<len(alphas):
    a = alphas[i]
    rmse_vals=[]
    splits = list(tscv.split(X_wx_train_s))
    j=0
    while j<len(splits):
        tr_idx, val_idx = splits[j]
        mdl = Ridge(alpha=a)
        mdl.fit(X_wx_train_s[tr_idx], y_train[tr_idx])
        p = mdl.predict(X_wx_train_s[val_idx])
        rmse_vals.append(np.sqrt(mean_squared_error(y_train[val_idx], p)))
        j+=1
    avg = np.mean(rmse_vals)
    std = np.std(rmse_vals)
    cv_results_wx.append({'alpha':a, 'rmse':avg, 'std':std})
    print(f"  alpha={a:>8.3f}  RMSE={avg:.2f} +/- {std:.2f}")
    if avg<best_rmse:
        best_rmse=avg
        best_a=a
    i+=1

print(f"  >> best alpha: {best_a}")
ridge_wx = Ridge(alpha=best_a)
ridge_wx.fit(X_wx_train_s, y_train)

wx_train_pred = ridge_wx.predict(X_wx_train_s)
wx_test_pred = ridge_wx.predict(X_wx_test_s)

r1,r2_v,r3,r4 = calc_metrics(y_test, wx_test_pred)
print(f"\n  Weather-Only Ridge TEST: RMSE={r1:.2f}  MAE={r2_v:.2f}  R2={r3:.4f}  MAPE={r4:.2f}%")
r1t,r2t,r3t,r4t = calc_metrics(y_train, wx_train_pred)
print(f"  Weather-Only Ridge TRAIN: RMSE={r1t:.2f}  MAE={r2t:.2f}  R2={r3t:.4f}  MAPE={r4t:.2f}%")

# ================================================================
# PART 2: Full Feature Ridge
# ================================================================
print("\n" + "="*60)
print("Full Feature Ridge")
print("="*60)

X_full = df[full_cols].values
X_full_train = X_full[train_mask]
X_full_test = X_full[test_mask]

# LEAKAGE CHECK: scaler fit on TRAIN only.
scaler_full = StandardScaler()
X_full_train_s = scaler_full.fit_transform(X_full_train)   # fit on train
X_full_test_s = scaler_full.transform(X_full_test)          # transform only

best_a2=None
best_rmse2=999999

i=0
while i<len(alphas):
    a = alphas[i]
    rmse_vals=[]
    splits = list(tscv.split(X_full_train_s))
    j=0
    while j<len(splits):
        tr_idx, val_idx = splits[j]
        mdl = Ridge(alpha=a)
        mdl.fit(X_full_train_s[tr_idx], y_train[tr_idx])
        p = mdl.predict(X_full_train_s[val_idx])
        rmse_vals.append(np.sqrt(mean_squared_error(y_train[val_idx], p)))
        j+=1
    avg = np.mean(rmse_vals)
    std = np.std(rmse_vals)
    print(f"  alpha={a:>8.3f}  RMSE={avg:.2f} +/- {std:.2f}")
    if avg<best_rmse2:
        best_rmse2=avg
        best_a2=a
    i+=1

print(f"  >> best alpha: {best_a2}")
ridge_full = Ridge(alpha=best_a2)
ridge_full.fit(X_full_train_s, y_train)

full_train_pred = ridge_full.predict(X_full_train_s)
full_test_pred = ridge_full.predict(X_full_test_s)

r1,r2_v,r3,r4 = calc_metrics(y_test, full_test_pred)
print(f"\n  Full Ridge TEST: RMSE={r1:.2f}  MAE={r2_v:.2f}  R2={r3:.4f}  MAPE={r4:.2f}%")

# ================================================================
# PART 3: Lasso with full sweep
# ================================================================
print("\n" + "="*60)
print("Lasso (Full Features)")
print("="*60)

lasso_alphas = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
best_la = None
best_lrmse = 999999

i=0
while i<len(lasso_alphas):
    a = lasso_alphas[i]
    rmse_vals=[]
    splits = list(tscv.split(X_full_train_s))
    j=0
    while j<len(splits):
        tr_idx, val_idx = splits[j]
        mdl = Lasso(alpha=a, max_iter=50000)
        mdl.fit(X_full_train_s[tr_idx], y_train[tr_idx])
        p = mdl.predict(X_full_train_s[val_idx])
        rmse_vals.append(np.sqrt(mean_squared_error(y_train[val_idx], p)))
        j+=1
    avg = np.mean(rmse_vals)
    std = np.std(rmse_vals)
    print(f"  alpha={a:>8.3f}  RMSE={avg:.2f} +/- {std:.2f}")
    if avg<best_lrmse:
        best_lrmse=avg
        best_la=a
    i+=1

print(f"  >> best alpha: {best_la}")
lasso_mdl = Lasso(alpha=best_la, max_iter=50000)
lasso_mdl.fit(X_full_train_s, y_train)
lasso_test_pred = lasso_mdl.predict(X_full_test_s)
lasso_train_pred = lasso_mdl.predict(X_full_train_s)

r1,r2_v,r3,r4 = calc_metrics(y_test, lasso_test_pred)
print(f"\n  Lasso TEST: RMSE={r1:.2f}  MAE={r2_v:.2f}  R2={r3:.4f}  MAPE={r4:.2f}%")

# ================================================================
# PART 4: ElasticNet
# ================================================================
print("\n" + "="*60)
print("ElasticNet (Full Features)")
print("="*60)

en_alphas = [0.01, 0.1, 1.0, 10.0]
en_l1_ratios = [0.2, 0.5, 0.8]
best_en_a = None
best_en_l1 = None
best_en_rmse = 999999

i=0
while i<len(en_alphas):
    j=0
    while j<len(en_l1_ratios):
        a = en_alphas[i]
        l1 = en_l1_ratios[j]
        rmse_vals=[]
        splits = list(tscv.split(X_full_train_s))
        k=0
        while k<len(splits):
            tr_idx, val_idx = splits[k]
            mdl = ElasticNet(alpha=a, l1_ratio=l1, max_iter=50000)
            mdl.fit(X_full_train_s[tr_idx], y_train[tr_idx])
            p = mdl.predict(X_full_train_s[val_idx])
            rmse_vals.append(np.sqrt(mean_squared_error(y_train[val_idx], p)))
            k+=1
        avg = np.mean(rmse_vals)
        std = np.std(rmse_vals)
        print(f"  a={a:>6.2f} l1={l1:.1f}  RMSE={avg:.2f} +/- {std:.2f}")
        if avg<best_en_rmse:
            best_en_rmse=avg
            best_en_a=a
            best_en_l1=l1
        j+=1
    i+=1

print(f"  >> best: alpha={best_en_a}, l1_ratio={best_en_l1}")
en_mdl = ElasticNet(alpha=best_en_a, l1_ratio=best_en_l1, max_iter=50000)
en_mdl.fit(X_full_train_s, y_train)
en_test_pred = en_mdl.predict(X_full_test_s)
en_train_pred = en_mdl.predict(X_full_train_s)

r1,r2_v,r3,r4 = calc_metrics(y_test, en_test_pred)
print(f"\n  ElasticNet TEST: RMSE={r1:.2f}  MAE={r2_v:.2f}  R2={r3:.4f}  MAPE={r4:.2f}%")

# ================================================================
# coefficients
# ================================================================
coef_df = pd.DataFrame({
    'feature': full_cols,
    'ridge_coef': ridge_full.coef_,
    'lasso_coef': lasso_mdl.coef_,
    'elasticnet_coef': en_mdl.coef_
})
print("\n--- Coefficients (top 15 by |ridge|) ---")
coef_df['abs_ridge'] = np.abs(coef_df['ridge_coef'])
coef_df = coef_df.sort_values('abs_ridge', ascending=False)
print(coef_df.head(15).to_string(index=False))

# nonzero lasso coefs
nonzero = (coef_df['lasso_coef']!=0).sum()
print(f"\nLasso nonzero features: {nonzero}/{len(full_cols)}")

# ================================================================
# PLOTS
# ================================================================
plt.figure(figsize=(16,6))
plt.plot(dates_test, y_test, label='Actual', linewidth=0.4, alpha=0.8)
plt.plot(dates_test, full_test_pred, label='Ridge (full)', linewidth=0.4, alpha=0.7)
plt.plot(dates_test, wx_test_pred, label='Ridge (weather-only)', linewidth=0.4, alpha=0.5, color='red')
plt.title('Linear Models: Predicted vs Actual (Test Set)')
plt.xlabel('Date')
plt.ylabel('MW')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '15_ridge_pred_vs_actual.png'), dpi=200)
plt.close()

# wx residuals
resid_wx = y_test - wx_test_pred
plt.figure(figsize=(16,6))
plt.scatter(dates_test, resid_wx, s=0.4, alpha=0.2, color='blue')
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Weather-Only Ridge: Test Residuals')
plt.xlabel('Date')
plt.ylabel('Residual (MW)')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '16_ridge_wx_residuals.png'), dpi=200)
plt.close()

# full ridge residuals
resid_full = y_test - full_test_pred
plt.figure(figsize=(16,6))
plt.scatter(dates_test, resid_full, s=0.4, alpha=0.2, color='green')
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Full Ridge: Test Residuals')
plt.xlabel('Date')
plt.ylabel('Residual (MW)')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '17_ridge_full_residuals.png'), dpi=200)
plt.close()

# coef importance
top20 = coef_df.head(20)
plt.figure(figsize=(10,8))
plt.barh(range(len(top20)), top20['ridge_coef'].values)
plt.yticks(range(len(top20)), top20['feature'].values)
plt.title('Ridge Coefficients (Top 20 by magnitude)')
plt.xlabel('Coefficient (standardized)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '18_ridge_coefficients.png'), dpi=200)
plt.close()

# cv alpha curve
cv_df = pd.DataFrame(cv_results_wx)
plt.figure(figsize=(8,5))
plt.errorbar(cv_df['alpha'], cv_df['rmse'], yerr=cv_df['std'], marker='o', capsize=4)
plt.xscale('log')
plt.title('Weather-Only Ridge: CV RMSE vs Alpha')
plt.xlabel('Alpha (log scale)')
plt.ylabel('CV RMSE')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '19_ridge_cv_curve.png'), dpi=200)
plt.close()

print("\nplots saved")

# save preds
test_out = pd.DataFrame({
    'datetime': dates_test,
    'actual': y_test,
    'ridge_wx_pred': wx_test_pred,
    'ridge_full_pred': full_test_pred,
    'lasso_pred': lasso_test_pred,
    'elasticnet_pred': en_test_pred
})
test_out.to_csv(os.path.join(PRED_DIR, 'ridge_test_preds.csv'), index=False)

train_out = pd.DataFrame({
    'datetime': dates_train,
    'actual': y_train,
    'ridge_wx_pred': wx_train_pred,
    'ridge_full_pred': full_train_pred,
    'lasso_pred': lasso_train_pred,
    'elasticnet_pred': en_train_pred
})
train_out.to_csv(os.path.join(PRED_DIR, 'ridge_train_preds.csv'), index=False)

# save models
with open(os.path.join(MODEL_DIR, 'ridge_wx.pkl'), 'wb') as f:
    pickle.dump({'model': ridge_wx, 'scaler': scaler_wx, 'features': weather_cols}, f)
with open(os.path.join(MODEL_DIR, 'ridge_full.pkl'), 'wb') as f:
    pickle.dump({'model': ridge_full, 'scaler': scaler_full, 'features': full_cols}, f)

print("saved predictions and models")
