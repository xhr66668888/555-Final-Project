import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import pickle
import json
import itertools

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

# all features except datetime and mw
skip = ['datetime','mw','week_avg']
feature_cols = [c for c in df.columns if c not in skip]
print(f"features: {len(feature_cols)}")

X = df[feature_cols].values
y = df['mw'].values
dates = df['datetime'].values

cutoff = pd.Timestamp('2025-01-01')
train_mask = df['datetime'] < cutoff
test_mask = df['datetime'] >= cutoff

X_train=X[train_mask]
X_test=X[test_mask]
y_train=y[train_mask]
y_test=y[test_mask]
dates_test=dates[test_mask]
dates_train=dates[train_mask]
print(f"train: {len(X_train)}, test: {len(X_test)}")

# exhaustive grid search
n_estimators_list = [500, 1000, 1500, 2000]
max_depth_list = [4, 6, 8, 10, 12]
learning_rate_list = [0.01, 0.02, 0.05, 0.1]
subsample_list = [0.7, 0.8, 0.9]
colsample_list = [0.7, 0.8, 0.9, 1.0]
min_child_list = [1, 3, 5, 10]
reg_alpha_list = [0, 0.1, 1.0]
reg_lambda_list = [1.0, 5.0, 10.0]

# too many combos for full grid, do staged approach
# Stage 1: rough search on key params
print("\n=== Stage 1: coarse grid on n_est/depth/lr ===")
tscv = TimeSeriesSplit(n_splits=5)

stage1_grid = []
for n in [500, 1000, 2000]:
    for d in [4, 6, 8, 10]:
        for lr in [0.01, 0.05, 0.1]:
            stage1_grid.append({'n_estimators':n, 'max_depth':d, 'learning_rate':lr})

best_params = None
best_score = 999999
all_cv=[]

i=0
while i<len(stage1_grid):
    p = stage1_grid[i]
    scores=[]
    splits = list(tscv.split(X_train))
    j=0
    while j<len(splits):
        tr_idx, val_idx = splits[j]
        mdl = xgb.XGBRegressor(
            n_estimators=p['n_estimators'],
            max_depth=p['max_depth'],
            learning_rate=p['learning_rate'],
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        mdl.fit(X_train[tr_idx], y_train[tr_idx],
                eval_set=[(X_train[val_idx], y_train[val_idx])],
                verbose=False)
        pred = mdl.predict(X_train[val_idx])
        rmse = np.sqrt(mean_squared_error(y_train[val_idx], pred))
        scores.append(rmse)
        j+=1

    avg=np.mean(scores)
    std=np.std(scores)
    all_cv.append({'params':p, 'rmse':avg, 'std':std})
    if i%10==0:
        print(f"  [{i+1}/{len(stage1_grid)}] n={p['n_estimators']}, d={p['max_depth']}, lr={p['learning_rate']} => {avg:.2f} +/- {std:.2f}")

    if avg<best_score:
        best_score=avg
        best_params=p.copy()
    i+=1

print(f"\nStage 1 best: {best_params} => RMSE={best_score:.2f}")

# Stage 2: fine tune subsample, colsample, regularization
print("\n=== Stage 2: fine-tuning subsample/colsample/reg ===")
stage2_grid = []
for sub in [0.7, 0.8, 0.9]:
    for col in [0.7, 0.8, 0.9, 1.0]:
        for mcw in [1, 3, 5]:
            for ra in [0, 0.1, 1.0]:
                for rl in [1, 5, 10]:
                    stage2_grid.append({'subsample':sub, 'colsample_bytree':col,
                                        'min_child_weight':mcw, 'reg_alpha':ra, 'reg_lambda':rl})

best_s2 = None
best_score2 = 999999

i=0
while i<len(stage2_grid):
    p2 = stage2_grid[i]
    scores=[]
    splits = list(tscv.split(X_train))
    j=0
    while j<len(splits):
        tr_idx, val_idx = splits[j]
        mdl = xgb.XGBRegressor(
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            learning_rate=best_params['learning_rate'],
            subsample=p2['subsample'],
            colsample_bytree=p2['colsample_bytree'],
            min_child_weight=p2['min_child_weight'],
            reg_alpha=p2['reg_alpha'],
            reg_lambda=p2['reg_lambda'],
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        mdl.fit(X_train[tr_idx], y_train[tr_idx],
                eval_set=[(X_train[val_idx], y_train[val_idx])],
                verbose=False)
        pred = mdl.predict(X_train[val_idx])
        rmse = np.sqrt(mean_squared_error(y_train[val_idx], pred))
        scores.append(rmse)
        j+=1

    avg=np.mean(scores)
    if i%50==0:
        print(f"  [{i+1}/{len(stage2_grid)}] sub={p2['subsample']}, col={p2['colsample_bytree']} => {avg:.2f}")
    if avg<best_score2:
        best_score2=avg
        best_s2=p2.copy()
    i+=1

print(f"\nStage 2 best: {best_s2} => RMSE={best_score2:.2f}")

# final params
final_params = {**best_params, **best_s2}
print(f"\nFinal params: {final_params}")

# final model
xgb_model = xgb.XGBRegressor(
    n_estimators=final_params['n_estimators'],
    max_depth=final_params['max_depth'],
    learning_rate=final_params['learning_rate'],
    subsample=final_params['subsample'],
    colsample_bytree=final_params['colsample_bytree'],
    min_child_weight=final_params['min_child_weight'],
    reg_alpha=final_params['reg_alpha'],
    reg_lambda=final_params['reg_lambda'],
    random_state=42,
    n_jobs=-1,
    verbosity=0
)
xgb_model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)

xgb_test_pred = xgb_model.predict(X_test)
xgb_train_pred = xgb_model.predict(X_train)

rmse = np.sqrt(mean_squared_error(y_test, xgb_test_pred))
mae = mean_absolute_error(y_test, xgb_test_pred)
r2 = r2_score(y_test, xgb_test_pred)
mape = np.mean(np.abs((y_test-xgb_test_pred)/y_test))*100

print(f"\nXGBoost TEST: RMSE={rmse:.2f}  MAE={mae:.2f}  R2={r2:.4f}  MAPE={mape:.2f}%")

train_rmse = np.sqrt(mean_squared_error(y_train, xgb_train_pred))
train_r2 = r2_score(y_train, xgb_train_pred)
print(f"XGBoost TRAIN: RMSE={train_rmse:.2f}  R2={train_r2:.4f}")

# k-fold test (rolling window CV on full data for reporting)
print("\n--- Rolling Window CV (full dataset) ---")
tscv_full = TimeSeriesSplit(n_splits=5)
cv_rmse=[]
cv_mae=[]
cv_r2=[]
splits = list(tscv_full.split(X))
k=0
while k<len(splits):
    tr_idx, val_idx = splits[k]
    tmp_mdl = xgb.XGBRegressor(**{k2:v for k2,v in final_params.items()},
                                random_state=42, n_jobs=-1, verbosity=0)
    tmp_mdl.fit(X[tr_idx], y[tr_idx], verbose=False)
    tmp_pred = tmp_mdl.predict(X[val_idx])
    cv_rmse.append(np.sqrt(mean_squared_error(y[val_idx], tmp_pred)))
    cv_mae.append(mean_absolute_error(y[val_idx], tmp_pred))
    cv_r2.append(r2_score(y[val_idx], tmp_pred))
    print(f"  fold {k+1}: RMSE={cv_rmse[-1]:.2f}  MAE={cv_mae[-1]:.2f}  R2={cv_r2[-1]:.4f}")
    k+=1

print(f"  mean RMSE: {np.mean(cv_rmse):.2f} +/- {np.std(cv_rmse):.2f}")
print(f"  mean MAE:  {np.mean(cv_mae):.2f} +/- {np.std(cv_mae):.2f}")
print(f"  mean R2:   {np.mean(cv_r2):.4f} +/- {np.std(cv_r2):.4f}")

# feature importance
importance = xgb_model.feature_importances_
imp_df = pd.DataFrame({'feature': feature_cols, 'importance': importance})
imp_df = imp_df.sort_values('importance', ascending=True)

plt.figure(figsize=(10,max(8, len(feature_cols)*0.25)))
top30 = imp_df.tail(30)
plt.barh(top30['feature'], top30['importance'], color='steelblue', edgecolor='navy', linewidth=0.3)
plt.title('XGBoost Feature Importance (Top 30)', fontsize=14)
plt.xlabel('Importance (Gain)')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '20_xgb_feature_importance.png'), dpi=200)
plt.close()
print("saved feature importance")

# pred vs actual
plt.figure(figsize=(16,6))
plt.plot(dates_test, y_test, label='Actual', linewidth=0.4, alpha=0.8)
plt.plot(dates_test, xgb_test_pred, label='XGBoost', linewidth=0.4, alpha=0.7, color='orange')
plt.title('XGBoost: Predicted vs Actual (Test Set)')
plt.xlabel('Date')
plt.ylabel('MW')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '21_xgb_pred_vs_actual.png'), dpi=200)
plt.close()

# residuals
residuals = y_test - xgb_test_pred
plt.figure(figsize=(16,6))
plt.scatter(dates_test, residuals, s=0.4, alpha=0.2, color='orange')
plt.axhline(y=0, color='r', linestyle='--')
plt.title('XGBoost Residuals (Test Set)')
plt.xlabel('Date')
plt.ylabel('Residual (MW)')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '22_xgb_residuals.png'), dpi=200)
plt.close()

# scatter pred vs actual
plt.figure(figsize=(8,8))
plt.scatter(y_test, xgb_test_pred, s=0.5, alpha=0.15)
mn = min(y_test.min(), xgb_test_pred.min())
mx = max(y_test.max(), xgb_test_pred.max())
plt.plot([mn,mx],[mn,mx],'r--', linewidth=1.5)
plt.title('XGBoost: Actual vs Predicted')
plt.xlabel('Actual (MW)')
plt.ylabel('Predicted (MW)')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '23_xgb_scatter_actual_pred.png'), dpi=200)
plt.close()

# save
test_res = pd.DataFrame({'datetime': dates_test, 'actual': y_test, 'xgb_pred': xgb_test_pred})
test_res.to_csv(os.path.join(PRED_DIR, 'xgb_test_preds.csv'), index=False)

train_res = pd.DataFrame({'datetime': dates_train, 'actual': y_train, 'xgb_pred': xgb_train_pred})
train_res.to_csv(os.path.join(PRED_DIR, 'xgb_train_preds.csv'), index=False)

xgb_model.save_model(os.path.join(MODEL_DIR, 'xgboost_model.json'))

with open(os.path.join(MODEL_DIR, 'xgb_params.json'), 'w') as f:
    json.dump(final_params, f, indent=2)

print("\nsaved predictions, model, and params")
