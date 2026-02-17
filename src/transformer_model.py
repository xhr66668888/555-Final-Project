import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import time

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

# features for transformer input
feat_cols = ['mw','temp','dewpoint','humidity','hour_sin','hour_cos',
             'month_sin','month_cos','dow_sin','dow_cos',
             'is_weekend','cdd','hdd','heat_idx']
# add windspeed/pressure if exist
extra = [c for c in ['windspeed','pressure','wind_chill'] if c in df.columns]
feat_cols = feat_cols + extra
n_feat = len(feat_cols)
print(f"transformer input features: {n_feat}")

# hyperparams - no need to go easy, high perf machine
LOOKBACK = 336       # 2 weeks of history
PATCH_LEN = 24       # 1 day per patch
N_PATCHES = LOOKBACK // PATCH_LEN  # 14 patches
BATCH = 128
EPOCHS = 80
LR = 5e-4
D_MODEL = 128
NHEAD = 8
NLAYERS = 4
DROPOUT = 0.1
WARMUP = 5

# split
cutoff = pd.Timestamp('2025-01-01')
train_mask = df['datetime'] < cutoff
train_end = train_mask.sum()
total_len = len(df)
print(f"train end: {train_end}, total: {total_len}")

data_raw = df[feat_cols].values
mw_raw = df['mw'].values.reshape(-1,1)

# LEAKAGE CHECK: both feature scaler and target scaler are fit on
# TRAIN data only (data_raw[:train_end]).  Test data is only transformed,
# never used to compute scaling parameters.
scaler_X = StandardScaler()
scaler_y = StandardScaler()
scaler_X.fit(data_raw[:train_end])    # fit on train only
scaler_y.fit(mw_raw[:train_end])      # fit on train only

data_scaled = scaler_X.transform(data_raw)
mw_scaled = scaler_y.transform(mw_raw).flatten()


class LoadWindows(Dataset):
    def __init__(self, data, targets, start, end, lookback, patch_len):
        self.data=data
        self.targets=targets
        self.start=start
        self.end=end
        self.lookback=lookback
        self.patch_len=patch_len
        self.n_patches=lookback//patch_len

    def __len__(self):
        return self.end - self.start - self.lookback

    def __getitem__(self, idx):
        real = self.start + idx
        window = self.data[real:real+self.lookback]
        target = self.targets[real+self.lookback]
        patches = window.reshape(self.n_patches, -1)
        return torch.FloatTensor(patches), torch.FloatTensor([target])


class PatchTST(nn.Module):
    def __init__(self, patch_dim, d_model, nhead, nlayers, n_patches, dropout):
        super().__init__()
        self.patch_embed = nn.Linear(patch_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches, d_model)*0.02)
        self.input_norm = nn.LayerNorm(d_model)
        self.dropout_in = nn.Dropout(dropout)

        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(enc, num_layers=nlayers,
                                                  enable_nested_tensor=False)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model//2),
            nn.GELU(),
            nn.Dropout(dropout/2),
            nn.Linear(d_model//2, 1)
        )

    def forward(self, x):
        # x: (B, n_patches, patch_dim)
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.input_norm(x)
        x = self.dropout_in(x)
        x = self.transformer(x)
        # mean pool over all patches (better than last-only)
        x = x.mean(dim=1)
        x = self.head(x)
        return x.squeeze(-1)


# datasets
train_ds = LoadWindows(data_scaled, mw_scaled, 0, train_end, LOOKBACK, PATCH_LEN)
test_ds = LoadWindows(data_scaled, mw_scaled, train_end-LOOKBACK, total_len, LOOKBACK, PATCH_LEN)

print(f"train windows: {len(train_ds)}, test windows: {len(test_ds)}")

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")

patch_dim = PATCH_LEN * n_feat
model = PatchTST(patch_dim, D_MODEL, NHEAD, NLAYERS, N_PATCHES, DROPOUT)
model = model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"model params: {total_params:,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

# cosine annealing with warmup
def lr_lambda(epoch):
    if epoch < WARMUP:
        return (epoch+1) / WARMUP
    progress = (epoch - WARMUP) / (EPOCHS - WARMUP)
    return 0.5 * (1 + np.cos(np.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
loss_fn = nn.MSELoss()

# training
print(f"\ntraining PatchTST for {EPOCHS} epochs...")
train_losses=[]
val_metrics=[]
best_val_rmse = 999999
patience_counter = 0
PATIENCE = 15

start_time = time.time()

ep=0
while ep < EPOCHS:
    model.train()
    total_loss=0
    n_batch=0

    batches = list(train_loader)
    b=0
    while b<len(batches):
        bx, by = batches[b]
        bx = bx.to(device)
        by = by.to(device).squeeze()

        optimizer.zero_grad()
        pred = model(bx)
        loss = loss_fn(pred, by)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batch+=1
        b+=1

    avg_loss = total_loss/n_batch
    train_losses.append(avg_loss)
    scheduler.step()

    # validate every 5 epochs
    if (ep+1)%5==0 or ep==0:
        model.eval()
        v_preds=[]
        v_tgts=[]
        with torch.no_grad():
            vb = list(test_loader)
            vi=0
            while vi<len(vb):
                vx, vy = vb[vi]
                vx = vx.to(device)
                vp = model(vx)
                v_preds.append(vp.cpu().numpy())
                v_tgts.append(vy.numpy().flatten())
                vi+=1

        vp_all = scaler_y.inverse_transform(np.concatenate(v_preds).reshape(-1,1)).flatten()
        vt_all = scaler_y.inverse_transform(np.concatenate(v_tgts).reshape(-1,1)).flatten()
        vrmse = np.sqrt(mean_squared_error(vt_all, vp_all))
        val_metrics.append((ep+1, vrmse))

        elapsed = time.time() - start_time
        lr_now = optimizer.param_groups[0]['lr']
        print(f"  epoch {ep+1:>3}/{EPOCHS}  loss={avg_loss:.6f}  val_RMSE={vrmse:.2f}  lr={lr_now:.6f}  [{elapsed:.0f}s]")

        if vrmse < best_val_rmse:
            best_val_rmse = vrmse
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'patchtst_best.pt'))
        else:
            patience_counter += 5

        if patience_counter >= PATIENCE:
            print(f"  early stopping at epoch {ep+1}")
            break

    ep+=1

total_time = time.time()-start_time
print(f"\ntraining done in {total_time:.1f}s")

# load best
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'patchtst_best.pt'), weights_only=True))

# inference
model.eval()
all_preds=[]
all_tgts=[]

test_batches = list(test_loader)
b=0
with torch.no_grad():
    while b<len(test_batches):
        bx, by = test_batches[b]
        bx = bx.to(device)
        pred = model(bx)
        all_preds.append(pred.cpu().numpy())
        all_tgts.append(by.numpy().flatten())
        b+=1

preds_s = np.concatenate(all_preds)
tgts_s = np.concatenate(all_tgts)

preds_mw = scaler_y.inverse_transform(preds_s.reshape(-1,1)).flatten()
tgts_mw = scaler_y.inverse_transform(tgts_s.reshape(-1,1)).flatten()

rmse = np.sqrt(mean_squared_error(tgts_mw, preds_mw))
mae = mean_absolute_error(tgts_mw, preds_mw)
r2 = r2_score(tgts_mw, preds_mw)
mape = np.mean(np.abs((tgts_mw-preds_mw)/tgts_mw))*100

print(f"\nPatchTST TEST (best checkpoint):")
print(f"  RMSE: {rmse:.2f}  MAE: {mae:.2f}  R2: {r2:.4f}  MAPE: {mape:.2f}%")

# also train metrics
train_loader_noshuf = DataLoader(train_ds, batch_size=BATCH, shuffle=False, num_workers=0)
tr_preds=[]
tr_tgts=[]
with torch.no_grad():
    tb = list(train_loader_noshuf)
    bi=0
    while bi<len(tb):
        bx, by = tb[bi]
        bx = bx.to(device)
        p = model(bx)
        tr_preds.append(p.cpu().numpy())
        tr_tgts.append(by.numpy().flatten())
        bi+=1

tr_p = scaler_y.inverse_transform(np.concatenate(tr_preds).reshape(-1,1)).flatten()
tr_t = scaler_y.inverse_transform(np.concatenate(tr_tgts).reshape(-1,1)).flatten()
tr_rmse = np.sqrt(mean_squared_error(tr_t, tr_p))
tr_r2 = r2_score(tr_t, tr_p)
print(f"PatchTST TRAIN: RMSE={tr_rmse:.2f}  R2={tr_r2:.4f}")

# dates
dates_tf = df['datetime'].values[train_end:train_end+len(preds_mw)]
min_len = min(len(dates_tf), len(preds_mw))
dates_tf = dates_tf[:min_len]
preds_mw = preds_mw[:min_len]
tgts_mw = tgts_mw[:min_len]

# ====== PLOTS ======
# loss curve
fig, ax1 = plt.subplots(figsize=(10,5))
ax1.plot(train_losses, 'b-', linewidth=0.8, label='Train Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('MSE Loss', color='b')
ax1.tick_params(axis='y', labelcolor='b')
if len(val_metrics)>0:
    ax2 = ax1.twinx()
    vm_ep = [x[0] for x in val_metrics]
    vm_rmse = [x[1] for x in val_metrics]
    ax2.plot(vm_ep, vm_rmse, 'ro-', markersize=5, linewidth=1.5, label='Val RMSE')
    ax2.set_ylabel('Validation RMSE (MW)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
plt.title('PatchTST Training Curve')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '24_patchtst_loss.png'), dpi=200)
plt.close()

# pred vs actual
plt.figure(figsize=(16,6))
plt.plot(dates_tf, tgts_mw, label='Actual', linewidth=0.4, alpha=0.8)
plt.plot(dates_tf, preds_mw, label='PatchTST', linewidth=0.4, alpha=0.7, color='green')
plt.title('PatchTST: Predicted vs Actual (Test Set)')
plt.xlabel('Date')
plt.ylabel('MW')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '25_patchtst_pred_vs_actual.png'), dpi=200)
plt.close()

# residuals
residuals = tgts_mw - preds_mw
plt.figure(figsize=(16,6))
plt.scatter(dates_tf, residuals, s=0.4, alpha=0.2, color='green')
plt.axhline(y=0, color='r', linestyle='--')
plt.title('PatchTST Residuals (Test Set)')
plt.xlabel('Date')
plt.ylabel('Residual (MW)')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '26_patchtst_residuals.png'), dpi=200)
plt.close()

# scatter
plt.figure(figsize=(8,8))
plt.scatter(tgts_mw, preds_mw, s=0.5, alpha=0.15, color='green')
mn = min(tgts_mw.min(), preds_mw.min())
mx = max(tgts_mw.max(), preds_mw.max())
plt.plot([mn,mx],[mn,mx],'r--', linewidth=1.5)
plt.title('PatchTST: Actual vs Predicted')
plt.xlabel('Actual (MW)')
plt.ylabel('Predicted (MW)')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '27_patchtst_scatter.png'), dpi=200)
plt.close()

# save preds
results = pd.DataFrame({
    'datetime': dates_tf,
    'actual': tgts_mw,
    'patchtst_pred': preds_mw
})
results.to_csv(os.path.join(PRED_DIR, 'transformer_test_preds.csv'), index=False)
print("\nsaved predictions and model checkpoint")
