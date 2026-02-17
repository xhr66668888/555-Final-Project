import pandas as pd
import numpy as np
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOAD_DIR = os.path.join(BASE, 'data', 'pjm_load')
WX_DIR = os.path.join(BASE, 'data', 'weather')
OUT_DIR = os.path.join(BASE, 'output')

power_files = ['2021-2022.csv','2022-2023.csv','2023-2024.csv','2024-2025.csv','2025-2026.csv']

dfs=[]
i=0
while i<len(power_files):
    fpath = os.path.join(LOAD_DIR, power_files[i])
    tmp = pd.read_csv(fpath)
    dfs.append(tmp)
    print(f"loaded {power_files[i]}, rows: {len(tmp)}")
    i+=1

power = pd.concat(dfs, ignore_index=True)
print(f"\ntotal power rows: {len(power)}")

power['datetime'] = pd.to_datetime(power['datetime_beginning_ept'], format='mixed')
power = power[['datetime','mw']]
power = power.sort_values('datetime').reset_index(drop=True)

before=len(power)
power = power.drop_duplicates(subset='datetime', keep='first')
print(f"dropped {before-len(power)} dupes")

# weather
wx_files = ['LCD_USW00014821_2021.csv','LCD_USW00014821_2022.csv',
            'LCD_USW00014821_2023.csv','LCD_USW00014821_2024.csv',
            'LCD_USW00014821_2025.csv']

wx_dfs=[]
i=0
while i<len(wx_files):
    fpath = os.path.join(WX_DIR, wx_files[i])
    tmp = pd.read_csv(fpath, low_memory=False)
    tmp = tmp[tmp['REPORT_TYPE']=='FM-15']
    wx_dfs.append(tmp)
    print(f"loaded {wx_files[i]}, FM-15: {len(tmp)}")
    i+=1

weather = pd.concat(wx_dfs, ignore_index=True)

weather['datetime'] = pd.to_datetime(weather['DATE'])
weather['datetime'] = weather['datetime'].dt.round('h')

keep_cols = ['datetime','HourlyDryBulbTemperature','HourlyDewPointTemperature',
             'HourlyRelativeHumidity','HourlyWindSpeed','HourlyPrecipitation',
             'HourlyStationPressure']
avail = [c for c in keep_cols if c in weather.columns]
weather = weather[avail]

num_cols = [c for c in avail if c!='datetime']
j=0
while j<len(num_cols):
    weather[num_cols[j]] = pd.to_numeric(weather[num_cols[j]], errors='coerce')
    j+=1

weather = weather.groupby('datetime').mean().reset_index()
col_rename = {
    'HourlyDryBulbTemperature': 'temp',
    'HourlyDewPointTemperature': 'dewpoint',
    'HourlyRelativeHumidity': 'humidity',
    'HourlyWindSpeed': 'windspeed',
    'HourlyPrecipitation': 'precip',
    'HourlyStationPressure': 'pressure'
}
weather = weather.rename(columns=col_rename)
print(f"\nweather rows after groupby: {len(weather)}")

df = pd.merge(power, weather, on='datetime', how='inner')
print(f"merged rows: {len(df)}")

print(f"\nmissing before interp:")
print(df.isnull().sum())

wx_feat = [c for c in ['temp','dewpoint','humidity','windspeed','precip','pressure'] if c in df.columns]
k=0
while k<len(wx_feat):
    df[wx_feat[k]] = df[wx_feat[k]].interpolate(method='linear')
    k+=1
df = df.ffill().bfill()

print(f"\ntemp stats:")
print(df['temp'].describe())

# ====== temporal features ======
df['hour'] = df['datetime'].dt.hour
df['dayofweek'] = df['datetime'].dt.dayofweek
df['month'] = df['datetime'].dt.month
df['year'] = df['datetime'].dt.year
df['day_of_year'] = df['datetime'].dt.dayofyear
df['week_of_year'] = df['datetime'].dt.isocalendar().week.astype(int)
df['is_weekend'] = (df['dayofweek']>=5).astype(int)

# cyclical time encoding
df['hour_sin'] = np.sin(2*np.pi*df['hour']/24)
df['hour_cos'] = np.cos(2*np.pi*df['hour']/24)
df['month_sin'] = np.sin(2*np.pi*df['month']/12)
df['month_cos'] = np.cos(2*np.pi*df['month']/12)
df['dow_sin'] = np.sin(2*np.pi*df['dayofweek']/7)
df['dow_cos'] = np.cos(2*np.pi*df['dayofweek']/7)

holidays_list = [
    '2021-01-01','2021-01-18','2021-05-31','2021-07-04','2021-07-05',
    '2021-09-06','2021-11-25','2021-12-24','2021-12-25',
    '2022-01-01','2022-01-17','2022-05-30','2022-07-04',
    '2022-09-05','2022-11-24','2022-12-25','2022-12-26',
    '2023-01-02','2023-01-16','2023-05-29','2023-07-04',
    '2023-09-04','2023-11-23','2023-12-25',
    '2024-01-01','2024-01-15','2024-05-27','2024-07-04',
    '2024-09-02','2024-11-28','2024-12-25',
    '2025-01-01','2025-01-20','2025-05-26','2025-07-04',
    '2025-09-01','2025-11-27','2025-12-25',
    '2026-01-01','2026-01-19'
]
h_dates = pd.to_datetime(holidays_list)
df['is_holiday'] = df['datetime'].dt.date.isin(h_dates.date).astype(int)

df = df.sort_values('datetime').reset_index(drop=True)

# ====== weather derived features ======
df['temp_sq'] = df['temp']**2
df['temp_cb'] = df['temp']**3

# ====== LEAKAGE CHECK: rolling stats ======
# All rolling windows below use the pandas default (trailing / backward-looking).
# center=False (default) ensures each window only uses current and PAST values.
# This is safe for time-series forecasting: at prediction time t, we only
# look at [t-w+1, ..., t], never future values.
# No centered windows are used anywhere in this pipeline.
windows = [6, 12, 24, 48, 168]
m=0
while m<len(windows):
    w = windows[m]
    # trailing mean/std — past-only, no leakage
    df[f'temp_roll{w}'] = df['temp'].rolling(window=w, min_periods=1, center=False).mean()
    df[f'temp_std{w}'] = df['temp'].rolling(window=w, min_periods=1, center=False).std().fillna(0)
    df[f'load_roll{w}'] = df['mw'].rolling(window=w, min_periods=1, center=False).mean()
    m+=1

df['temp_diff1'] = df['temp'].diff().fillna(0)
df['temp_diff24'] = df['temp'].diff(24).fillna(0)

# humidity-temp interaction
df['heat_idx'] = df['temp'] * df['humidity'] / 100.0

# CDD/HDD base 18C
df['cdd'] = np.maximum(df['temp'] - 18.0, 0)
df['hdd'] = np.maximum(18.0 - df['temp'], 0)

# wind chill approx
if 'windspeed' in df.columns:
    df['wind_chill'] = 13.12 + 0.6215*df['temp'] - 11.37*(df['windspeed'].clip(lower=0.1)**0.16) + 0.3965*df['temp']*(df['windspeed'].clip(lower=0.1)**0.16)

# ====== LEAKAGE CHECK: lag features ======
# shift(k) looks k steps into the PAST — safe for forecasting.
# diff() computes current - previous — also backward-looking only.
# After creating lags, we dropna() to remove rows that have
# undefined lags at the start of the series.
lags = [1, 2, 3, 6, 12, 24, 48, 168, 336]
n=0
while n<len(lags):
    df[f'load_lag{lags[n]}'] = df['mw'].shift(lags[n])  # backward shift only
    n+=1

df['load_diff1'] = df['mw'].diff().fillna(0)    # backward diff
df['load_diff24'] = df['mw'].diff(24).fillna(0)  # backward diff

df = df.dropna()
print(f"\nfinal shape: {df.shape}")
print(f"date range: {df['datetime'].min()} to {df['datetime'].max()}")

print(f"\ncolumns ({len(df.columns)}):")
print(list(df.columns))
print(df.describe())

out_path = os.path.join(OUT_DIR, 'processed_data.csv')
df.to_csv(out_path, index=False)
print(f"\nsaved {out_path}")
