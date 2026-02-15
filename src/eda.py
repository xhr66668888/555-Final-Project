import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE, 'output')
PLOT_DIR = os.path.join(OUT_DIR, 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(OUT_DIR, 'processed_data.csv'))
df['datetime'] = pd.to_datetime(df['datetime'])
print(f"loaded {len(df)} rows")

# 1 full timeseries
plt.figure(figsize=(16,5))
plt.plot(df['datetime'], df['mw'], linewidth=0.2, alpha=0.6, color='navy')
plt.title('AEP Ohio Hourly Load (MW) — Full Period', fontsize=14)
plt.xlabel('Date')
plt.ylabel('MW')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '01_load_timeseries.png'), dpi=200)
plt.close()
print("saved 01")

# 2 monthly avg by year
monthly = df.groupby([df['year'], df['month']])['mw'].mean().reset_index()
monthly.columns = ['year','month','avg_mw']

plt.figure(figsize=(14,7))
years = sorted(df['year'].unique())
i=0
while i<len(years):
    yr = years[i]
    tmp = monthly[monthly['year']==yr]
    plt.plot(tmp['month'], tmp['avg_mw'], marker='o', linewidth=2, label=str(yr))
    i+=1
plt.title('Monthly Average Load by Year', fontsize=14)
plt.xlabel('Month')
plt.ylabel('Average MW')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '02_monthly_avg_by_year.png'), dpi=200)
plt.close()
print("saved 02")

# 3 load vs temp scatter colored by year
plt.figure(figsize=(12,7))
cmap = {2021:'#1f77b4', 2022:'#2ca02c', 2023:'#ff7f0e', 2024:'#d62728', 2025:'#9467bd', 2026:'#8c564b'}
i=0
while i<len(years):
    yr = years[i]
    tmp = df[df['year']==yr]
    c = cmap.get(yr, 'gray')
    plt.scatter(tmp['temp'], tmp['mw'], s=0.4, alpha=0.15, color=c, label=str(yr))
    i+=1
plt.title('Load vs Temperature (colored by year)', fontsize=14)
plt.xlabel('Temperature (°C)')
plt.ylabel('MW')
plt.legend(markerscale=15, fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '03_load_vs_temp.png'), dpi=200)
plt.close()
print("saved 03")

# 4 correlation heatmap (key features only)
feat_cols = ['mw','temp','dewpoint','humidity','hour','dayofweek','month',
             'is_weekend','cdd','hdd','heat_idx',
             'load_lag1','load_lag24','load_lag168','temp_sq']
avail = [c for c in feat_cols if c in df.columns]
corr = df[avail].corr()
plt.figure(figsize=(14,11))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True, linewidths=0.5)
plt.title('Feature Correlation Matrix', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '04_correlation_heatmap.png'), dpi=200)
plt.close()
print("saved 04")

# 5 hourly profile
plt.figure(figsize=(10,6))
hourly = df.groupby('hour')['mw'].mean()
plt.bar(hourly.index, hourly.values, color='steelblue', edgecolor='navy', linewidth=0.5)
plt.title('Average Load by Hour of Day', fontsize=14)
plt.xlabel('Hour')
plt.ylabel('Average MW')
plt.xticks(range(0,24))
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '05_hourly_pattern.png'), dpi=200)
plt.close()
print("saved 05")

# 6 weekly pattern
plt.figure(figsize=(10,6))
day_names = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
daily = df.groupby('dayofweek')['mw'].mean()
plt.bar(range(7), daily.values, color='steelblue', edgecolor='navy', linewidth=0.5)
plt.xticks(range(7), day_names)
plt.title('Average Load by Day of Week', fontsize=14)
plt.ylabel('Average MW')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '06_weekly_pattern.png'), dpi=200)
plt.close()
print("saved 06")

# 7 load distribution
plt.figure(figsize=(10,6))
plt.hist(df['mw'], bins=100, edgecolor='black', alpha=0.7, linewidth=0.3)
plt.title('Load Distribution', fontsize=14)
plt.xlabel('MW')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '07_load_distribution.png'), dpi=200)
plt.close()
print("saved 07")

# 8 yearly avg load bars
yearly_avg = df.groupby('year')['mw'].mean()
plt.figure(figsize=(10,6))
colors = ['steelblue']*4 + ['orangered']*(len(yearly_avg)-4)
bars = plt.bar(yearly_avg.index.astype(str), yearly_avg.values, color=colors, edgecolor='black', linewidth=0.5)
plt.title('Annual Average Load — Growth Trend', fontsize=14)
plt.ylabel('Average MW')
vals = yearly_avg.values
idx=0
while idx<len(vals):
    plt.text(idx, vals[idx]+30, f'{vals[idx]:.0f}', ha='center', fontsize=11, fontweight='bold')
    idx+=1
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '08_yearly_avg_load.png'), dpi=200)
plt.close()
print("saved 08")

# 9 box plot by year
plt.figure(figsize=(12,6))
year_data=[]
year_labels=[]
i=0
while i<len(years):
    yr = years[i]
    year_data.append(df[df['year']==yr]['mw'].values)
    year_labels.append(str(yr))
    i+=1
bp = plt.boxplot(year_data, tick_labels=year_labels, patch_artist=True)
j=0
while j<len(bp['boxes']):
    bp['boxes'][j].set_facecolor('steelblue' if j<4 else 'orangered')
    bp['boxes'][j].set_alpha(0.6)
    j+=1
plt.title('Load Distribution by Year', fontsize=14)
plt.ylabel('MW')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '09_load_boxplot_yearly.png'), dpi=200)
plt.close()
print("saved 09")

# 10 temp distribution by year
plt.figure(figsize=(12,6))
i=0
while i<len(years):
    yr = years[i]
    tmp = df[df['year']==yr]['temp']
    plt.hist(tmp, bins=60, alpha=0.35, label=str(yr), density=True)
    i+=1
plt.title('Temperature Distribution by Year (normalized)', fontsize=14)
plt.xlabel('Temperature (°C)')
plt.ylabel('Density')
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '10_temp_dist_by_year.png'), dpi=200)
plt.close()
print("saved 10")

# 11 heatmap hour vs month
pivot = df.pivot_table(values='mw', index='hour', columns='month', aggfunc='mean')
plt.figure(figsize=(14,8))
sns.heatmap(pivot, cmap='YlOrRd', annot=True, fmt='.0f', linewidths=0.3)
plt.title('Average Load Heatmap — Hour vs Month', fontsize=14)
plt.ylabel('Hour of Day')
plt.xlabel('Month')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '11_load_heatmap_hour_month.png'), dpi=200)
plt.close()
print("saved 11")

# 12 CDD/HDD vs load
if 'cdd' in df.columns and 'hdd' in df.columns:
    fig, axes = plt.subplots(1,2, figsize=(14,6))
    axes[0].scatter(df['cdd'], df['mw'], s=0.3, alpha=0.1, color='red')
    axes[0].set_title('Load vs CDD (Cooling Degree)')
    axes[0].set_xlabel('CDD')
    axes[0].set_ylabel('MW')
    axes[1].scatter(df['hdd'], df['mw'], s=0.3, alpha=0.1, color='blue')
    axes[1].set_title('Load vs HDD (Heating Degree)')
    axes[1].set_xlabel('HDD')
    axes[1].set_ylabel('MW')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '12_cdd_hdd_vs_load.png'), dpi=200)
    plt.close()
    print("saved 12")

# 13 lag autocorrelation plot
fig, axes = plt.subplots(1,3, figsize=(16,4))
lags_show = [('load_lag1','t-1'), ('load_lag24','t-24'), ('load_lag168','t-168')]
q=0
while q<len(lags_show):
    col, lbl = lags_show[q]
    if col in df.columns:
        axes[q].scatter(df[col], df['mw'], s=0.2, alpha=0.05)
        axes[q].set_title(f'Load vs {lbl}')
        axes[q].set_xlabel(f'Load at {lbl}')
        axes[q].set_ylabel('Load (MW)')
    q+=1
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '13_lag_scatter.png'), dpi=200)
plt.close()
print("saved 13")

# 14 rolling load by year (weekly average)
plt.figure(figsize=(16,6))
df['week_avg'] = df['mw'].rolling(168, min_periods=24).mean()
i=0
while i<len(years):
    yr = years[i]
    tmp = df[df['year']==yr]
    plt.plot(tmp['datetime'], tmp['week_avg'], linewidth=0.8, alpha=0.8, label=str(yr))
    i+=1
plt.title('Weekly Rolling Average Load by Year', fontsize=14)
plt.xlabel('Date')
plt.ylabel('MW (168h rolling avg)')
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '14_rolling_load_by_year.png'), dpi=200)
plt.close()
print("saved 14")

n_plots = len([f for f in os.listdir(PLOT_DIR) if f.endswith('.png')])
print(f"\n=== EDA done, {n_plots} plots saved ===")
