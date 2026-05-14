"""
Exploratory Data Analysis and Baseline Modeling
Cyclist Traffic, Spatial, and Weather Data (2024)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import statsmodels.api as sm

# --- 0. Setup and Data Loading ---
file_path = '/MDA Assignment/data/final_integrated_data.csv'

# Set dtypes to avoid mixed-type warnings for specific columns
df = pd.read_csv(file_path, dtype={'wegnr': str, 'district': str})

# Handle missing spatial values: assume NA means outside search radius (e.g., 3km)
df['dist_nearest_station'] = df['dist_nearest_station'].fillna(0)
df['dist_nearest_school'] = df['dist_nearest_school'].fillna(0)


# --- 1. Feature Engineering (Temporal) ---
# Synchronize observations with 2024 calendar dates
df['date_dt'] = pd.to_datetime(df[['year', 'month', 'day']])
df['day_name'] = df['date_dt'].dt.day_name()
df['day_of_week'] = df['date_dt'].dt.dayofweek
df['is_weekend'] = df['day_name'].apply(lambda x: 'Weekend' if x in ['Saturday', 'Sunday'] else 'Weekday')


# --- 2. Traffic Pattern Exploratory ---

# 2.1 Hourly Trends: Weekdays vs. Weekends
plt.figure(figsize=(12, 6))
hourly_avg = df.groupby(['hour', 'is_weekend'])['count'].mean().reset_index()
sns.lineplot(data=hourly_avg, x='hour', y='count', hue='is_weekend', marker='o')
plt.title('Average Cyclist Count by Hour (2024)')
plt.grid(True, alpha=0.3)
plt.show()

# 2.2 Directional Analysis (Inbound vs. Outbound)
plt.figure(figsize=(12, 6))
dir_trend = df.groupby(['hour', 'direction'])['count'].mean().reset_index()
sns.lineplot(data=dir_trend, x='hour', y='count', hue='direction', marker='o')
plt.title('Directional Traffic Flow by Hour')
plt.show()


# --- 3. Spatial and Geographical Analysis ---

# 3.1 Ranking by Municipality (Top 20)
top_gemeentes = df.groupby('gemeente')['count'].mean().sort_values(ascending=False).head(20).reset_index()
plt.figure(figsize=(10, 8))
sns.barplot(data=top_gemeentes, y='gemeente', x='count', palette='magma')
plt.title('Top 20 Municipalities by Traffic Volume')
plt.show()

# 3.2 Proximity Correlation (Parks, Schools, Stations)
spatial_cols = ['count', 'park_count', 'school_count', 'station_count', 'dist_nearest_station', 'dist_nearest_school']
plt.figure(figsize=(10, 8))
sns.heatmap(df[spatial_cols].corr(), annot=True, cmap='RdBu_r', center=0, fmt='.2f')
plt.title('Correlation: Traffic vs. Infrastructure')
plt.show()


# --- 4. Environmental and Weather Impact ---

# 4.1 Temperature and Solar Radiation Sensitivity
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Binned Temperature Analysis
df['temp_bin'] = pd.cut(df['temperature_2m'], bins=range(-10, 45, 5))
temp_stats = df.groupby('temp_bin', observed=False)['count'].mean().reset_index()
sns.barplot(data=temp_stats, x='temp_bin', y='count', ax=ax1, palette='coolwarm')
ax1.set_title('Impact of Temperature Intervals')
ax1.tick_params(axis='x', rotation=45)

# Solar Radiation Trend
sns.regplot(data=df.sample(20000), x='shortwave_radiation', y='count',
            scatter_kws={'alpha':0.1}, line_kws={'color':'red'}, ax=ax2)
ax2.set_title('Impact of Solar Radiation')
plt.show()

# 4.2 Impact of Precipitation (Rain/Snow)
# Compare average counts: Snow vs. Clear
df['has_snow'] = df['snowfall'] > 0
print("Traffic Reduction during Snowfall:")
print(df.groupby('has_snow')['count'].mean())


# --- 5. Baseline Predictive Model (OLS) ---

# Define feature set including Temporal, Weather, and Spatial variables
features = [
    'hour', 'temperature_2m', 'shortwave_radiation',
    'precipitation', 'wind_speed_10m', 'relative_humidity_2m',
    'park_count', 'school_count', 'station_count'
]

X = df[features].copy()
X['is_weekend'] = (df['is_weekend'] == 'Weekend').astype(int)
X = sm.add_constant(X)  # Add intercept
y = df['count']

# Fit Ordinary Least Squares model
model = sm.OLS(y, X).fit()
print(model.summary())

# Extract relative feature importance based on t-values
importance = pd.DataFrame({
    'Predictor': model.params.index,
    't-value': model.tvalues.values
}).sort_values('t-value', ascending=False)

print("\nFeature Importance Ranking:")
print(importance)