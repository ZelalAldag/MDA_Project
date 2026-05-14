import pandas as pd
import pyarrow.parquet as pq

# --- 1. Configuration ---
PATH_SITES = '/Users/zhaohan/Desktop/Main MacBook Air/Modern Data Analytics/Assignment/sites&directions.csv'
PATH_WEATHER = '/Users/zhaohan/Desktop/Main MacBook Air/Modern Data Analytics/Assignment/weather_data_2024_filtered.csv'
PATH_FLOW = '/Users/zhaohan/Desktop/Main MacBook Air/Modern Data Analytics/Assignment/fietstellingen_clean.parquet'
PATH_OUTPUT = '/Users/zhaohan/Desktop/Main MacBook Air/Modern Data Analytics/Assignment/final_integrated_data.csv'

# --- 2. Data Loading & Preprocessing ---

# 2.1 Load site metadata and standardize direction labels
df_sites = pd.read_csv(PATH_SITES)
df_sites['direction'] = df_sites['direction'].str.lower()

# 2.2 Load weather data and extract time components for joining
df_weather = pd.read_csv(PATH_WEATHER)
df_weather['time'] = pd.to_datetime(df_weather['time'])
df_weather['year'] = df_weather['time'].dt.year
df_weather['month'] = df_weather['time'].dt.month
df_weather['day'] = df_weather['time'].dt.day
df_weather['hour'] = df_weather['time'].dt.hour

# 2.3 Load 2024 traffic flow data using Parquet filters
table = pq.read_table(PATH_FLOW, filters=[('year', '==', 2024)])
df_flow_raw = table.to_pandas()

# --- 3. Hourly Aggregation ---

# Filter traffic data to include only valid site IDs
valid_ids = df_sites['site_ID'].unique()
df_flow_filtered = df_flow_raw[df_flow_raw['site_id'].isin(valid_ids)].copy()

# Aggregate counts by hour, site, and direction
group_cols = ['site_id', 'direction', 'year', 'month', 'day', 'hour']
df_hourly = df_flow_filtered.groupby(group_cols + ['lat', 'lon'])['count'].agg(['sum', 'count']).reset_index()
df_hourly.rename(columns={'sum': 'total_count', 'count': 'obs_per_hour'}, inplace=True)

# Keep only complete hours (e.g., exactly 4 intervals of 15-min data)
df_hourly = df_hourly[df_hourly['obs_per_hour'] == 4].copy()
df_hourly['direction'] = df_hourly['direction'].str.lower()

# --- 4. Multi-Source Data Merging ---

# Step A: Join traffic flow with weather data on time and location
df_merged = pd.merge(
    df_hourly,
    df_weather,
    on=['site_id', 'year', 'month', 'day', 'hour'],
    how='left'
)

# Step B: Join the result with site metadata on site ID and direction
df_final = pd.merge(
    df_merged,
    df_sites,
    left_on=['site_id', 'direction'],
    right_on=['site_ID', 'direction'],
    how='left',
    suffixes=('_flow', '_meta')
)

# --- 5. Final Cleanup & Export ---

# Remove redundant columns generated during merge
if 'site_ID' in df_final.columns:
    df_final.drop(columns=['site_ID'], inplace=True)

# Export the integrated dataset to CSV
df_final.to_csv(PATH_OUTPUT, index=False)

print(f"Integration complete. Final dataset shape: {df_final.shape}")
print(f"File saved to: {PATH_OUTPUT}")