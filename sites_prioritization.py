
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.features import DivIcon
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ==============================================================================
# 01. Environment Setup & Data Loading
# ==============================================================================

FILE_PATH = '/MDA Assignment/data/final_integrated_data.csv'
df = pd.read_csv(FILE_PATH, low_memory=False)

# Global feature engineering
df['datum_van'] = pd.to_datetime(df['datum_van'])
df['day_of_week'] = df['datum_van'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# Define proximity indicators
df['is_near_station'] = df['station_count'] > 0
df['is_near_school'] = df['school_count'] > 0
df['is_near_park'] = df['park_count'] > 0


def get_poi_group(row):
    if row['is_near_station']: return 'Station'
    if row['is_near_school']:  return 'School'
    if row['is_near_park']:    return 'Park'
    return 'Other'


df['poi_group'] = df.apply(get_poi_group, axis=1)

# Dynamically identify the precipitation column
precip_cols = [c for c in df.columns if 'precip' in c.lower() or 'prcp' in c.lower()]
if precip_cols:
    df['is_rainy'] = df[precip_cols[0]] > 0
else:
    raise ValueError("Precipitation column not found in dataset.")

# Hour-precipitation interaction feature
df['hour_rain_interaction'] = df['hour'] * df['is_rainy'].astype(int)

# ==============================================================================
# 02. Cycling Traffic Profiling & Clustering
# ==============================================================================

# Pivot to extract 24h profiles and apply row-wise scaling to cluster by shape
hourly_profiles = df.groupby(['site_id', 'hour'])['count'].mean().unstack().fillna(0)
profiles_scaled = StandardScaler().fit_transform(hourly_profiles.T).T

# Apply KMeans Clustering (K=4)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
hourly_profiles['cluster'] = kmeans.fit_predict(profiles_scaled)

CLUSTER_MAP = {
    0: "Standard Commuter (Mixed)",
    1: "Recreational/Park (Elastic)",
    2: "Low Volume/Outliers",
    3: "High-Peak School/Work (Critical)"
}
hourly_profiles['cluster_name'] = hourly_profiles['cluster'].map(CLUSTER_MAP)

# Visualize cluster archetypes
plt.figure(figsize=(12, 6))
for cid, cname in CLUSTER_MAP.items():
    c_data = hourly_profiles[hourly_profiles['cluster'] == cid].drop(['cluster', 'cluster_name'], axis=1)
    plt.plot(c_data.mean(), label=f'{cname} (n={len(c_data)})', linewidth=2.5)

plt.title('Cycling Traffic Behavioral Archetypes (Normalized Profiles)', fontsize=12)
plt.xlabel('Hour of Day')
plt.ylabel('Scaled Intensity')
plt.xticks(range(24))
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Cross-tabulate clusters with POI groups to analyze composition
cluster_summary = hourly_profiles[['cluster', 'cluster_name']].reset_index()
cluster_summary = cluster_summary.merge(df[['site_id', 'poi_group']].drop_duplicates(), on='site_id')
print("\n--- Cluster Composition by POI Group ---")
print(pd.crosstab(cluster_summary['cluster_name'], cluster_summary['poi_group']))

# ==============================================================================
# 03. Weather Resilience Audit & Infrastructure Prioritization
# ==============================================================================

# Calculate global resilience metrics by POI group
resilience_poi = df.groupby(['poi_group', 'is_rainy'])['count'].mean().unstack()
resilience_poi['drop_pct'] = (1 - (resilience_poi[True] / resilience_poi[False])) * 100
resilience_poi.columns = ['Avg_Clear', 'Avg_Rainy', 'Drop_%']
print("\n--- POI Resilience Summary ---")
print(resilience_poi)

# Calculate drop percentages and resilience scores per site
site_resilience = df.groupby(['site_id', 'is_rainy'])['count'].mean().unstack()
site_resilience['drop_pct'] = (1 - (site_resilience[True] / site_resilience[False])) * 100
site_resilience.replace([np.inf, -np.inf], np.nan, inplace=True)
site_resilience.dropna(subset=['drop_pct'], inplace=True)
site_resilience['resilience_score'] = 100 - site_resilience['drop_pct']

# Compile site baseline statistics
site_stats = df.groupby('site_id').agg(
    avg_hourly_traffic=('count', 'mean'),
    poi_group=('poi_group', 'first'),
    lat=('lat_x', 'first'),
    lon=('lon', 'first')
).join(site_resilience[['resilience_score', 'drop_pct']], how='inner')

# Isolate essential demand categories (Schools and Stations)
essential_sites = site_stats[site_stats['poi_group'].isin(['School', 'Station'])].copy()

# Min-Max normalization for traffic volume and vulnerability
t_min, t_max = essential_sites['avg_hourly_traffic'].min(), essential_sites['avg_hourly_traffic'].max()
essential_sites['norm_traffic'] = (essential_sites['avg_hourly_traffic'] - t_min) / (t_max - t_min)

v_min, v_max = (100 - essential_sites['resilience_score']).min(), (100 - essential_sites['resilience_score']).max()
essential_sites['norm_vuln'] = (100 - essential_sites['resilience_score'] - v_min) / (v_max - v_min)

# Compute Investment Priority Score (50% Traffic Volume + 50% Vulnerability)
essential_sites['priority_score'] = (essential_sites['norm_traffic'] * 0.5 + essential_sites['norm_vuln'] * 0.5) * 100
top_investment_sites = essential_sites.sort_values(by='priority_score', ascending=False).head(15)

print("\n--- Top 15 Infrastructure Investment Priority Sites ---")
print(top_investment_sites[['poi_group', 'avg_hourly_traffic', 'resilience_score', 'priority_score']])

# Generate spatial mapping for top 15 priority targets
map_center = [top_investment_sites['lat'].mean(), top_investment_sites['lon'].mean()]
p_map = folium.Map(location=map_center, zoom_start=11, tiles='CartoDB positron')

for idx, row in top_investment_sites.iterrows():
    folium.Marker(
        location=[row['lat'], row['lon']],
        popup=f"ID: {idx} | POI: {row['poi_group']}<br>Score: {row['priority_score']:.2f}",
        icon=folium.Icon(color='orange', icon='info-sign')
    ).add_to(p_map)

    folium.Marker(
        location=[row['lat'], row['lon']],
        icon=DivIcon(
            icon_size=(150, 36),
            icon_anchor=(0, 0),
            html=f'<div style="font-size: 10pt; color: #d35400; font-weight: bold;">ID: {idx}</div>',
        )
    ).add_to(p_map)

# ==============================================================================
# 04. Behavioral Phase Analysis: Anticipation vs. Physical Impact
# ==============================================================================

df_sorted = df.sort_values(by=['site_id', 'month', 'day', 'hour']).copy()

# Shift precipitation column to define the 1-hour "Forecast Window"
df_sorted['is_forecast_rain'] = df_sorted.groupby('site_id')[precip_cols[0]].shift(-1) > 0


def assign_behavioral_phase(row):
    if row['is_rainy']: return 'Physical Reaction'
    if row['is_forecast_rain']: return 'Psychological Anticipation'
    return 'Clear Baseline'


df_sorted['behavioral_phase'] = df_sorted.apply(assign_behavioral_phase, axis=1)

# Measure volume drops across different psychological and physical phases
phase_stats = df_sorted.groupby(['poi_group', 'behavioral_phase'])['count'].mean().unstack()
for col in ['Psychological Anticipation', 'Physical Reaction']:
    phase_stats[f'{col} % Drop'] = ((phase_stats[col] - phase_stats['Clear Baseline']) / phase_stats[
        'Clear Baseline']) * 100

print("\n--- Behavioral Phase Impact (% Drop from Baseline) ---")
print(phase_stats.filter(like='% Drop'))

# ==============================================================================
# 05. Demand Modeling (Predictive Analytics)
# ==============================================================================

features = [
    'hour', 'month', 'day_of_week', 'is_weekend', 'is_rainy',
    'poi_group', 'dist_nearest_station', 'dist_nearest_school',
    'school_count', 'station_count', 'park_count', 'hour_rain_interaction'
]
X = df[features].copy()
y = df['count']

num_features = ['hour', 'month', 'dist_nearest_station', 'dist_nearest_school', 'school_count', 'station_count',
                'park_count', 'hour_rain_interaction']
cat_features = ['poi_group', 'is_rainy', 'is_weekend', 'day_of_week']

# Define preprocessor transformations
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_features),
        ('cat', Pipeline(
            [('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]),
         cat_features)
    ])

# Build pipeline using Random Forest Regressor
rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=50, max_depth=15, n_jobs=-1, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('\nTraining Random Forest model (processing 2.4M records)...')
rf_pipeline.fit(X_train, y_train)

# Model performance evaluation
y_pred = rf_pipeline.predict(X_test)
print(f"--- Model Performance Evaluation ---")
print(f"R-squared Score: {r2_score(y_test, y_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")

# ==============================================================================
# 06. Model Diagnostics & Error Analysis (Scenario Application)
# ==============================================================================

# Extract a specific day (May 15th) for model diagnostic deployment
may15_data = df[(df['month'] == 5) & (df['day'] == 15)].copy()
may15_data['predicted_count'] = rf_pipeline.predict(may15_data[features])
may15_data['signed_error'] = may15_data['count'] - may15_data['predicted_count']

# Filter for under-predictions (Actual > Predicted) to identify capacity bottlenecks
under_pred_df = may15_data[may15_data['signed_error'] > 0]
hourly_error_poi = under_pred_df.groupby(['poi_group', 'hour'])['signed_error'].mean().reset_index()

# Visualize prediction residual paths by POI group
plt.figure(figsize=(12, 6))
sns.lineplot(data=hourly_error_poi, x='hour', y='signed_error', hue='poi_group', marker='o')
plt.axvline(8, color='gray', linestyle='--', alpha=0.7, label='Morning Rush Hour')
plt.title('Average Under-Prediction Magnitude by POI Category & Hour (May 15th)')
plt.xlabel('Hour of Day')
plt.ylabel('Signed Error (Actual - Predicted)')
plt.xticks(range(24))
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()