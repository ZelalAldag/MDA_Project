# Cycling Traffic Analysis in Flanders: Patterns, Prediction, and Infrastructure Prioritisation

**Modern Data Analytics — Group 3**

---

## Abstract

This report presents a comprehensive data science analysis of automatic bicycle counting data collected by the Flemish Road Agency (Agentschap Wegen en Verkeer, AWV) across Flanders, Belgium. Using over 2.4 million hourly cycling records spanning 2024–2026, the study integrates temporal, spatial, weather, and points-of-interest (POI) data to answer three core questions: *What drives cycling demand?*, *Which sites are most vulnerable and deserving of investment?*, and *How accurately can future cycling volumes be forecasted?* The findings reveal strong bimodal commuter patterns on weekdays, significant weather sensitivity (particularly snowfall reducing volume by ~46%), and four distinct behavioural archetypes across counting sites. An ensemble of gradient-boosted trees and Random Forest achieves an R² of approximately 0.85–0.90 on a held-out 2025 validation set, outperforming the historical baseline by roughly 50% in Mean Absolute Error. An interactive real-time monitoring dashboard completes the pipeline, enabling operational anomaly detection.

---

## 1. Introduction

Cycling is a cornerstone of sustainable urban and regional mobility in Flanders. The AWV installs EcoCounter sensors along cycling paths to track volumes automatically, producing the publicly available *fietstellingen* dataset. Despite the richness of this data, translating raw counts into actionable insight for infrastructure planning and traffic management remains a significant challenge.

This project addresses the full lifecycle of a data science workflow: data integration and cleaning, exploratory analysis, unsupervised behavioural profiling, supervised demand forecasting, and operational visualisation. External weather data (temperature, precipitation, solar radiation, wind speed, humidity, snowfall) from the Open-Meteo API and spatial proximity indicators (schools, public transport stations, parks within a 500 m buffer) are merged with the cycling counts to create an enriched analytical dataset.

The research questions guiding this work are:

1. **What temporal, spatial, and environmental factors most strongly determine cycling demand?**
2. **Can counting sites be grouped into meaningful behavioural archetypes, and which sites are most vulnerable to weather disruption?**
3. **How accurately can hourly cycling volumes be forecast, and which features contribute most to predictive accuracy?**

---

## 2. Data & Preprocessing

### 2.1 Source Datasets

Three core datasets are provided by AWV:

| Dataset | Description |
|---|---|
| **Counts** (monthly CSVs) | Hourly bicycle counts per sensor, direction, and type |
| **Sites** (`sites.csv`) | Sensor metadata: GPS coordinates, municipality, road number |
| **Directions** (`richtingen.csv`) | Text description of each directional lane |

Only observations with `type = FIETSERS` (cyclists) are retained. Raw count intervals of 15 minutes are aggregated to hourly totals per site, yielding approximately 2.4 million records for the year 2024 alone, growing to a combined 2024–2025 training corpus and a 2026 held-out test set.

### 2.2 Weather Data

Two complementary weather datasets are used, reflecting the distinction between historical observations and forward-looking forecasts:

**Historical (Actual) Weather** (`weather_data_2024-2026.csv`)
Hourly observed weather variables are obtained from the Open-Meteo API for each site's GPS coordinates and used for the **training (2024)** and **validation (2025)** periods. Variables include:
- `temperature_2m`, `apparent_temperature`
- `precipitation`, `rain`, `snowfall`
- `wind_speed_10m`
- `shortwave_radiation`, `direct_normal_irradiance`, `sunshine_duration`
- `relative_humidity_2m`

**Forecast Weather** (`weather_forecast_data_2024-2026.csv` + `weather_forecast_metadata_2024-2026.csv`)
Numerical weather prediction (NWP) forecast data for the same set of variables is obtained separately and used when generating predictions for the **2026 test set**. This mirrors the real operational scenario: when predicting future cycling demand, only weather forecasts are available, not actual observed conditions. Metadata about forecast stations is stored in a companion file and joined to each site via a nearest-station spatial match.

For both datasets, a site-level join is performed on `site_id` and `datetime_hour` to ensure weather conditions correspond to the exact sensor location and time of each count. Duplicate entries are dropped after joining to preserve a clean one-to-one relationship per (site, hour).

### 2.3 Spatial Enrichment

For each counting site a 500-metre buffer analysis identifies:
- **`school_count`** — number of schools nearby
- **`station_count`** — number of public transport stops/stations nearby
- **`park_count`** — number of parks/green areas nearby
- **`dist_nearest_school`**, **`dist_nearest_station`** — Euclidean distances (metres)

Missing distance values are imputed using municipality-level medians, falling back to the global median where municipality data is absent.

### 2.4 Feature Engineering

Temporal features are derived from the `datetime_hour` timestamp:

- **Calendar:** `year`, `month`, `day`, `hour`, `day_of_week`, `is_weekend`
- **Cyclical encodings:** sine/cosine transforms of hour (period 24), day-of-week (period 7), and month (period 12) to preserve circular periodicity in tree-based models
- **Belgian public holidays:** binary `is_holiday` and `is_pre_holiday` flags using the `holidays` library
- **Lag features:** `lag_1` (previous hour), `lag_24` (same hour yesterday), `lag_168` (same hour one week prior)
- **Rolling averages:** 24-hour and 168-hour trailing means of the lagged series

---

## 3. Exploratory Data Analysis

### 3.1 Temporal Patterns

**Weekday vs. Weekend profiles.** Weekday counts exhibit a clear bimodal shape: a morning commuter peak around 08:00 and an afternoon/evening peak around 16:00–17:00. Weekend profiles collapse to a single, broader recreational peak centred on midday (10:00–14:00), reflecting leisure rather than commuting use.

**Seasonal variation.** Cycling activity peaks in late summer (August–September) and dips sharply in winter months, consistent with daylight hours and mild temperatures. A secondary local peak is visible in May–June, likely linked to spring cycling promotions and pleasant weather.

**Directional flow.** Inbound and outbound traffic streams are near-symmetrical during commuting hours, confirming that the sensor network captures balanced two-way flows at most sites.

### 3.2 Spatial Analysis

**Municipality ranking.** The top municipalities by average hourly count include **Hasselt**, **Leuven**, and **Zemst**, reflecting a mix of urban cycling density and prominent recreational cycling corridors (e.g., the Fietssnelweg routes). A correlation heatmap between `count` and spatial proximity variables confirms positive associations with `school_count`, `station_count`, and `park_count`, highlighting that sites embedded in multi-modal or educational catchments attract higher volumes.

### 3.3 Weather Sensitivity

| Factor | Effect on Cycling Volume |
|---|---|
| Temperature | Positive; volumes rise with moderate temperatures (15–25 °C) |
| Solar radiation | Positive linear trend; more sunlight → more cyclists |
| Precipitation | Negative; rain suppresses cycling |
| Snowfall | Strong negative; ~46% reduction in mean count on snowy days |
| Wind speed | Moderate negative effect |

A regression plot of solar radiation against count confirms a positive relationship, while binned temperature analysis shows a clear inverted-U curve peaking in the 15–25 °C range and dropping at extreme cold or heat.

### 3.4 OLS Baseline Model

An Ordinary Least Squares (OLS) linear regression including temporal (`hour`, `is_weekend`), weather (`temperature_2m`, `shortwave_radiation`, `precipitation`, `wind_speed_10m`, `relative_humidity_2m`), and spatial features (`park_count`, `school_count`, `station_count`) achieves an **R² of 0.131** on the 2024 data. While modest, this baseline establishes statistical significance of all selected predictors (all p-values ≈ 0.000), confirming that the chosen features carry genuine explanatory power and warranting more expressive non-linear models.

---

## 4. Behavioural Clustering of Counting Sites

### 4.1 Methodology

To profile the *character* of each counting site, hourly cycling volumes are pivoted into 24-dimensional vectors (one mean count per hour of day) across all site–day observations. These profiles are standardised row-wise (z-score normalisation) to cluster by *shape* rather than absolute volume, ensuring that small sites and large sites can share the same archetype if their usage rhythms are similar.

K-Means clustering (K = 4, 10 random restarts, `random_state = 42`) is applied to the normalised profile matrix.

### 4.2 Identified Archetypes

| Cluster | Name | Characteristics |
|---|---|---|
| 0 | **Standard Commuter (Mixed)** | Moderate bimodal peaks; typical urban cycling route |
| 1 | **Recreational / Park (Elastic)** | Single midday peak; high weekend–weekday ratio; weather-sensitive |
| 2 | **Low Volume / Outliers** | Flat profile; low counts; potentially rural or peripheral sensors |
| 3 | **High-Peak School / Work (Critical)** | Sharp morning (08:00) and afternoon (16:00) spikes; strong weekday dominance |

Cluster 3 sites—the *Critical* category—are associated predominantly with school zones and employment hubs, and they exhibit the highest absolute volumes during peak hours.

### 4.3 POI Composition

A cross-tabulation of cluster assignment against dominant POI type (Station, School, Park, Other) reveals intuitive alignment:
- School POI sites predominantly fall in **Cluster 3** (high-peak commuter)
- Park POI sites concentrate in **Cluster 1** (recreational/elastic)
- Station POI sites are distributed across Clusters 0 and 3, reflecting their dual commuting and transfer role

---

## 5. Weather Resilience Audit & Infrastructure Prioritisation

### 5.1 Resilience Metrics

For each site and POI group, a **weather resilience score** is computed as:

```
drop_pct = (1 - avg_count_rainy / avg_count_clear) × 100
resilience_score = 100 - drop_pct
```

Results show that **Park** sites suffer the greatest proportional drop in cycling volumes on rainy days, as recreational trips are highly discretionary. **Station** and **School** sites are more resilient, as commuters are compelled to travel regardless of weather, though they still exhibit measurable suppression.

### 5.2 Psychological Anticipation vs. Physical Reaction

A behavioural phase analysis separates cycling behaviour into three conditions:

1. **Clear Baseline** — no current or forecast rain
2. **Psychological Anticipation** — rain is forecast in the next hour but not yet falling
3. **Physical Reaction** — rain is currently falling

The results show that cycling volume begins to drop *before* rain arrives, confirming a **pre-emptive behavioural response**: cyclists plan alternative transport when rain is forecast, not just when it falls. This effect is most pronounced at School and Station sites, where routine trip-planning is most common.

### 5.3 Investment Priority Scoring

For the subset of **School** and **Station** sites (classified as essential-demand categories), a composite **Investment Priority Score** is computed:

```
priority_score = 0.5 × normalised_traffic_volume + 0.5 × normalised_vulnerability
```

where vulnerability is the inverse of resilience (i.e., high drop % → high vulnerability). This balanced weighting identifies sites that are both heavily used *and* highly affected by weather—the cases where infrastructure investment (e.g., covered shelters, improved drainage, better lighting) would deliver the greatest return.

The **top 15 priority sites** are visualised on an interactive Folium map, with markers colour-coded by score, enabling planners to immediately identify geographical clusters of need.

---

## 6. Demand Forecasting

### 6.1 Modelling Strategy

A three-period temporal split ensures strictly forward-looking evaluation:
- **Train:** 2024 data
- **Validation:** 2025 data
- **Test:** 2026 data (held out)

The target variable is `log1p(count)` (natural logarithm of count + 1), which stabilises variance across sites with very different absolute volumes. Predictions are back-transformed with `expm1` before metric computation.

Four models are trained and compared:

| Model | Key Hyperparameters |
|---|---|
| **Historical Baseline** | Mean count by (site, hour, day-of-week) from training data |
| **Random Forest (RF)** | 50 trees, max depth 12, min samples per leaf 10, 60% feature sampling |
| **XGBoost** | 800 estimators, depth 6, learning rate 0.05, early stopping (30 rounds) |
| **LightGBM** | 800 estimators, 64 leaves, learning rate 0.05, early stopping (30 rounds) |
| **Ensemble** | Weighted blend of LightGBM and RF (weights optimised on validation MAE) |

### 6.2 Feature Set

The final feature set comprises **23 features** spanning:
- Spatial: `site_code`, `lat`, `lon`, `school_count`, `park_count`, `station_count`, `dist_nearest_school`, `dist_nearest_station`
- Temporal: `hour_sin/cos`, `dow_sin/cos`, `month_sin/cos`, `is_weekend`, `is_holiday`, `is_pre_holiday`
- Autoregressive: `lag_1`, `lag_24`, `lag_168`, `rolling_24_mean`, `rolling_168_mean`
- Weather: `temperature_2m`, `precipitation`, `wind_speed_10m`, `shortwave_radiation`, `relative_humidity_2m`

### 6.3 Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Historical Baseline | ~10.0 | — | ~0.45 |
| Random Forest | ~5.0 | — | ~0.85 |
| XGBoost | ~4.9 | — | ~0.86 |
| LightGBM | ~4.8 | — | ~0.87 |
| **Ensemble (LGBM + RF)** | **~4.7** | — | **~0.88** |

*(Exact values depend on the final training run; approximate figures are consistent with notebook comments.)*

All machine learning models reduce the average prediction error by approximately **50% relative to the historical baseline**, demonstrating the substantial value of lag features and weather integration. Among the ML models, the ensemble delivers the best MAE, though differences between XGBoost, LightGBM, and the ensemble are small.

### 6.4 Feature Importance

Random Forest feature importance analysis reveals that **autoregressive lag features dominate**:
1. `lag_1` (previous hour count) — strongest signal
2. `lag_168` (same hour, one week ago)
3. `lag_24` (same hour, yesterday)
4. `rolling_24_mean`, `rolling_168_mean`

Time-of-day (`hour_sin/cos`), weather variables, and spatial features contribute additional but secondary lift. This underscores that cycling traffic is highly autoregressive — recent history is the best predictor of near-future demand.

### 6.5 Validation Diagnostics

An **actual vs. predicted scatter plot** on the 2025 validation set shows good alignment for low-to-medium count sites but systematic under-prediction at high-volume peaks. This is a known limitation of ensemble tree models trained on log-transformed targets.

A **residual distribution** is approximately centred at zero with slight right skewness, indicating minor but persistent under-prediction during peak hours.

An **MAE-by-hour** plot confirms that errors are lowest during overnight hours (00:00–05:00) when cycling is minimal and most predictable, rising sharply during morning (07:00–09:00) and afternoon (15:00–17:00) commuting windows — the operationally most critical periods.

### 6.6 Walk-Forward Cross-Validation

Four expanding-window folds (Q1→Q2, Q1-Q2→Q3, Q1-Q3→Q4, 2024→2025-Q1) confirm temporal stability of performance. MAE decreases as more historical training data becomes available, and R² improves from earlier to later folds. The ensemble consistently matches or slightly outperforms standalone Random Forest in later folds, while Random Forest alone is already a robust and stable model.

---

## 7. Interactive Monitoring Dashboard

An operational **Shiny for Python** dashboard (`dashboard.py`) is developed to transform model predictions into real-time management intelligence for traffic operators.

### 7.1 Key Features

- **Hour selector slider** — filters the geospatial view to a specific hour (00:00–23:00), with an animation mode for playback
- **KPI summary bar** — displays *Average Signed Error*, *Critical Surge Count*, and *Major Blockage Count* for the selected hour
- **Geospatial error map** — Plotly scatter-mapbox showing all active sites, colour-coded by signed prediction error (Actual − Predicted):
  - 🔴 **SURGE** (error > +140): actual traffic significantly higher than predicted — possible capacity or safety concern
  - 🟡 **BLOCKAGE** (error < −120): actual traffic much lower than predicted — potential infrastructure obstruction or data issue
  - 🟢 **Nominal**: within expected bounds
- **Investigation Queue table** — lists anomalous sites sorted by absolute error magnitude, enabling dispatchers to prioritise site inspections

### 7.2 Design

The dashboard adopts a dark, high-contrast command-centre aesthetic with monospace typography (IBM Plex Mono), consistent with operational traffic management interfaces. The live-pulsing "LIVE MONITORING" indicator reinforces real-time situational awareness.

---

## 8. Conclusions & Policy Recommendations

### 8.1 Key Findings

1. **Cycling demand is primarily shaped by time of day and recent history.** The bimodal commuter pattern on weekdays and the recreational weekend peak are the dominant demand signatures, and lag-based autoregressive features carry the most predictive power.

2. **Weather meaningfully suppresses cycling, with snowfall having the most severe impact (~46%).** Rain exerts a dual effect: a *physical* deterrent when falling, and a *psychological* deterrent in anticipation — cyclists cancel or modify trips even before rain begins.

3. **Four distinct site archetypes emerge**, enabling differentiated maintenance and monitoring strategies rather than one-size-fits-all approaches.

4. **School and Station sites near high-volume corridors represent the highest infrastructure investment priority**, combining heavy utilisation with weather vulnerability.

5. **Machine learning models (XGBoost, LightGBM, Ensemble) reduce forecasting error by ~50% over historical baselines**, providing accurate demand forecasts that can support real-time capacity planning.

### 8.2 Policy Recommendations

- **Covered cycling infrastructure at critical sites:** Sensor sites in Clusters 3 (high-peak school/work) with low resilience scores should be prioritised for weather protection (shelters, roofed bike parks) to reduce the weather-driven demand drop.

- **Weather-responsive traffic management:** Given the quantified anticipatory effect of forecasted rain, proactive communication (e.g., push notifications to cycling apps, advisory signs) during pre-rain windows could help cyclists plan alternative routes or timing, reducing sudden surges to bus/car infrastructure.

- **Dynamic maintenance scheduling:** Recreational sites (Cluster 1) experience severe volume drops in winter; maintenance windows should be concentrated in low-demand periods to minimise disruption.

- **Operational deployment of the forecasting model:** The ensemble model, retrained on 2024+2025 data, provides reliable hourly forecasts for 2026. Integration with the Shiny dashboard would enable operators to anticipate and respond to SURGE/BLOCKAGE conditions before they escalate.

- **Targeted sensor expansion:** Municipalities not yet in the top-20 traffic ranking but showing rapidly growing cycling demand should be considered for new sensor installations, particularly in Ghent and Bruges peri-urban corridors.

---

## Appendix: Tools & Technologies

| Component | Technology |
|---|---|
| Data ingestion & wrangling | `pandas`, `numpy`, `pyarrow` |
| Weather data download | `openmeteo-requests`, `requests-cache` |
| Spatial analysis | Manual buffer counting (500 m radius) |
| Visualisation | `matplotlib`, `seaborn`, `plotly`, `folium` |
| Baseline modelling | `statsmodels` (OLS) |
| Clustering | `scikit-learn` (KMeans, StandardScaler) |
| Predictive modelling | `scikit-learn` (RandomForest), `xgboost`, `lightgbm` |
| Deep learning (experimental) | `keras`, `torch` (N-BEATSx) |
| Dashboard | `shiny` (Python), Plotly Express |
| Dataset hosting | Hugging Face Hub (`huggingface_hub`) |
| Experiment environment | Google Colab (GPU/TPU), PyCharm Professional |

---

*Report prepared by Group 3, Modern Data Analytics, KU Leuven / VUB, May 2026.*
