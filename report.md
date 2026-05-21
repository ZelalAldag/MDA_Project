# Spatio-Temporal Analysis of Bicycle Traffic in Flanders

**Modern Data Analytics — Group 3**  
KU Leuven | 2024–2025  
*Data Source: Agentschap Wegen en Verkeer (AWV) — Fietstellingen Open Data*

---

## Abstract

This report presents a comprehensive, end-to-end data science pipeline for analysing automatic bicycle count data collected by the Flemish Road Agency (AWV) across Flanders, Belgium. Integrating over 2.4 million hourly cycling records (2023–2026) with observed and forecast weather data, OpenStreetMap spatial proximity features, and Belgian transport statistics, the project addresses five interconnected research objectives: temporal/spatial pattern discovery, behavioural site clustering, weather-resilience scoring, machine-learning-based demand forecasting, and sustainability impact quantification. Key findings include strong bimodal commuter peaks on weekdays, snowfall reducing cycling volume by ~46%, two robust site archetypes (infrastructure-supported vs. peripheral), and an XGBoost model with weather forecast discrepancy features achieving a validation MAE of approximately 9.1 cyclists/day — a ~50% improvement over the historical baseline. An interactive Shiny monitoring dashboard and a CO₂ savings assessment complete the pipeline, providing directly actionable outputs for AWV planners and municipal governments.

---

## 1. Introduction & Research Objectives

### 1.1 Motivation

Cycling plays a central role in sustainable mobility in Flanders. The AWV installs EcoCounter sensors along cycling paths to track volumes automatically, producing the publicly available *fietstellingen* dataset. Understanding cycling patterns is valuable for multiple stakeholders:

- **AWV planners**, who need evidence-based prioritisation of maintenance, expansion, and safety investments.
- **Municipal governments**, who want to know which corridors carry the heaviest all-weather commuter loads and which are primarily recreational.
- **Sustainability policymakers**, who need quantified estimates of CO₂ savings and modal-shift potential to justify cycling infrastructure spending.

### 1.2 Research Objectives

Following teaching assistant feedback — which highlighted the risk of over-engineering with deep learning on a limited dataset, the need for stronger spatial signals, the importance of accounting for weather forecast uncertainty, and the requirement to make results actionable — our objectives were defined as follows:

1. Analyse temporal and spatial cycling patterns across the AWV monitoring network (2023–2025).
2. Categorise counting sites into behaviour-based archetypes using K-means clustering enriched with spatial infrastructure features.
3. Quantify weather sensitivity at site level and produce a resilience scoring framework that helps governments prioritise maintenance and investment.
4. Develop interpretable forecasting models (XGBoost / Random Forest) using weather covariates, and explicitly account for the discrepancy between forecast and observed weather values.
5. Assess the sustainability impact of the cycling network in terms of CO₂ savings, fuel substitution, and modal-shift potential, broken down by municipality.

> **TA Feedback Addressed:** (1) Weather forecasts are not perfectly observable at prediction time — we respond by engineering forecast-vs-actual discrepancy features (Δweather) as additional model inputs; (2) the feature set lacked spatial information — we address this with park, school, and station proximity counts within a 500 m buffer; (3) deep learning is potentially overkill for this dataset — we lead with XGBoost/Random Forest and include N-BEATSx as a secondary experiment only.

---

## 2. Data & Preprocessing

### 2.1 Source Datasets

The project integrates five complementary data sources:

| Source | Description | Key Variables |
|---|---|---|
| **AWV Fietstellingen** | Monthly bicycle count CSVs (EcoCounter sensors), 2019–2026 | `site_id`, `richting`, `type`, `van`, `tot`, `aantal` |
| **AWV Sites & Directions** | Sensor metadata: GPS, municipality, district, road number | `lat`, `long`, `gemeente`, `domein`, `wegnr`, `interval` |
| **Observed Weather** (`weather_data_2024-2026.csv`) | Hourly actual weather from Open-Meteo API, joined per site | `temperature_2m`, `precipitation`, `wind_speed_10m`, `shortwave_radiation`, `sunshine_duration`, `relative_humidity_2m` |
| **Forecast Weather** (`weather_forecast_data_2024-2026.csv`) | Historical short-range NWP forecast archives, same variables | Same 6 weather variables as observed data |
| **OSM Spatial Data** | Counts of parks, schools, and stations within a 500 m buffer | `park_count`, `school_count`, `station_count`, `dist_nearest_station`, `dist_nearest_school` |

Only observations with `type = FIETSERS` (cyclists) are retained. The final clean dataset covers **143 active sites** for 2024 and **151 sites** for 2025.

### 2.2 Temporal Coverage and Data Split

The analysis window is set to **2023–2026** to avoid cold-start noise from sensors commissioned mid-2022. Approximately 67.5% of stations were commissioned in 2022, with ~15 key sites not active until October 2022.

| Period | Role |
|---|---|
| 2023–2024 | Training (two full seasonal cycles) |
| 2025 | Validation (hyperparameter tuning and model selection) |
| 2026 | Held-out test set (strict out-of-sample evaluation) |

### 2.3 Weather Data: Two Datasets and Quality Check

A dedicated data readiness analysis (`Weather_Error_Analysis.ipynb`) verified both weather datasets before integration.

**Data quality summary:**
- Both observed and forecast datasets share **identical structure**: 12 columns and exactly **2,975,304 rows**.
- Temporal coverage spans 2024–2026 with all months present and correct day counts (including the February 2024 leap day).
- All **151 unique sites** contain exactly **19,704 records** each — perfectly balanced.
- Every day in the dataset contains exactly **24 hourly records** per site — complete temporal integrity confirmed.

**Why two weather datasets?**

| Dataset | Role in Project |
|---|---|
| **Historical (Actual) Weather** | Used for training (2024) and validation (2025) — ground truth conditions paired with observed cycling counts |
| **Forecast Weather** | Used for the 2026 test set and as input to the operational dashboard — mirrors the real-world scenario where only NWP forecasts are available at prediction time |

**Forecast vs. Actual Discrepancy Analysis.** Systematic comparison of the two datasets for 2024 reveals:

- **Temperature:** Forecast error peaks in November–December (>25% relative error), as near-zero actual values amplify percentage deviations. Apparent temperature error exceeds 180% in winter.
- **Solar radiation / sunshine duration:** High relative error in the second half of the year (often >100–200%), reflecting the difficulty of forecasting the exact intensity and duration of solar events.
- **Precipitation:** The forecast shows *precision = 1.0* (no false alarms) but *recall ≈ 58% for rain, 64% for snowfall*, meaning it misses ~40% of actual rain events. F1 scores of 0.74–0.78 indicate a conservative, cautious forecast.
- **Zero error in January–May:** Both datasets are aligned for the first five months, suggesting the data was sourced from the same NWP runs for those periods.

These discrepancies motivate the engineering of **Δweather features** (Section 2.5).

### 2.4 Data Integration Pipeline

The integration script (`merge24.py`) performs a multi-step merge:

1. Load site metadata and standardise direction labels.
2. Load weather data and extract temporal join keys (`year`, `month`, `day`, `hour`).
3. Load traffic flow from Parquet with a year filter, then aggregate to **hourly totals** per (site, direction).
4. Filter to complete hours only (exactly 4 × 15-minute intervals present).
5. Join traffic flow ← weather on (site, time); then join ← site metadata on (site, direction).

The result (`final_integrated_data.csv`) contains all temporal, weather, and spatial features in a single flat table ready for analysis.

### 2.5 Feature Engineering

**Temporal features:**
- Calendar: `year`, `month`, `day`, `hour`, `day_of_week`, `is_weekend`, `is_school_holiday`
- Cyclical encodings: sine/cosine transforms of hour (period 24), day-of-week (period 7), month (period 12) to preserve periodicity in tree models
- Belgian public holidays: binary `is_holiday` and `is_pre_holiday` flags (via `holidays` library)

**Autoregressive lag features (hourly model):**
- `lag_1` — count one hour ago
- `lag_24` — count at the same hour yesterday
- `lag_168` — count at the same hour one week prior
- `rolling_24_mean`, `rolling_168_mean` — 24-hour and 168-hour trailing means

**Lag features (daily model for N-BEATSx):**
- `lag_1d`, `lag_7d`, `lag_14d` — previous day, same day last week, same day two weeks ago
- 7-day rolling mean and standard deviation

**Weather forecast discrepancy features (TA feedback addressed):**
For each weather variable *v*, a discrepancy feature is computed as:
```
Δv = forecast_v − observed_v
```
This captures systematic forecast bias and is available at prediction time. Δtemperature, Δprecipitation, Δwind, and Δsunshine are all included as model covariates.

**Spatial proximity features:**
`school_count`, `station_count`, `park_count`, `dist_nearest_school`, `dist_nearest_station`

Missing distance values are imputed using municipality-level medians, falling back to the global median.

---

## 3. Exploratory Data Analysis

### 3.1 Temporal Patterns

**Weekday vs. Weekend profiles.** Weekday counts exhibit a clear bimodal shape: a morning commuter peak around 08:00 and an afternoon/evening peak around 17:00–19:00. Weekend profiles collapse to a single broader recreational peak centred on midday (10:00–14:00). Weekday counts are 15–25% higher at high-activity infrastructure-supported sites, while lower-activity peripheral sites sometimes show the reverse, reflecting predominantly recreational use.

**Seasonal variation.** Cycling activity peaks between April and September, with a pronounced dip in December–January. A secondary peak in May–June is consistent with spring cycling promotions and pleasant weather. The amplitude of seasonal variation is substantially larger at infrastructure-supported sites. Year-on-year, established sites show stable counts rather than growth, suggesting the network has matured.

**Directional flow.** Inbound and outbound streams are near-symmetrical during commuting hours, confirming balanced two-way flows at most sensor locations.

### 3.2 Spatial Analysis

**Municipality ranking.** Top municipalities by average hourly count include **Hasselt**, **Leuven**, **Zemst**, **Antwerp**, and **Ghent**, reflecting dense commuter infrastructure and major cycling corridors (Fietssnelweg routes). Higher-activity sites cluster around the Antwerp–Brussels–Ghent triangle and Leuven; lower-activity peripheral sites are distributed across rural Flanders and coastal areas.

**Infrastructure proximity correlation.** A correlation heatmap confirms positive associations between `count` and `school_count`, `station_count`, and `park_count`, with inverse correlations for `dist_nearest_station` and `dist_nearest_school`. Sites embedded in multi-modal or educational catchments consistently attract higher volumes.

### 3.3 Weather Sensitivity

| Factor | Effect on Cycling Volume |
|---|---|
| Temperature | Positive; volumes rise with moderate temperatures (15–25 °C), plateauing above 20 °C; r ≈ 0.55–0.65 |
| Solar radiation | Positive linear trend; more sunlight → more cyclists (partially collinear with temperature) |
| Precipitation | Negative; light rain (≥1 mm/h) causes measurable drops; heavy rain (≥5 mm/h) causes 40–60% drops; r ≈ −0.35 to −0.50 |
| Snowfall | Strong negative; **~46% reduction** in mean count on snowy days |
| Wind speed | Moderate negative; stronger effect at coastal sites (r ≈ −0.30) |

Cross-site heterogeneity is significant: peripheral and recreational sites show much higher rain-sensitivity than infrastructure-supported commuter sites.

### 3.4 OLS Baseline Model

An Ordinary Least Squares regression including temporal, weather, and spatial features achieves an **R² of 0.131** on 2024 data. While modest, all selected predictors are highly statistically significant (all p-values ≈ 0.000), confirming genuine explanatory power and motivating more expressive non-linear models.

---

## 4. Behavioural Clustering of Counting Sites

### 4.1 Methodology

Two complementary clustering approaches were applied:

1. **k=2 global clustering** on 15 standardised site-level features — provides a clean, strategically actionable binary split.
2. **k=4 hourly profile clustering** on row-normalised 24-hour count vectors — provides finer temporal archetypes.

For the global clustering, features were grouped into three categories:
- **Usage intensity:** `mean_count`, `median_count`, `max_count`, `std_count`
- **Temporal profile:** `weekday_mean`, `weekend_mean`, `rush_hour_mean`, `weekend_weekday_ratio`, `rush_regular_ratio`, `count_variability`
- **Spatial context:** `park_count`, `school_count`, `station_count`, `dist_nearest_station`, `dist_nearest_school`

Cluster count selection was validated using the **Silhouette Score**; k=2 provided the cleanest separation (Silhouette ≈ 0.42). **PCA** (PC1 = 45.8% variance, PC2 = 12.5% variance) confirmed good cluster separation in the first two principal components.

### 4.2 k=2 Global Archetypes

| Feature Group | Cluster 0 — Lower-activity / Peripheral | Cluster 1 — Higher-activity / Infrastructure-supported |
|---|---|---|
| Usage intensity (mean_count) | −0.59 (2024), −0.49 (2025) | +0.90 (2024), +1.20 (2025) |
| Rush-hour mean | −0.59 (2024), −0.48 (2025) | +0.89 (2024), +1.20 (2025) |
| Weekend/weekday ratio | Near-zero (−0.04) | Near-zero (+0.061) — both clusters weekday-dominant |
| Count variability | +0.18 (2024) | −0.27 (2024) — higher-activity is more temporally stable |
| Park count | −0.42 (2024) | +0.64 (2024) — near parks and schools |
| Dist. nearest station | +0.35 (2024), +0.12 (2025) | −0.54 (2024), −0.31 (2025) — closer to transit |

**Spatial shift analysis (2024 → 2025).** Applying the 2024-trained model to 2025 data identifies three groups: "Stable lower-activity," "Shifted downward: higher to lower," and "Stable higher-activity." Several sites near Antwerp and Brussels shifted downward between years, potentially indicating infrastructure disruption, sensor recalibration, or genuine behavioural change warranting investigation.

### 4.3 k=4 Hourly Profile Archetypes

K-means (k=4) applied to row-normalised 24-hour count profiles reveals four finer usage archetypes:

| Cluster | Name | Characteristics |
|---|---|---|
| 0 | **Standard Commuter (Mixed)** | Moderate bimodal peaks; typical urban cycling route |
| 1 | **Recreational / Park (Elastic)** | Single midday peak; high weekend–weekday ratio; weather-sensitive |
| 2 | **Low Volume / Outliers** | Flat profile; low counts; rural or peripheral sensors |
| 3 | **High-Peak School / Work (Critical)** | Sharp morning (08:00) and afternoon (16:00) spikes; strong weekday dominance |

**POI cross-tabulation:** School POI sites → predominantly Cluster 3; Park POI sites → Cluster 1; Station POI sites distributed across Clusters 0 and 3, reflecting their dual commuting and transfer role.

---

## 5. Weather Resilience Audit & Infrastructure Prioritisation

### 5.1 Resilience Scoring Framework

For each site, a weather-sensitivity regression quantifies how much variance in daily counts is explained by weather variables. A composite **Resilience Score** is computed:

```
Resilience Score = 1 − (weather_R² × 0.5 + rain_sensitivity_coeff × 0.3 + wind_sensitivity_coeff × 0.2)
```

A site-level operationalisation also computes:
```
drop_pct = (1 − avg_count_rainy / avg_count_clear) × 100
resilience_score = 100 − drop_pct
```

**Results by POI group:**
- **Park** sites suffer the greatest proportional drop on rainy days — recreational trips are highly discretionary.
- **Station** and **School** sites are more resilient — commuters travel regardless of weather, though measurable suppression persists.
- Infrastructure-supported **Cluster 1** sites consistently show higher mean Resilience Scores, confirming that proximity to stations, schools, and parks correlates with weather-independent use.
- Coastal and rural **Cluster 0** sites show the lowest scores — counts collapse in winter and on rainy days.
- Sites in **Antwerp, Ghent, and Leuven** show above-average resilience, consistent with high commuter density and well-developed cycling infrastructure.

### 5.2 Psychological Anticipation vs. Physical Reaction

A behavioural phase analysis categorises each observation by shifting the precipitation column by one hour to define a one-hour forecast window:

1. **Clear Baseline** — no current or forecast rain
2. **Psychological Anticipation** — rain forecast in the next hour but not yet falling
3. **Physical Reaction** — rain currently falling

Cycling volume begins to drop **before** rain arrives, confirming a pre-emptive behavioural response: cyclists plan alternative transport when rain is forecast, not just when it falls. This effect is most pronounced at School and Station sites where routine trip-planning is most common.

### 5.3 Investment Priority Matrix

A **2×2 priority matrix** combining Resilience Score and usage volume provides four strategically distinct quadrants directly actionable by government planners:

| Quadrant | Description | Recommended Action |
|---|---|---|
| **High usage, high resilience (Backbone)** | Commuter spines: stable, essential corridors | All-weather maintenance; snow removal, surface repair |
| **High usage, low resilience (At-risk)** | High-traffic sites vulnerable to weather disruption | Covered paths, drainage, lighting investment |
| **Low usage, high resilience (Latent)** | Stable but low counts — suppressed demand | Marketing, signage, connectivity improvements |
| **Low usage, low resilience (Greenways)** | Recreational/leisure sites, weather-dependent | Seasonal maintenance; experience-driven enhancements |

For the subset of **School** and **Station** sites (essential demand categories), a composite **Investment Priority Score** is computed:

```
priority_score = 0.5 × normalised_traffic_volume + 0.5 × normalised_vulnerability
```

The **top 15 priority sites** are visualised on an interactive Folium map with markers colour-coded by score, enabling planners to identify geographical clusters of need.

---

## 6. Demand Forecasting

### 6.1 Modelling Strategy

A strictly forward-looking temporal split is used:
- **Train:** 2023–2024 (two full seasonal cycles)
- **Validation:** 2025 (model selection and hyperparameter tuning)
- **Test:** 2026 (strict holdout)

Both **hourly** (XGBoost/Random Forest/LightGBM) and **daily** (N-BEATSx) forecasting frameworks were developed. All models are trained as **global forecasters** across all sites simultaneously, with `site_id` as a categorical feature.

The target variable for the hourly pipeline is `log1p(count)` (back-transformed with `expm1`), stabilising variance across sites with very different absolute volumes.

### 6.2 Hourly Forecasting Pipeline

**Feature set (23+ features):**

| Category | Features |
|---|---|
| Spatial | `site_code`, `lat`, `lon`, `school_count`, `park_count`, `station_count`, `dist_nearest_school`, `dist_nearest_station` |
| Temporal | `hour_sin/cos`, `dow_sin/cos`, `month_sin/cos`, `is_weekend`, `is_holiday`, `is_pre_holiday` |
| Autoregressive | `lag_1`, `lag_24`, `lag_168`, `rolling_24_mean`, `rolling_168_mean` |
| Weather (observed) | `temperature_2m`, `precipitation`, `wind_speed_10m`, `shortwave_radiation`, `relative_humidity_2m` |
| Weather (discrepancy) | `Δtemperature`, `Δprecipitation`, `Δwind`, `Δsunshine` |
| Cluster | Binary cluster label from k=2 model |

**Mean encoding (XGBoost, `forecasting_xgboost.ipynb`):** Target-mean encodings are computed per (site, direction, hour), (site, direction, day_of_week), and (site, direction, month) strictly within each training fold to prevent leakage. 4-fold sliding-window cross-validation with 3-month validation windows is used within 2024.

**Model comparison (2025 validation):**

| Model | MAE (hourly) | MAE (daily) | R² | Notes |
|---|---|---|---|---|
| Historical Baseline (7-day lag) | ~10.0 | ~18 | ~0.45 | Mean count by (site, hour, day-of-week) |
| Random Forest | ~5.0 | ~11.2 | ~0.85 | 50 trees, max depth 12 |
| XGBoost | ~4.9 | ~9.8 | ~0.86 | 800 estimators, depth 6, early stopping |
| **XGBoost + Δweather features** | — | **~9.1** | **~0.87** | Best validation — discrepancy features add meaningful lift |
| LightGBM | ~4.8 | — | ~0.87 | 800 estimators, 64 leaves |
| **Ensemble (LightGBM + RF)** | **~4.7** | — | **~0.88** | Optimised blend weights |

All ML models reduce the average prediction error by approximately **50% relative to the historical baseline**. The inclusion of Δweather features improves XGBoost MAE from ~9.8 to ~9.1, validating the TA's suggestion.

**Final test on 2026:** The best-performing XGBoost model (with discrepancy features), retrained on 2023–2025 data, achieves a test MAE within **8% of validation MAE**, indicating minimal overfitting and strong generalisation.

### 6.3 Feature Importance

**Random Forest feature importance** reveals autoregressive lag features dominate:
1. `lag_1` (previous hour count) — strongest signal
2. `lag_168` (same hour, one week ago)
3. `lag_24` (same hour, yesterday)
4. `rolling_24_mean`, `rolling_168_mean`

**SHAP analysis (XGBoost)** top features by mean |SHAP|:
1. `lag_7d` (same day last week) — strongest predictor across all sites
2. `temperature_2m` (observed) — especially influential at recreational sites
3. `month` — captures seasonal amplitude
4. `precipitation` and **`Δprecipitation`** (forecast error) — both significant, validating discrepancy feature engineering
5. `cluster_label` — contributes meaningfully, confirming behavioural archetypes carry predictive value

Time-of-day (`hour_sin/cos`), other weather variables, and spatial features provide secondary but meaningful lift, consistent across walk-forward CV folds.

### 6.4 Validation Diagnostics

- **Actual vs. predicted scatter (2025):** Good alignment for low-to-medium counts; systematic under-prediction at high-volume peak hours — a known limitation of log-transformed tree models.
- **Residual distribution:** Approximately centred at zero with slight right skewness, indicating minor persistent under-prediction during peaks.
- **MAE-by-hour:** Lowest during overnight hours (00:00–05:00); rises sharply during morning (07:00–09:00) and afternoon (15:00–17:00) commuting windows — the operationally most critical periods.

### 6.5 Walk-Forward Cross-Validation

Four expanding-window folds (Q1→Q2, Q1-Q2→Q3, Q1-Q3→Q4, 2024→2025-Q1) confirm temporal stability:
- MAE decreases as more historical training data becomes available
- R² improves from earlier to later folds
- The ensemble consistently matches or slightly outperforms standalone Random Forest in later folds

### 6.6 N-BEATSx Daily Forecasting (Secondary Experiment)

As a secondary deep learning experiment, **N-BEATSx** (Neural Basis Expansion Analysis for Time Series with Exogenous variables) was trained on daily cyclist counts (`N-BEATSx.ipynb`).

**Data pipeline:**
- AWV monthly CSVs (2024–2026) downloaded directly from opendata.apps.mow.vlaanderen.be
- Aggregated to daily totals per site; gap-filled via linear interpolation (up to 3 days) then seasonal fallback (±7 days)
- **Two-layer outlier detection:**
  1. *Rolling median filter:* flags counts exceeding 4× the 30-day local median
  2. *Isolation Forest (per-site multivariate):* detects contextualised anomalies (e.g., suspicious count given heavy rain on a Tuesday in January); flagged days retained as `sensor_fault` binary covariate
- Per-series z-score normalisation for equal site weighting during training

**Model configuration:**
- Horizon: 7 days; lookback window: 365 days
- Interpretable mode: trend + seasonality stacks
- Historical exogenous: `temperature_2m_max`, `precipitation_sum`, `wind_speed_10m_max`, `sunshine_duration`, `sensor_fault`
- Future exogenous: `temperature_2m_max`, `precipitation_sum`, `wind_speed_10m_max`, `sunshine_duration` (from weather forecast)
- Temporal split: Train → 2025-09-30; Val → 2025-10-01 to 2025-12-31; Test → 2026+

**Results:** N-BEATSx achieves ~10.5 cyclists/day MAE on the 7-day horizon, competitive but not superior to XGBoost at this data scale. This is consistent with the TA's caution about deep learning being "overkill." N-BEATSx is retained as a documented experiment illustrating the trade-off between model complexity and interpretability; its interpretable stacks provide potentially useful per-site trend and seasonality decompositions for future analytical work.

---

## 7. Sustainability Impact Assessment

### 7.1 Assumptions

| Parameter | Value | Source |
|---|---|---|
| Average trip distance | 5.0 km | Belgian national cycling survey median |
| Modal substitute | Each cycling trip substitutes one car trip of equal distance | National transport statistics |
| CO₂ per car-km | 0.21 kg CO₂/km | EU average passenger car, 2023 EEA figure |
| Fuel saving | 0.065 L petrol/km | Average passenger car consumption |
| Scope | FIETSERS-type observations at sites with valid 2024–2025 data | — |

Three scenarios are modelled for sensitivity analysis:

| Scenario | Trip Distance | Modal Substitute |
|---|---|---|
| Conservative | 3 km | 70% car trips (30% replace walking/transit) |
| Base case | 5 km | 100% car substitution |
| Optimistic | 7 km (e-bike trips) | 100% + 10% induced demand bonus |

### 7.2 Network-Level Metrics (2024 Estimates)

| Metric | Estimated Value |
|---|---|
| Total cyclist-trips recorded | Computed from total hourly count aggregates across 143 monitored sites |
| Total distance cycled (estimated) | Total trips × 5 km |
| CO₂ avoided (vs. car substitute) | Total distance × 0.21 kg/km |
| Fuel saved (litres petrol equivalent) | Total distance × 0.065 L/km |
| Equivalent cars removed from road (annual) | CO₂ avoided / avg. annual car emissions |

**Municipality-level breakdown:** Antwerp, Ghent, and Leuven account for the largest absolute CO₂ savings; several smaller Flemish municipalities show the highest savings per capita relative to population density.

### 7.3 Strategic Network Roles

Based on the combined clustering, resilience, forecasting, and sustainability results, four actionable network roles emerge:

1. **Backbone Commuter Spines** — prioritise all-weather maintenance (snow removal, surface repair)
2. **Leisure Greenways** — seasonal experience enhancements; lower priority for year-round investment
3. **Capacity Gap Zones** — address bottlenecks to unlock latent cycling potential
4. **Cross-Border Benchmarks** — future comparison with Netherlands and Germany to identify best-practice corridors

---

## 8. Interactive Monitoring Dashboard

An operational **Shiny for Python** dashboard (`dashboard.py`) transforms model predictions into real-time management intelligence for traffic operators.

### 8.1 Key Features

- **Hour selector slider** (00:00–23:00) with animation playback mode
- **KPI summary bar:** Average Signed Error, Critical Surge Count, Major Blockage Count for the selected hour
- **Geospatial error map** (Plotly scatter-mapbox) colour-coded by signed prediction error (Actual − Predicted):
  - 🔴 **SURGE** (error > +140): actual traffic significantly higher than predicted — possible capacity or safety concern
  - 🟡 **BLOCKAGE** (error < −120): actual traffic much lower than predicted — potential infrastructure obstruction or sensor fault
  - 🟢 **Nominal:** within expected bounds
- **Investigation Queue table:** anomalous sites sorted by absolute error magnitude for dispatcher prioritisation

### 8.2 Design

The dashboard adopts a dark, high-contrast command-centre aesthetic (IBM Plex Mono, Syne typography), consistent with operational traffic management interfaces. The live-pulsing "LIVE MONITORING" indicator reinforces real-time situational awareness.

---

## 9. Conclusions & Policy Recommendations

### 9.1 Key Findings

1. **Cycling demand is primarily shaped by time of day and recent history.** The bimodal commuter pattern on weekdays and the recreational weekend peak are the dominant demand signatures; autoregressive lag features carry the most predictive power.

2. **Weather meaningfully suppresses cycling, with snowfall having the most severe impact (~46%).** Rain exerts a dual effect: a *physical* deterrent when falling, and a *psychological* deterrent in anticipation — cyclists cancel or modify trips even before rain begins.

3. **Two robust site archetypes** (infrastructure-supported vs. peripheral) and four finer hourly archetypes enable differentiated maintenance and monitoring strategies rather than one-size-fits-all approaches. The 2024→2025 spatial shift analysis identifies a small number of sites that degraded in activity, warranting investigation.

4. **School and Station sites near high-volume corridors represent the highest infrastructure investment priority**, combining heavy utilisation with measurable weather vulnerability.

5. **XGBoost with Δweather discrepancy features reduces forecasting error by ~50% over historical baselines** and generalises well to the 2026 holdout (test MAE within 8% of validation MAE). The Δweather features meaningfully improve performance, validating the TA's recommendation.

6. **N-BEATSx is competitive but not superior to XGBoost** at this data scale (MAE ~10.5 vs. ~9.1), confirming that well-engineered gradient boosting is the appropriate primary modelling choice.

7. **The Flemish cycling network avoids substantial CO₂ emissions annually.** Municipality-level CO₂ savings provide a ready-to-use communication tool for local governments.

### 9.2 Policy Recommendations for AWV and Municipalities

1. **Adopt the resilience priority matrix** as an annual maintenance planning tool, updated each year with new fietstellingen data.
2. **Investigate sites that shifted downward (2024 → 2025)** — these may indicate sensor faults, construction disruption, or behavioural change requiring intervention.
3. **Invest in weather protection at At-risk sites** (high-peak school/work sites with low resilience scores): covered shelters, improved drainage, better lighting.
4. **Deploy weather-responsive communication:** given the quantified anticipatory effect of forecasted rain, proactive push notifications to cycling apps during pre-rain windows can help cyclists plan alternative routes, reducing sudden surges on bus/car infrastructure.
5. **Use the forecasting model to anticipate seasonal capacity demands** and schedule maintenance windows during low-traffic periods (overnight, winter low-activity phases).
6. **Publish municipality-level CO₂ savings dashboards** to support public communication of cycling's environmental value.
7. **Extend the monitoring network** to capacity gap zones and municipalities showing rapidly growing cycling demand (e.g., Ghent and Bruges peri-urban corridors).
8. **Investigate the 40% of rain events missed by the forecast system** — improving NWP precipitation recall from ~60% to ~80% could meaningfully improve cycling demand predictions for rainy days.

### 9.3 Limitations and Future Work

- Analysis is limited to sensor-equipped sites; unmonitored corridors are invisible to the model.
- Modal substitution assumptions in the sustainability module are national averages and may not reflect local trip-purpose distributions.
- N-BEATSx underperformed relative to XGBoost at this data scale; a larger longitudinal dataset (post-2026) may make deep learning more competitive.
- Cross-border comparisons (Belgium vs. Netherlands vs. Germany) are scoped as future work.
- Sensor-fault flags (Isolation Forest) are included as covariates but not yet used to automatically trigger site inspection workflows in the dashboard.

---

## Appendix: Tools & Technologies

| Component | Technology |
|---|---|
| Data ingestion & wrangling | `pandas`, `numpy`, `pyarrow` |
| Weather data download | `openmeteo-requests`, `requests-cache`, `retry-requests` |
| Spatial analysis | 500 m buffer counting, OSM/AGIV proximity features |
| Visualisation | `matplotlib`, `seaborn`, `plotly`, `folium` |
| Baseline modelling | `statsmodels` (OLS) |
| Clustering | `scikit-learn` (KMeans, StandardScaler, PCA, IsolationForest) |
| Predictive modelling | `scikit-learn` (RandomForest), `xgboost`, `lightgbm` |
| Deep learning forecasting | `neuralforecast` (N-BEATSx), `keras`, `torch` |
| Dashboard | `shiny` (Python), Plotly Express |
| Dataset hosting | Hugging Face Hub (`huggingface_hub`) |
| Experiment environment | Google Colab (GPU/TPU), PyCharm Professional |

---

*Report prepared by Group 3, Modern Data Analytics, KU Leuven, May 2026.*
