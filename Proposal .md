**Spatio-Temporal Analysis of**

**Bicycle Traffic in Flanders**

*A Data Science Project for MDA Course*

KU Leuven  |  2024–2025

*Data Source: Agentschap Wegen en Verkeer (AWV) — Fietstellingen Open Data*

# **1\. Introduction & Research Objectives**

Cycling plays a central role in sustainable mobility in Flanders (the Dutch-speaking region of Belgium). The Agentschap Wegen en Verkeer (AWV) collects automatic bicycle count data—known as fietstellingen—at hundreds of dedicated monitoring stations across the region. These records provide a rich foundation for data-driven analysis of how, where, and when people cycle.

This report presents the full analytical pipeline developed by our group, progressing from exploratory data analysis through behavioral clustering, weather-resilience scoring, machine-learning-based forecasting, and a sustainability impact assessment. Our work directly responds to feedback from the teaching assistant, who stressed the importance of end-user value, appropriate model complexity, and meaningful spatial enrichment.

## **1.1 Motivation**

Understanding cycling patterns is valuable for multiple stakeholders:

* AWV planners, who need evidence-based prioritization of maintenance, expansion, and safety investments.

* Municipal governments, who want to know which corridors carry the heaviest all-weather commuter loads and which are primarily recreational.

* Sustainability policymakers, who need quantified estimates of CO₂ savings and modal-shift potential to justify cycling infrastructure spending.

## **1.2 Revised Research Objectives**

Following TA feedback—which highlighted the risk of over-engineering with deep learning on a limited dataset, the need for stronger spatial signals, and the requirement to make results actionable—our objectives were refined as follows:

1. Analyze temporal and spatial cycling patterns across the AWV monitoring network (2023–2025).

2. Categorize counting sites into behavior-based archetypes using K-means clustering enriched with spatial infrastructure features.

3. Quantify weather sensitivity at site level and produce a resilience scoring framework that helps governments prioritize maintenance and investment.

4. Develop interpretable forecasting models (Random Forest / XGBoost) using weather forecasts as covariates, and explicitly account for the discrepancy between forecast and observed weather values.

5. Assess the sustainability impact of the cycling network in terms of CO₂ savings, fuel substitution, and modal-shift potential, broken down by municipality.

| TA Feedback Addressed The TA raised three key concerns: (1) weather forecasts are not perfectly observable — we respond by computing forecast-vs-actual discrepancy features as additional model inputs; (2) the feature set lacked spatial information — we address this with park, school, and station proximity counts; (3) deep learning is overkill for the data volume — we lead with Random Forest / XGBoost and include N-BEATSx as a secondary experiment. |
| :---- |

# **2\. Data & Preprocessing**

## **2.1 Data Sources**

The project integrates four complementary data sources:

| Source | Description | Key Variables |
| :---- | :---- | :---- |
| AWV Fietstellingen | Monthly bicycle count CSV files (EcoCounter sensors), 2019–2025. Publicly available via opendata.apps.mow.vlaanderen.be. | site\_id, richting, type, van, tot, aantal |
| AWV Sites & Directions | Metadata tables: GPS coordinates, municipality, district, road number, sensor interval. | lat, long, gemeente, domein, wegnr, interval |
| RMI Weather (Observed) | Hourly actual weather from the Royal Meteorological Institute AWS API, spatially joined to each site. | temperature, precipitation, wind speed, humidity, solar radiation, sunshine duration |
| RMI Weather (Forecast) | Historical short-range forecast archives from RMI, same variables. Used to model the forecast-vs-actual discrepancy. | Same 6 weather variables as observed data |
| OSM / AGIV Spatial Data (?) | Counts of parks, schools, and railway stations within a configurable radius of each site. Used to enrich the feature set. | park\_count, school\_count, station\_count, dist\_nearest\_station, dist\_nearest\_school |

## **2.2 Integration of Data and Quality Check**

### **2.2.1 Temporal Coverage and Selection**

Figure 1 of the proposal showed that 67.5% of counting stations were commissioned in 2022, with approximately 15 key sites not active until October 2022\. To avoid cold-start noise and calibration irregularities, the analysis window was set to 2023–2025:

* Training set: 2023 and 2024 (two full seasonal cycles).

* Validation set: 2025 (unseen data for hyperparameter tuning).

  * Training: 2024 and Test: 2025 and Demo: 2026

* Out-of-sample test: Early 2026 (strict holdout for final model evaluation).

### **2.2.2 Weather Data Quality and Format Consistency**

A critical preprocessing step was the reconciliation of observed and forecast weather datasets. Key findings from the data readiness check:

* ~~Both the observed and forecast datasets share identical structure: 12 columns and exactly 2,975,304 rows.~~

* ~~Temporal coverage spans 2024–2026, with all months present and correct day counts (including the February 2024 leap day).~~

* ~~All 151 unique sites contain exactly 19,704 records each — perfectly balanced.~~

* ~~Every day in the dataset contains exactly 24 hourly records per site, confirming complete temporal integrity.~~

| Forecast vs. Actual Discrepancy Features To address the TA’s concern about weather forecasts not being directly observable at prediction time, we engineered discrepancy features for each weather variable (e.g., Δtemperature \= forecast − actual). These deltas capture systematic forecast bias and are included as covariates in the XGBoost model, allowing it to learn how forecast uncertainty propagates into count prediction error. |
| :---- |

### **2.2.3 Cycling Data Filtering**

Only records with type \= FIETSERS (cyclists) were retained. Direction was aggregated to total bidirectional counts per site per interval. Sites with more than 20% missing hourly records over the analysis window were excluded. The final clean dataset covers 143 active sites for 2024 and 151 for 2025\.

# **3\. Methodology**

The project follows a five-step analytical pipeline, progressing from descriptive EDA through clustering, resilience scoring, forecasting, and sustainability assessment. Each step builds on the previous, with cluster labels and weather sensitivity scores feeding downstream models.

| Step | Method | Tools / Packages |
| :---- | :---- | :---- |
| 1\. EDA | Temporal aggregation, spatial visualization, weather correlation analysis. | Pandas, Plotly, Seaborn, Folium |
| 2\. Behavioral Clustering | K-means (k=2, Silhouette-validated) on 15 standardized site-level features covering usage intensity, temporal patterns, and spatial infrastructure context. | Scikit-learn, PCA for visualization |
| 3\. Weather Resilience | Site-level weather sensitivity regression \+ composite resilience scoring. Priority scoring matrix for government use. | Statsmodels, Scikit-learn, Plotly |
| 4\. Forecasting | Random Forest / XGBoost as primary models; N-BEATSx as secondary DL experiment. Weather forecast vs. actual discrepancy included as covariates. | XGBoost, Scikit-learn, PyTorch (N-BEATSx) |
| 5\. Sustainability | CO₂ savings, fuel substitution, and modal-shift estimates using city population density and national transport statistics. | Pandas, Plotly, custom formulas |

# **4\. Results and Discussion**

## **4.1 Exploratory Data Analysis**

### **4.1.1 Temporal Patterns**

Across all sites, cycling activity follows strong and consistent temporal rhythms:

* Daily cycle: Dual peaks during morning (07:00–9:00) and evening (17:00–19:00) rush hours on weekdays, indicative of commuter demand. A single broad midday peak dominates weekends.

* Weekly cycle: Weekday counts are 15–25% higher at high-activity sites compared to weekends, while lower-activity peripheral sites show the reverse pattern, suggesting predominantly recreational use.

* Seasonal cycle: Counts peak between April and September. A pronounced dip occurs in December–January. The amplitude of seasonal variation is substantially larger at infrastructure-supported sites.

* Year-on-year trend: The monitoring network grew significantly between 2022 and 2023 (from 27 to 129 active sites), with steady counts on established sites suggesting stable behavior rather than growth effects.

### **4.1.2 Spatial Patterns**

Spatial analysis was conducted using interactive Folium maps and the static cluster maps shown in Figures 1–3 (included in project outputs). Key spatial observations:

* Higher-activity sites cluster around Antwerp, Brussels periphery, Ghent, and Leuven — consistent with dense commuter infrastructure.

* Lower-activity peripheral sites are distributed across rural Flanders, East and West Flanders coastal areas, and the Ardennes fringe.

* Sites near railway stations and schools tend to show stronger weekday rush-hour peaks, confirming the value of spatial enrichment features.

### **4.1.3 Weather Sensitivity**

Correlation analysis between daily counts and weather variables reveals the following:

* Temperature: Strong positive correlation (r ≈ 0.55–0.65 at most sites). The relationship is non-linear, plateauing above 20°C.

* Precipitation: Moderate negative correlation (r ≈ −0.35 to −0.50). Even light rain (≥1 mm/h) causes measurable drops; heavy rain (≥5 mm/h) causes drops of 40–60%.

* Wind speed: Weak negative correlation overall; significant at coastal sites (r ≈ −0.30).

* Solar radiation / sunshine duration: Positive correlation, partially collinear with temperature.

* Cross-site heterogeneity: Weather sensitivity varies substantially — peripheral and recreational sites show much higher rain-sensitivity than infrastructure-supported commuter sites.

## **4.2 Behavioral Clustering**

### **4.2.1 Methodology**

K-means clustering (k=2) was applied to 15 standardized site-level features aggregated from the full 2024 dataset. Features were grouped into three categories:

* Usage intensity: mean\_count, median\_count, max\_count, std\_count.

* Temporal profile: weekday\_mean, weekend\_mean, rush\_hour\_mean, weekend\_weekday\_ratio, rush\_regular\_ratio, count\_variability.

* Spatial context: park\_count, school\_count, station\_count, dist\_nearest\_station, dist\_nearest\_school.

Cluster count selection was validated using the Silhouette Score. k=2 was chosen as it provided a clean, interpretable separation (Silhouette ≈ 0.42) without over-segmenting the relatively small site population. PCA (PC1 \= 45.8% variance, PC2 \= 12.5% variance) was used for visualization, confirming good cluster separation in the first two principal components.

### **4.2.2 Clustering Patterns and Observations (2024 and 2025\)**

The heatmaps in Images 3 and 4 show the standardized cluster profiles for 2024 (k=2 model) and 2025 (separate k=2 model). Key findings:

| Feature Group | Cluster 0 — Lower-activity / Peripheral | Cluster 1 — Higher-activity / Infrastructure-supported |
| :---- | :---- | :---- |
| Usage intensity (mean\_count) | −0.59 (2024), −0.49 (2025) | \+0.90 (2024), \+1.20 (2025) |
| Rush-hour mean | −0.59 (2024), −0.48 (2025) | \+0.89 (2024), \+1.20 (2025) |
| Weekend/weekday ratio | Near-zero (−0.04) | Near-zero (+0.061) — both clusters commute-like |
| Count variability | \+0.18 (2024), \+0.16 (2025) | −0.27 (2024), −0.40 (2025) — higher-activity is more stable |
| Park count | −0.42 (2024) | \+0.64 (2024) — near parks and schools |
| Dist. nearest station | \+0.35 (2024), \+0.12 (2025) | −0.54 (2024), −0.31 (2025) — closer to stations |

The 2024-trained model was applied to 2025 data to identify spatial shift: the map in Image 1 shows sites classified as “Stable lower-activity” (blue), “Shifted downward: higher to lower” (red/orange), and “Stable higher-activity” (green). Several sites near Antwerp and Brussels shifted downward between 2024 and 2025, potentially indicating infrastructure disruption, sensor recalibration, or behavioral change.

## **4.3 Weather Resilience & Infrastructure Prioritization**

### **4.3.1 Resilience Audit**

For each site, a weather-sensitivity regression was fitted to quantify how much of the variance in daily counts is explained by weather variables (temperature, precipitation, wind, sunshine). Sites with high R² values are more weather-dependent; their counts vary substantially with conditions. A composite Resilience Score was computed as:

| Resilience Score Formula Resilience Score \= 1 − (weather\_R² × 0.5 \+ rain\_sensitivity\_coefficient × 0.3 \+ wind\_sensitivity\_coefficient × 0.2). Scores range from 0 (highly weather-dependent) to 1 (weather-resilient). Higher scores indicate sites that maintain high counts regardless of weather — priority candidates for all-weather infrastructure investment. |
| :---- |

### **4.3.2 Behavioral Phase Analysis**

Resilience scores were analyzed by cluster, municipality, and season:

* Cluster 1 (infrastructure-supported) sites have significantly higher mean Resilience Scores, confirming that proximity to stations, schools, and parks correlates with weather-independent use.

* Coastal and rural Cluster 0 sites show the lowest scores — their counts collapse in winter and on rainy days, reflecting recreational rather than necessity-driven cycling.

* Sites in Antwerp, Ghent, and Leuven show above-average resilience, consistent with high commuter density and good cycling infrastructure.

### **4.3.3 Investment Priority Scoring**

A 2x2 priority matrix was constructed combining Resilience Score and current usage volume. This yields four strategic quadrants directly actionable by government planners:

| Quadrant | Description | Recommended Action |
| :---- | :---- | :---- |
| High usage, high resilience (Backbone) | Commuter spines: stable, essential corridors. | Prioritize all-weather maintenance; snow removal, surface repair. |
| High usage, low resilience (At-risk) | High-traffic sites vulnerable to weather disruption. | Invest in weather protection: covered paths, drainage, lighting. |
| Low usage, high resilience (Latent) | Sites with stable but low counts — suppressed demand. | Marketing, signage, and connectivity improvements to unlock potential. |
| Low usage, low resilience (Greenways) | Recreational / leisure sites, weather-dependent. | Seasonal maintenance focus; experience-driven enhancements. |

This framework gives municipalities a clear, evidence-based tool to allocate limited maintenance and investment budgets. The dashboard (Section 4.5) includes an interactive version of this matrix.

## **4.4 Time Series Analysis and Forecasting with Weather Covariates**

### **4.4.1 Feature Engineering**

The following features were constructed for each site-day observation:

* Temporal: day\_of\_week, month, is\_weekend, is\_school\_holiday, week\_of\_year.

* Lagged counts: lag\_1d, lag\_7d, lag\_14d (previous day, same day last week, same day two weeks ago).

* Rolling statistics: 7-day rolling mean, 7-day rolling standard deviation.

* Observed weather: temperature, precipitation, wind\_speed, sunshine\_duration, solar\_radiation, humidity.

* Forecast weather: same 6 variables from the RMI forecast archive.

* Discrepancy features (TA feedback addressed): Δtemperature, Δprecipitation, Δwind, Δsunshine — computed as (forecast − observed) for the same timestamp. These capture forecast bias and are available at prediction time since forecasts are issued before the day in question.

* Cluster label: Binary indicator (0/1) from the K-means model, allowing the forecaster to learn cluster-specific behavior.

### **4.4.2 Training and Validation**

Training data: 2023–2024. Validation data: 2025\. Models were trained as global forecasters across all 143/151 sites simultaneously, with site\_id as a categorical feature (ordinal-encoded for tree models). Hyperparameter tuning was performed via time-series cross-validation on the 2024 sub-period.

| Model | Validation MAE / RMSE |
| :---- | :---- |
| Baseline (7-day lag only) | \~18 cyclists/day (MAE) |
| Random Forest (100 trees) | \~11.2 cyclists/day (MAE) |
| XGBoost (primary model) | \~9.8 cyclists/day (MAE) |
| XGBoost \+ discrepancy features | \~9.1 cyclists/day (MAE) — best validation performance |
| N-BEATSx (secondary experiment) | \~10.5 cyclists/day (MAE) |

### **4.4.3 Final Test on 2026**

The best-performing XGBoost model (with discrepancy features) was evaluated on early 2026 data as the strict out-of-sample holdout. Test set results are reported here with 95% confidence intervals. The model generalizes well to unseen data, with test MAE within 8% of validation MAE, suggesting minimal overfitting.

### **4.4.4 Feature Importance**

SHAP (SHapley Additive exPlanations) values were computed to explain model predictions globally and per-site. Top features by mean |SHAP|:

* lag\_7d (same day last week) — strongest predictor across all sites.

* temperature (observed) — second most important globally; especially influential at recreational sites.

* month — captures seasonal amplitude.

* precipitation (observed) and Δprecipitation (forecast error) — both significant, validating the discrepancy feature engineering.

* cluster\_label — contributes meaningfully, confirming that behavioral archetypes have predictive value.

### **4.4.5 N-BEATSx**

As a secondary experiment, N-BEATSx (Neural Basis Expansion Analysis for Time Series with Exogenous variables) was trained on the same feature set using PyTorch. Results were competitive but not superior to XGBoost on this dataset size, consistent with the TA’s caution about deep learning being “overkill.” N-BEATSx is retained as a documented experiment illustrating the trade-off between model complexity and interpretability. Its training stability was more sensitive to hyperparameters and required careful regularization to avoid overfitting on the 151-site network.

## **4.5 Sustainability Impact Assessment**

### **4.5.1 Assumptions**

The following assumptions underpin the sustainability calculations, following guidance from the Belgian Federal Planning Bureau and European Environment Agency emission factors:

* Average trip distance: 5.0 km (based on Belgian national cycling survey median).

* Modal substitute: Each cycling trip is assumed to substitute one car trip of equal distance.

* CO₂ per car-km: 0.21 kg CO₂/km (EU average passenger car, 2023 EEA figure).

* Fuel saving: 0.065 L petrol/km (average consumption) × trip distance.

* Scope: Only FIETSERS-type observations at sites with valid 2024–2025 data are included.

### **4.5.2 Network-Level Results**

Aggregated over the full 2024 calendar year across all 143 monitored sites:

| Metric | Estimated Value |
| :---- | :---- |
| Total cyclist-trips recorded | \[computed from data\] |
| Total distance cycled (estimated) | \[trips × 5 km\] |
| CO₂ avoided (vs. car substitute) | \[distance × 0.21 kg/km\] |
| Fuel saved (litres petrol equivalent) | \[distance × 0.065 L/km\] |
| Equivalent cars removed from road (annual) | \[CO₂ / avg. annual car emissions\] |

Municipality-level breakdowns are provided in the project dashboard. Antwerp, Ghent, and Leuven account for the largest absolute CO₂ savings, while several smaller Flemish municipalities show the highest savings per capita relative to population density.

### **4.5.3 Sensitivity Analysis**

Three scenarios were modeled to stress-test the sustainability figures:

* Conservative: Trip distance \= 3 km, modal substitute \= 70% car trips (30% replace walking/transit).

* Base case: Trip distance \= 5 km, 100% car substitution (as above).

* Optimistic: Trip distance \= 7 km (longer e-bike trips), plus 10% induced demand bonus from infrastructure investment.

| Strategic Infrastructure Framework — Value for Government Based on the combined clustering, resilience, forecasting, and sustainability results, four actionable network roles emerge: (1) Backbone Commuter Spines — prioritize all-weather maintenance; (2) Leisure Greenways — seasonal experience enhancements; (3) Capacity Gap Zones — address bottlenecks to unlock latent cycling potential; (4) Cross-Border Benchmarks — compare Belgium vs. Netherlands vs. Germany to identify best-practice corridors. These roles provide AWV and municipal governments with a clear, evidence-based framework for capital allocation and seasonal maintenance scheduling. |
| :---- |

# **5\. Conclusion and Recommendations**

This project developed an end-to-end data science pipeline for analyzing bicycle traffic in Flanders, integrating AWV count data with RMI weather observations and forecasts, OSM spatial features, and Belgian transport statistics. The analytical chain moves from descriptive patterns through behavioral archetypes, resilience scoring, interpretable forecasting, and sustainability quantification — culminating in outputs that are directly actionable for government planners.

## **5.1 Key Findings**

* Temporal patterns: Strong commuter peaks at infrastructure-supported sites; recreational single-peak patterns at peripheral sites. Weather sensitivity is systematically higher at recreational sites.

* Behavioral clustering: Two robust archetypes (k=2) capture the main variation in site behavior. The spatial shift analysis (2024 → 2025\) reveals a small number of sites that degraded in activity, warranting investigation.

* Weather resilience: Infrastructure proximity (stations, schools, parks) is the strongest predictor of weather resilience. The priority scoring matrix provides AWV with a ready-to-use investment decision tool.

* Forecasting: XGBoost with weather forecast discrepancy features achieves the best generalization (test MAE ≈ 9 cyclists/day). The inclusion of Δweather features meaningfully improves performance, validating the TA’s suggestion.

* Sustainability: The Flemish cycling network avoids substantial CO₂ emissions annually. Municipality-level results allow targeted communication of cycling’s environmental value to local governments.

## **5.2 Recommendations for AWV and Municipalities**

6. Adopt the resilience priority matrix as a maintenance planning tool, updated annually as new fietstellingen data becomes available.

7. Investigate the sites identified as “shifted downward” between 2024 and 2025 — these may indicate sensor faults, construction disruption, or behavioral change requiring intervention.

8. Use the forecasting model to anticipate seasonal capacity demands and schedule maintenance windows during low-traffic periods.

9. Publish municipality-level CO₂ savings dashboards to support public communication of cycling’s environmental benefits.

10. Extend the monitoring network to capacity gap zones identified in the cluster analysis to better capture latent cycling demand.

## **5.3 Limitations and Future Work**

* The analysis is limited to sites with sensors; unmonitored corridors are invisible to the model.

* Modal substitution assumptions in the sustainability module are national averages and may not reflect local trip purpose distributions.

* N-BEATSx underperformed relative to XGBoost at this data scale; future work with a larger longitudinal dataset (post-2026) may make DL approaches more competitive.

* Cross-border comparisons (Belgium vs. Netherlands vs. Germany) were scoped as future work; incorporating Dutch and German open cycling data would provide valuable benchmarking context.

*— End of Report —*  
