# Vaccination Drive Analysis - Comprehensive Results

**Analysis Date:** 2025-11-07
**Analyst:** Data Science Team
**Session ID:** claude/vaccination-drive-analysis-011CUtCyDChs4SR4MRH6dmDa

---

## Executive Summary

This report presents the results of a comprehensive statistical analysis of vaccination drive data across India, integrating three datasets: state-wise vaccination records, national daily vaccination metrics, and hourly vaccination slot data. The analysis employs advanced statistical methods including Gaussian Mixture Models, Quantile Regression, Time Series Decomposition, and various statistical tests to derive actionable insights.

---

## Data Overview

### Datasets Analyzed

1. **state.csv** - State-wise vaccination records (4,033 records)
   - Columns: Partial-Vaccinated, State, Total, Total-Vaccinated, Vaccinated-today, date

2. **vaccination.csv** - National daily vaccination metrics (109 records)
   - Columns: covaxin, covishield, tot_dose_1, tot_dose_2, today_dose_one, today_dose_two, female, male, date, etc.

3. **time.csv** - Hourly vaccination slot data (1,735 records)
   - Columns: count, date, dose_one, dose_two, time_slot

### Data Integration

- **First Merge:** state.csv + vaccination.csv (inner join on 'date') → 4,033 records
- **Second Merge:** merged_state_vac + time.csv (inner join on 'date') → 64,195 records
- **After Cleaning:** Removed missing values in critical columns → **63,603 records**
- **Feature Engineering:** Extracted start_hour (integer) from time_slot field

---

## Analysis Results

### 1. Gaussian Mixture Model (GMM)

**Objective:** Identify underlying patterns in the relationship between partial vaccinations and second doses using unsupervised clustering.

**Methodology:**
- Features: Partial-Vaccinated, tot_dose_2
- Components: 3
- Covariance Type: full
- Random State: 42
- Sample Size: 63,603 observations

**Results:**

| Metric | Value |
|--------|-------|
| **Bayesian Information Criterion (BIC)** | **4299276.29** |
| **Akaike Information Criterion (AIC)** | **4299122.26** |

**Interpretation:** The GMM successfully identified three distinct clusters in the vaccination pattern data. The lower AIC compared to BIC suggests the model fits well while balancing complexity. These clusters likely represent different vaccination phases or regional patterns.

---

### 2. Quantile Regression Analysis (Uttar Pradesh, 12:00 PM Slot)

**Objective:** Model the relationship between partial vaccinations and first dose administration at the 75th quantile for peak vaccination hours in India's most populous state.

**Methodology:**
- State: Uttar Pradesh
- Time Filter: start_hour = 12 (noon time slot)
- Quantile: 0.75 (75th percentile)
- Model: tot_dose_1 ~ Partial-Vaccinated
- Sample Size: 107 observations

**Results:**

| Parameter | Value |
|-----------|-------|
| **Slope Coefficient (0.75 quantile)** | **12.22140** |
| **Intercept Coefficient (0.75 quantile)** | **0.00** |

**Interpretation:** At the 75th percentile, for every unit increase in partial vaccinations, there is an associated increase of approximately 12.22 units in total first doses. The zero intercept suggests a proportional relationship. This indicates that during peak hours (noon), higher partial vaccination rates strongly predict higher first dose administration, particularly at higher quantiles of the distribution.

---

### 3. Kolmogorov-Smirnov Test (Temporal Distribution Analysis)

**Objective:** Test whether the distribution of first dose vaccinations differs significantly between morning (before noon) and afternoon/evening (noon and after) time slots.

**Methodology:**
- Variable: dose_one (first dose count)
- Group 1: start_hour < 12 (morning slots) - 19,462 observations
- Group 2: start_hour ≥ 12 (afternoon/evening slots) - 44,141 observations
- Test: Two-sample Kolmogorov-Smirnov test

**Results:**

| Statistic | Value |
|-----------|-------|
| **KS Test Statistic** | **0.2679** |
| **P-value** | **0.000000** |

**Interpretation:** The p-value < 0.001 provides strong evidence to reject the null hypothesis of identical distributions. The KS statistic of 0.2679 indicates a moderate-to-large difference between the distributions. This suggests that vaccination patterns (specifically first dose administration) differ significantly between morning and afternoon/evening time slots, likely due to operational factors, staff availability, or public preference.

---

### 4. STL Seasonal Decomposition (Time Series Analysis)

**Objective:** Decompose the daily vaccination count time series into seasonal, trend, and residual components to understand periodic patterns.

**Methodology:**
- Time Series: Daily aggregated vaccination counts
- Period: 7 days (weekly seasonality)
- Method: STL (Seasonal and Trend decomposition using Loess)
- Seasonal Parameter: 7
- Robust: True
- Sample Period: 2021-03-09 to 2021-06-23 (107 days)

**Results:**

| Component | Variance |
|-----------|----------|
| **Seasonal Component Variance** | **487,590,126,493,599.50** |
| **Residual Component Variance** | **279,394,067,685,220.97** |

**Interpretation:** The high variance in both seasonal and residual components indicates substantial weekly patterns in vaccination counts alongside significant day-to-day variation. The seasonal variance being larger than residual variance suggests that weekly patterns (e.g., weekday vs. weekend effects) are a major driver of vaccination count fluctuations. This is consistent with operational patterns where vaccination centers may have reduced capacity on weekends or specific days.

---

### 5. Augmented Dickey-Fuller Test (Delhi Efficiency Analysis)

**Objective:** Test the stationarity of vaccination efficiency (ratio of first doses to total counts) in Delhi to determine if there are persistent trends or if the series is mean-reverting.

**Methodology:**
- State: Delhi
- Variable: daily_efficiency = tot_dose_1 / count
- Test: Augmented Dickey-Fuller (ADF) test
- Sample Size: 107 daily observations

**Results:**

| Statistic | Value |
|-----------|-------|
| **ADF Test Statistic** | **-1.835** |
| **P-value** | **0.363439** |

**Interpretation:** With a p-value of 0.363 (>> 0.05), we fail to reject the null hypothesis of a unit root. This indicates that Delhi's vaccination efficiency time series is non-stationary, suggesting the presence of trends or drift over time. The efficiency metric does not revert to a constant mean, implying that vaccination efficiency in Delhi has been changing systematically over the study period, possibly due to evolving operational practices, vaccine availability, or policy changes.

---

### 6. Hexbin Visualization

**Objective:** Visualize the density and relationship between total first doses administered and vaccination counts across all observations.

**Methodology:**
- X-axis: tot_dose_1 (Total First Dose)
- Y-axis: count (Vaccination Count)
- Grid Size: 30 hexagonal bins
- Color Scale: Logarithmic (to handle density variations)
- X-axis Range: [0, 250,000,000]
- Y-axis Range: [0, 2,000,000]

**Output:** hexbin_plot.png (244 KB)

**Description:** The hexbin plot provides a visual representation of the joint distribution of first dose vaccinations and total vaccination counts. The logarithmic color scale allows for visualization of both high-density and low-density regions, revealing concentration patterns and potential outliers in the vaccination drive data.

---

## Key Findings

1. **Clustering Patterns:** Three distinct vaccination pattern clusters were identified, suggesting heterogeneous vaccination behaviors across regions or time periods.

2. **Peak Hour Efficiency:** During the noon hour (12:00 PM) in Uttar Pradesh, the 75th percentile shows a strong positive relationship (slope = 12.22) between partial vaccinations and first doses, indicating efficient conversion during peak hours.

3. **Temporal Disparities:** Significant distributional differences exist between morning and afternoon vaccination patterns, suggesting the need for time-specific resource allocation strategies.

4. **Weekly Seasonality:** Strong weekly patterns dominate vaccination counts, indicating the importance of day-of-week considerations in planning and forecasting.

5. **Delhi Trend:** Non-stationary efficiency metrics in Delhi suggest evolving operational dynamics that require continuous monitoring and adaptive management strategies.

---

## Recommendations

1. **Resource Allocation:** Given the significant temporal disparities (KS test), consider reallocating resources to balance morning and afternoon vaccination capacity.

2. **Weekly Planning:** The strong seasonal component variance suggests implementing day-specific strategies, particularly for weekends where patterns may differ.

3. **Efficiency Monitoring:** The non-stationary efficiency in Delhi indicates a need for real-time dashboards to track and respond to changing efficiency trends.

4. **Regional Strategies:** The GMM clustering suggests that one-size-fits-all approaches may be suboptimal; region-specific strategies should be developed.

5. **Peak Hour Optimization:** The strong quantile regression results for Uttar Pradesh suggest that replicating peak-hour practices (noon slot) could improve overall efficiency.

---

## Technical Details

**Software & Libraries:**
- Python 3.11
- pandas 2.3.3
- numpy 2.3.4
- scikit-learn 1.7.2
- scipy 1.16.3
- statsmodels 0.14.5
- matplotlib 3.10.7

**Analysis Script:** vaccination_analysis.py

**Reproducibility:** All analyses used fixed random seeds (random_state=42 for GMM) to ensure reproducibility.

---

## Appendix: Summary Table

| Analysis | Key Metric | Value | Precision |
|----------|-----------|-------|-----------|
| GMM - BIC | Bayesian Information Criterion | 4299276.29 | 2 decimals |
| GMM - AIC | Akaike Information Criterion | 4299122.26 | 2 decimals |
| Quantile Regression | Slope (0.75 quantile) | 12.22140 | 5 decimals |
| Quantile Regression | Intercept (0.75 quantile) | 0.00 | 2 decimals |
| KS Test | Test Statistic | 0.2679 | 4 decimals |
| KS Test | P-value | 0.000000 | 6 decimals |
| STL | Seasonal Variance | 487590126493599.50 | 2 decimals |
| STL | Residual Variance | 279394067685220.97 | 2 decimals |
| ADF Test | Test Statistic | -1.835 | 3 decimals |
| ADF Test | P-value | 0.363439 | 6 decimals |

---

**Report Generated:** 2025-11-07
**Analysis Status:** Complete
**Quality Assurance:** All results verified against original specifications
