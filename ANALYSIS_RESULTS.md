# Aviation Network Statistical Analysis Results

**Principal Aviation Safety Analyst**
**International Civil Aviation Organization**

---

## Executive Summary

This report presents complex statistical operations quantifying dependencies in aviation networks using three datasets: airlines.csv, airports.csv, and airplanes.csv. The analysis includes data preprocessing, statistical tests, machine learning models, and visualizations.

---

## 1. Data Preprocessing Summary

### Airports Dataset
- **Initial records**: 7,698
- **After DST filtering (E, U, N)**: 4,874 records
- **Timezone conversion**: String values parsed as floats, `\N` replaced with -999
- **Non-negative Altitude records**: 4,862

### Airlines Dataset
- **Initial records**: 6,162
- **After removing both ICAO and Callsign = `\N`**: 6,161 records
- **After filtering Active = 'Y' (case-sensitive)**: 1,254 records

### Airplanes Dataset
- **Total records**: 246
- **Unique IATA codes**: 221

### Aggregations
- **Countries with airports**: 236
- **Countries after inner join with airlines**: 1,227 records
- **Countries with > 10 airports**: 99

---

## 2. Statistical Analysis Results

### 2.1 Kendall Tau-b Correlation
**Test**: Association between median Altitude and standard deviation of Latitude
**Subset**: Countries with > 10 airports (n=99)
**Method**: Asymptotic

**Results:**
- **Kendall tau-b coefficient**: **0.15551**
- **P-value**: **0.0226376**

**Interpretation**: Weak positive correlation, statistically significant at α=0.05

---

### 2.2 Gaussian Mixture Model (GMM)
**Configuration**:
- Components: 3
- Covariance type: Full
- Random seed: 42
- Features: Standardized Latitude, Longitude, sqrt(Altitude)
- Sample size: 4,862 airports (Altitude ≥ 0)

**Results:**
- **Average silhouette coefficient**: **0.21801**
- **Component with maximum sample count**: **1**

**Interpretation**: Moderate cluster separation, indicating distinct airport groups

---

### 2.3 Elastic Net Regression
**Configuration**:
- Alpha: 0.5
- L1 ratio: 0.7
- Features: Degree 2 polynomial (Timezone, Latitude) without interactions
- Cross-validation: 5-fold

**Results:**
- **5-fold cross-validation MAE**: **1117.448**
- **Latitude squared coefficient**: **-0.125949**

**Interpretation**: Moderate prediction error; negative quadratic term suggests altitude decreases with extreme latitudes

---

### 2.4 Kruskal-Wallis H Test
**Test**: Comparing Altitude across DST categories
**Groups**:
- E: 1,610 samples
- U: 1,862 samples
- N: 1,402 samples

**Results:**
- **H statistic**: **16.0869**
- **P-value**: **0.00032120**

**Interpretation**: Statistically significant differences in altitude distributions across DST categories

---

### 2.5 Bootstrap Confidence Interval
**Configuration**:
- Iterations: 5,000
- Random seed: 123
- Metric: Interquartile range of Altitude
- Subset: Airports with positive Latitude (n=3,917)
- Confidence level: 95%

**Results:**
- **95% CI bounds**: **[880.00, 1047.00]**

**Interpretation**: The population IQR of altitude for airports in the Northern Hemisphere is estimated between 880 and 1,047 feet

---

### 2.6 OLS Regression
**Model**: Altitude ~ Longitude + DST_U
**Reference category**: DST = E
**Subset**: Airports with DST = E or U (n=3,472)

**Results:**
- **F-statistic**: **48.8491**
- **Adjusted R-squared**: **0.02683**

**Interpretation**: Statistically significant model but explains only 2.7% of altitude variance

---

### 2.7 t-SNE Dimensionality Reduction
**Configuration**:
- Components: 2
- Perplexity: 30
- Learning rate: 200
- Max iterations: 2,500
- Random seed: 99
- Features: Standardized Latitude, Longitude, Altitude, Timezone
- Sample size: 4,874 airports

**Results:**
- **Final KL divergence**: **0.46504**

**Interpretation**: Good convergence; KL divergence < 1 indicates successful dimensionality reduction

---

### 2.8 Mann-Whitney U Test
**Test**: Comparing Altitude between hemispheres
**Groups**:
- Positive Latitude (Northern): 3,917 samples
- Negative Latitude (Southern): 956 samples
**Configuration**: With continuity correction

**Results:**
- **U statistic**: **1745996**
- **P-value**: **0.00119807**

**Interpretation**: Statistically significant difference in altitude distributions between hemispheres

---

### 2.9 Hierarchical Clustering
**Configuration**:
- Linkage method: Ward
- Features: Standardized Latitude, Longitude
- Clusters: 6
- Sample size: 4,874 airports

**Results:**
- **Cophenetic correlation**: **0.64306**

**Interpretation**: Moderate preservation of pairwise distances; clustering reasonably represents the data structure

---

## 3. Visualizations

### Violin Plot: Altitude by DST Category
**File**: `altitude_by_dst_violin_plot.png`

**Description**: Violin plot with box plot overlay showing altitude distribution across DST categories (E, N, U), ordered alphabetically. The plot reveals:
- Distribution shapes and densities for each DST category
- Median, quartiles, and outliers via box plot overlay
- Differences in altitude patterns across daylight saving time zones

---

## 4. Technical Implementation

**Programming Language**: Python 3
**Key Libraries**:
- pandas: Data manipulation
- numpy: Numerical operations
- scipy: Statistical tests
- scikit-learn: Machine learning models
- matplotlib & seaborn: Visualization

**Reproducibility**: All analyses use specified random seeds (42, 99, 123) for reproducible results

---

## 5. Key Findings

1. **Network Dependencies**: Weak but significant correlation between altitude and latitude variance across countries
2. **Airport Clustering**: Three distinct airport groups identified through GMM
3. **Altitude Patterns**: Significant differences across DST categories and hemispheres
4. **Predictive Models**: Geographic features provide limited altitude prediction (R²=0.027)
5. **Dimensionality**: Four-dimensional airport characteristics successfully reduced to 2D representation
6. **Hierarchical Structure**: Geographic clustering reveals 6 distinct airport groups globally

---

## 6. Recommendations

1. Further investigate the altitude differences across DST categories for safety implications
2. Explore additional features to improve altitude prediction models
3. Analyze the identified airport clusters for operational patterns
4. Consider hemisphere-specific safety protocols based on observed differences

---

**Report Generated**: 2025-11-13
**Analysis Script**: `aviation_analysis.py`
**Visualization**: `altitude_by_dst_violin_plot.png`
