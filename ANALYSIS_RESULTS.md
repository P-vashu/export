# Formula 1 Multi-Dataset Statistical Modeling Analysis Results

## Executive Summary

This document presents the complete results of a comprehensive multi-dataset statistical modeling analysis on historical Formula 1 data, covering driver championship performance with strict reproducibility requirements.

---

## Analysis Results

### PHASE 1: Data Merging and Aggregation

**Objective:** Merge drivers_updated.csv with winners.csv and aggregate win counts

**Methodology:**
- Left join on `drivers_updated.Code = winners.Name Code` AND `drivers_updated.year = winners.year`
- Aggregated winners.csv by grouping on Name Code and year to count wins
- Merged aggregated win counts back to driver dataset

**Result:**
- **Total rows in final merged dataset: 1661**

---

### PHASE 2: Data Cleaning

**Objective:** Clean and standardize the Pos column, handle missing Car values

**Methodology:**
- Converted Pos column from object dtype to numeric using `pd.to_numeric()` with `errors='coerce'`
- Dropped all rows where Car column contains missing values
- Calculated percentage of data loss

**Results:**
- **Percentage of rows dropped due to missing Car values: 0.66%**
- Rows before: 1661
- Rows after: 1650
- Rows dropped: 11

---

### PHASE 3: Kolmogorov-Smirnov Two-Sample Test

**Objective:** Compare PTS distributions between top finishers and others

**Methodology:**
- Group A: Drivers with Pos ≤ 3 (podium finishers)
- Group B: Drivers with Pos > 3 (non-podium finishers)
- Applied two-sample KS test to compare distributions

**Sample Sizes:**
- Group A: 226 samples
- Group B: 1423 samples

**Results:**
- **KS test statistic: 0.7622**
- **KS test p-value: 0.0000**

**Interpretation:** The p-value of 0.0000 (< 0.05) indicates a statistically significant difference between the PTS distributions of top finishers versus others. The high KS statistic (0.7622) suggests a substantial distributional difference.

---

### PHASE 4: SMOTE Oversampling

**Objective:** Create balanced dataset for binary classification of high performers

**Methodology:**
- Created binary target `high_performer`: 1 if PTS > 75th percentile, 0 otherwise
- 75th percentile threshold: 32.0 points
- Applied SMOTE with:
  - `random_state=42` (for reproducibility)
  - `sampling_strategy=1.0` (balance classes to 1:1 ratio)
  - Features: PTS, Pos, win_count

**Class Distribution Before SMOTE:**
- Class 0 (low performers): 1242 samples
- Class 1 (high performers): 407 samples

**Class Distribution After SMOTE:**
- Class 0: 1242 samples
- Class 1: 1242 samples (407 original + 835 synthetic)

**Result:**
- **Total count of synthetic samples generated: 835**

---

### PHASE 5: Quantile Regression

**Objective:** Model the relationship between year and normalized team PTS at the 90th quantile

**Methodology:**
- Extracted year from winners.Date column
- Aggregated race counts per year (75 unique years)
- Merged teams_updated.csv with annual_races on year
- Created `normalized_PTS = team PTS / annual_races`
- Fitted Quantile Regression at τ = 0.90 quantile
- Model: `normalized_PTS ~ year` (with constant term)

**Results:**
- **Intercept coefficient: -577.921**
- **Slope coefficient for year: 0.296**

**Interpretation:** For high-performing teams (90th quantile), each additional year is associated with an increase of 0.296 points in normalized team PTS. The negative intercept reflects the model's baseline before the modern F1 era.

---

### PHASE 6: Vector Autoregression (VAR)

**Objective:** Model the dynamic relationship between driver and team PTS over time

**Methodology:**
- Created two time series:
  1. Annual driver PTS sum (aggregated by year)
  2. Annual team PTS sum (aggregated by year)
- Time series length: 67 years (65 observations after lag adjustment)
- Fitted VAR model with lag order = 2
- Extracted coefficient representing team's PTS lag-1 effect on driver PTS

**VAR Model Specification:**
- Variables: driver_pts_sum, team_pts_sum
- Lag order: 2
- Observations: 65

**Results:**
- **Coefficient at row 0, column 2 (team's PTS lag-1 effect on driver PTS): -0.0036**
- **Akaike Information Criterion (AIC): 18.45**

**Interpretation:** The coefficient of -0.0036 suggests a negligible negative relationship between team PTS from the previous year and driver PTS in the current year. The low AIC indicates good model fit for the given lag structure.

---

### PHASE 7: Visualization

**Objective:** Visualize the relationship between Championship Points and Race Victories

**Methodology:**
- Created hexbin plot with:
  - X-axis: Driver PTS (Championship Points)
  - Y-axis: win_count (Race Victories)
  - Grid size: 30
  - Colormap: YlOrRd (Yellow-Orange-Red)

**Result:**
- Hexbin plot generated and saved as `f1_hexbin_plot.png`

**Key Observations:**
- Strong positive correlation between championship points and race victories
- High density of drivers at low points and win counts (darker red in bottom-left)
- Clear stratification showing that more wins lead to higher championship points
- Visual validation of the competitive structure of F1 championships

---

## Technical Details

### Reproducibility

All analyses were conducted with the following reproducibility measures:
- Random seed set to 42 for all stochastic operations
- SMOTE random_state = 42
- Fixed lag order (2) for VAR model
- Fixed quantile (0.90) for quantile regression

### Software Environment

- Python 3.11
- pandas 2.3.3
- numpy 2.3.4
- scipy 1.16.3
- scikit-learn 1.7.2
- imbalanced-learn 0.14.0
- statsmodels 0.14.5
- matplotlib 3.10.7

### Dataset Information

**Initial Dataset Sizes:**
- drivers_updated.csv: 1661 rows
- teams_updated.csv: 695 rows
- winners.csv: 1110 rows

**Final Working Dataset:**
- After merging and cleaning: 1650 rows

---

## Key Findings Summary

1. **Data Integration:** Successfully merged 1661 driver records with race win data, identifying 399 unique driver-year combinations with victories

2. **Data Quality:** High data quality with only 0.66% of records requiring removal due to missing car information

3. **Performance Distribution:** Statistically significant difference (p < 0.0001) between top-3 finishers and other drivers in terms of championship points

4. **Class Balance:** Successfully created balanced dataset for high-performer classification using SMOTE, generating 835 synthetic samples

5. **Temporal Trends:** Normalized team performance shows positive temporal trend (0.296 points/year at 90th quantile)

6. **Inter-Series Dynamics:** VAR analysis reveals minimal direct lag-1 effect (-0.0036) from team PTS to driver PTS

7. **Performance Correlation:** Strong positive relationship between race victories and championship points, as expected in F1's points-based system

---

## Files Generated

1. `f1_statistical_analysis.py` - Complete analysis script with all phases
2. `f1_hexbin_plot.png` - Hexbin visualization of points vs victories
3. `ANALYSIS_RESULTS.md` - This summary document

---

## Conclusion

This comprehensive multi-dataset statistical modeling analysis successfully executed all required phases with strict reproducibility. The results provide robust insights into F1 driver championship performance across multiple analytical dimensions: data merging, statistical testing, machine learning preprocessing, regression modeling, time series analysis, and visualization.

All code and results have been committed to the repository for full transparency and reproducibility.

---

**Analysis Completed:** November 8, 2025
**Branch:** claude/f1-statistical-modeling-analysis-011CUunrqUJdWapwfZURqhN8
