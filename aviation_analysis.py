#!/usr/bin/env python3
"""
Advanced Statistical Analysis of Aviation Datasets
Senior Data Scientist - International Aviation Regulatory Body
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import spearmanr, ks_2samp
from sklearn.cluster import DBSCAN
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("AVIATION STATISTICAL ANALYSIS - COMPREHENSIVE REPORT")
print("="*80)
print()

# ============================================================================
# 1. DATA LOADING AND PREPROCESSING
# ============================================================================
print("1. DATA LOADING AND PREPROCESSING")
print("-" * 80)

# Load datasets
airlines = pd.read_csv('airlines.csv')
airports = pd.read_csv('airports.csv')
airplanes = pd.read_csv('airplanes.csv')

print(f"Initial data loaded:")
print(f"  Airlines: {len(airlines)} records")
print(f"  Airports: {len(airports)} records")
print(f"  Airplanes: {len(airplanes)} records")
print()

# Filter airports: DST equals E or U
airports_filtered = airports[airports['DST'].isin(['E', 'U'])].copy()
print(f"Airports after DST filter (E or U): {len(airports_filtered)} records")

# Convert Timezone to numeric: treat strings as floats, replace \N with 0
def convert_timezone(tz):
    if pd.isna(tz):
        return 0.0
    if isinstance(tz, (int, float)):
        return float(tz)
    if isinstance(tz, str):
        if tz == r'\N' or tz == '\\N':
            return 0.0
        try:
            return float(tz)
        except ValueError:
            return 0.0
    return 0.0

airports_filtered['Timezone'] = airports_filtered['Timezone'].apply(convert_timezone)
print(f"Timezone converted to numeric (\\N → 0)")

# Remove airlines records where ICAO contains \N
airlines_clean = airlines[~airlines['ICAO'].astype(str).str.contains(r'\\N', regex=False, na=False)].copy()
print(f"Airlines after removing ICAO with \\N: {len(airlines_clean)} records")

# Retain only Active = 'Y' (case-sensitive)
airlines_clean = airlines_clean[airlines_clean['Active'] == 'Y'].copy()
print(f"Airlines after filtering Active='Y': {len(airlines_clean)} records")
print()

# ============================================================================
# 2. GROUPING OPERATIONS
# ============================================================================
print("2. GROUPING OPERATIONS")
print("-" * 80)

# Group airplanes by Name, count occurrences
airplanes_grouped = airplanes.groupby('Name').size().reset_index(name='Count')
print(f"Airplanes grouped by Name: {len(airplanes_grouped)} unique aircraft types")
print(f"Sample aircraft counts:")
print(airplanes_grouped.head(10).to_string(index=False))
print()

# Group airports by Country: mean Altitude and count
airport_summary = airports_filtered.groupby('Country').agg(
    mean_Altitude=('Altitude', 'mean'),
    airport_count=('Airport ID', 'count')
).reset_index()
print(f"Airports grouped by Country: {len(airport_summary)} countries")
print(f"Sample country statistics:")
print(airport_summary.head(10).to_string(index=False))
print()

# ============================================================================
# 3. DATA INTEGRATION
# ============================================================================
print("3. DATA INTEGRATION")
print("-" * 80)

# Inner join airport summary with airlines on Country
integrated_data = pd.merge(airport_summary, airlines_clean, on='Country', how='inner')
print(f"Integrated dataset: {len(integrated_data)} records")
print(f"Columns: {list(integrated_data.columns)}")
print()

# ============================================================================
# 4. SPEARMAN RANK CORRELATION
# ============================================================================
print("4. SPEARMAN RANK CORRELATION")
print("-" * 80)

# Spearman correlation between mean Altitude and airport count
correlation_coef, correlation_pval = spearmanr(
    airport_summary['mean_Altitude'],
    airport_summary['airport_count']
)
print(f"Spearman correlation (mean Altitude vs airport count):")
print(f"  Correlation coefficient: {correlation_coef:.4f}")
print(f"  P-value: {correlation_pval:.6f}")
print()

# ============================================================================
# 5. DBSCAN CLUSTERING
# ============================================================================
print("5. DBSCAN CLUSTERING")
print("-" * 80)

# DBSCAN on Latitude and Longitude (eps=0.5, min_samples=15)
coords = airports_filtered[['Latitude', 'Longitude']].dropna()
dbscan = DBSCAN(eps=0.5, min_samples=15)
clusters = dbscan.fit_predict(coords)

# Count clusters excluding noise (-1)
unique_clusters = set(clusters)
total_clusters = len([c for c in unique_clusters if c != -1])
noise_points = sum(clusters == -1)
print(f"DBSCAN Clustering (eps=0.5, min_samples=15):")
print(f"  Total clusters (excluding noise): {total_clusters}")
print(f"  Noise points: {noise_points}")
print()

# ============================================================================
# 6. RIDGE REGRESSION
# ============================================================================
print("6. RIDGE REGRESSION")
print("-" * 80)

# Ridge regression: Timezone ~ Altitude + Latitude (alpha=10.0)
ridge_data = airports_filtered[['Timezone', 'Altitude', 'Latitude']].dropna()
X_ridge = ridge_data[['Altitude', 'Latitude']].values
y_ridge = ridge_data['Timezone'].values

ridge_model = Ridge(alpha=10.0)
ridge_model.fit(X_ridge, y_ridge)
y_pred_ridge = ridge_model.predict(X_ridge)
r2_ridge = r2_score(y_ridge, y_pred_ridge)

altitude_coef = ridge_model.coef_[0]
latitude_coef = ridge_model.coef_[1]

print(f"Ridge Regression (alpha=10.0): Timezone ~ Altitude + Latitude")
print(f"  R-squared: {r2_ridge:.4f}")
print(f"  Altitude coefficient: {altitude_coef:.5f}")
print(f"  Latitude coefficient: {latitude_coef:.5f}")
print()

# ============================================================================
# 7. KOLMOGOROV-SMIRNOV TEST
# ============================================================================
print("7. KOLMOGOROV-SMIRNOV TEST")
print("-" * 80)

# KS test: Altitude distributions DST=E vs DST=U
altitude_dst_e = airports_filtered[airports_filtered['DST'] == 'E']['Altitude'].dropna()
altitude_dst_u = airports_filtered[airports_filtered['DST'] == 'U']['Altitude'].dropna()

ks_statistic, ks_pvalue = ks_2samp(altitude_dst_e, altitude_dst_u)
print(f"Kolmogorov-Smirnov Test (Altitude: DST=E vs DST=U):")
print(f"  KS statistic: {ks_statistic:.4f}")
print(f"  P-value: {ks_pvalue:.6f}")
print(f"  Sample sizes: DST=E: {len(altitude_dst_e)}, DST=U: {len(altitude_dst_u)}")
print()

# ============================================================================
# 8. BOOTSTRAP CONFIDENCE INTERVAL
# ============================================================================
print("8. BOOTSTRAP CONFIDENCE INTERVAL")
print("-" * 80)

# Bootstrap (10000 resamples) for 90% CI of median Altitude
altitudes = airports_filtered['Altitude'].dropna().values
n_bootstrap = 10000
bootstrap_medians = []

np.random.seed(42)  # For reproducibility
for i in range(n_bootstrap):
    sample = np.random.choice(altitudes, size=len(altitudes), replace=True)
    bootstrap_medians.append(np.median(sample))

# 90% confidence interval
ci_lower = np.percentile(bootstrap_medians, 5)
ci_upper = np.percentile(bootstrap_medians, 95)

print(f"Bootstrap Analysis (10000 resamples):")
print(f"  Median Altitude: {np.median(altitudes):.2f}")
print(f"  90% Confidence Interval: [{ci_lower:.2f}, {ci_upper:.2f}]")
print()

# ============================================================================
# 9. QUANTILE REGRESSION
# ============================================================================
print("9. QUANTILE REGRESSION")
print("-" * 80)

# Quantile regression: 80th percentile of Altitude ~ Longitude
quant_data = airports_filtered[['Altitude', 'Longitude']].dropna()
X_quant = sm.add_constant(quant_data['Longitude'])
y_quant = quant_data['Altitude']

quant_model = QuantReg(y_quant, X_quant)
quant_result = quant_model.fit(q=0.80)

intercept_quant = quant_result.params['const']
slope_quant = quant_result.params['Longitude']

print(f"Quantile Regression (80th percentile): Altitude ~ Longitude")
print(f"  Intercept: {intercept_quant:.3f}")
print(f"  Slope: {slope_quant:.5f}")
print()

# ============================================================================
# 10. GAUSSIAN COPULA (KENDALL TAU)
# ============================================================================
print("10. GAUSSIAN COPULA WITH KENDALL TAU")
print("-" * 80)

# Gaussian copula on standardized Latitude and Longitude
copula_data = airports_filtered[['Latitude', 'Longitude']].dropna()

# Z-score normalization
scaler = StandardScaler()
standardized_coords = scaler.fit_transform(copula_data)

# Calculate Kendall tau correlation
kendall_tau, kendall_pval = stats.kendalltau(
    standardized_coords[:, 0],
    standardized_coords[:, 1]
)

print(f"Gaussian Copula (standardized Latitude and Longitude):")
print(f"  Kendall tau correlation: {kendall_tau:.4f}")
print(f"  P-value: {kendall_pval:.6f}")
print()

# ============================================================================
# 11. PERMUTATION TEST
# ============================================================================
print("11. PERMUTATION TEST")
print("-" * 80)

# Permutation test (8000 permutations): Altitude variance DST=E vs DST=U
observed_var_diff = np.var(altitude_dst_e, ddof=1) - np.var(altitude_dst_u, ddof=1)
combined_altitudes = np.concatenate([altitude_dst_e, altitude_dst_u])
n_e = len(altitude_dst_e)
n_u = len(altitude_dst_u)
n_permutations = 8000

np.random.seed(42)
permuted_diffs = []

for i in range(n_permutations):
    # Shuffle combined data
    permuted = np.random.permutation(combined_altitudes)
    perm_e = permuted[:n_e]
    perm_u = permuted[n_e:]
    perm_diff = np.var(perm_e, ddof=1) - np.var(perm_u, ddof=1)
    permuted_diffs.append(perm_diff)

# Calculate p-value (two-tailed)
permuted_diffs = np.array(permuted_diffs)
p_value_perm = np.mean(np.abs(permuted_diffs) >= np.abs(observed_var_diff))

print(f"Permutation Test (8000 permutations): Altitude variance DST=E vs DST=U")
print(f"  Observed variance difference: {observed_var_diff:.2f}")
print(f"  P-value: {p_value_perm:.6f}")
print()

# ============================================================================
# 12. CONTOUR DENSITY PLOT
# ============================================================================
print("12. CONTOUR DENSITY PLOT")
print("-" * 80)

# 2D KDE contour plot: Latitude (x) vs Longitude (y)
plot_data = airports_filtered[['Latitude', 'Longitude']].dropna()

fig, ax = plt.subplots(figsize=(12, 8))

# Create 2D histogram for KDE estimation
x = plot_data['Latitude'].values
y = plot_data['Longitude'].values

# Kernel Density Estimation using scipy
from scipy.stats import gaussian_kde

# Create KDE
xy = np.vstack([x, y])
kde = gaussian_kde(xy)

# Create grid
x_min, x_max = x.min(), x.max()
y_min, y_max = y.min(), y.max()
xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
positions = np.vstack([xx.ravel(), yy.ravel()])
density = np.reshape(kde(positions).T, xx.shape)

# Contour plot
contour = ax.contourf(xx, yy, density, levels=20, cmap='viridis', alpha=0.8)
ax.contour(xx, yy, density, levels=10, colors='black', alpha=0.3, linewidths=0.5)

plt.colorbar(contour, ax=ax, label='Density')
ax.set_xlabel('Latitude', fontsize=12, fontweight='bold')
ax.set_ylabel('Longitude', fontsize=12, fontweight='bold')
ax.set_title('2D Kernel Density Estimation: Airport Locations\n(Latitude vs Longitude)',
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('airport_density_contour.png', dpi=300, bbox_inches='tight')
print(f"Contour density plot saved as 'airport_density_contour.png'")
print()

# ============================================================================
# SUMMARY REPORT
# ============================================================================
print("="*80)
print("SUMMARY OF KEY FINDINGS")
print("="*80)
print()
print(f"1. Spearman Correlation (mean Altitude vs airport count): {correlation_coef:.4f} (p={correlation_pval:.6f})")
print(f"2. DBSCAN Clusters (excluding noise): {total_clusters}")
print(f"3. Ridge Regression R²: {r2_ridge:.4f}, Altitude coef: {altitude_coef:.5f}")
print(f"4. KS Test Statistic: {ks_statistic:.4f} (p={ks_pvalue:.6f})")
print(f"5. Bootstrap 90% CI for median Altitude: [{ci_lower:.2f}, {ci_upper:.2f}]")
print(f"6. Quantile Regression (Q80): Intercept={intercept_quant:.3f}, Slope={slope_quant:.5f}")
print(f"7. Gaussian Copula Kendall tau: {kendall_tau:.4f}")
print(f"8. Permutation Test p-value: {p_value_perm:.6f}")
print()
print("="*80)
print("ANALYSIS COMPLETE")
print("="*80)
