#!/usr/bin/env python3
"""
Comprehensive Vaccination Drive Analysis
Author: Data Science Team
Date: 2025-11-07
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import kstest, ks_2samp
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import STL
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("VACCINATION DRIVE ANALYSIS - COMPREHENSIVE REPORT")
print("=" * 80)
print()

# ============================================================================
# STEP 1: DATA LOADING AND MERGING
# ============================================================================
print("STEP 1: Loading and Merging Datasets")
print("-" * 80)

# Load datasets
state_df = pd.read_csv('state.csv')
vaccination_df = pd.read_csv('vaccination.csv')
time_df = pd.read_csv('time.csv')

print(f"State dataset shape: {state_df.shape}")
print(f"Vaccination dataset shape: {vaccination_df.shape}")
print(f"Time dataset shape: {time_df.shape}")
print()

# Merge state.csv with vaccination.csv on 'date' (inner join)
merged_state_vac = pd.merge(state_df, vaccination_df, on='date', how='inner')
print(f"After first merge (state + vaccination): {merged_state_vac.shape}")

# Merge with time.csv on 'date' (inner join)
final_dataset = pd.merge(merged_state_vac, time_df, on='date', how='inner')
print(f"After second merge (+ time): {final_dataset.shape}")
print()

# ============================================================================
# STEP 2: DATA CLEANING
# ============================================================================
print("STEP 2: Data Cleaning")
print("-" * 80)

print(f"Dataset shape before cleaning: {final_dataset.shape}")

# Remove rows with missing values in specified columns
columns_to_check = ['Total-Vaccinated', 'tot_dose_1', 'count', 'dose_one']
final_dataset = final_dataset.dropna(subset=columns_to_check)

print(f"Dataset shape after removing missing values: {final_dataset.shape}")
print()

# ============================================================================
# STEP 3: EXTRACT START_HOUR FROM TIME_SLOT
# ============================================================================
print("STEP 3: Extracting Start Hour from Time Slot")
print("-" * 80)

# Extract starting hour from time_slot (format: "HH:MM-HH:MM")
final_dataset['start_hour'] = final_dataset['time_slot'].str.split(':').str[0].astype(int)

print(f"Sample time_slot and start_hour values:")
print(final_dataset[['time_slot', 'start_hour']].head(10))
print()

# ============================================================================
# ANALYSIS 1: GAUSSIAN MIXTURE MODEL
# ============================================================================
print("=" * 80)
print("ANALYSIS 1: Gaussian Mixture Model (GMM)")
print("=" * 80)

# Prepare data for GMM
gmm_data = final_dataset[['Partial-Vaccinated', 'tot_dose_2']].dropna()
print(f"GMM data shape: {gmm_data.shape}")

# Fit Gaussian Mixture Model
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
gmm.fit(gmm_data)

# Calculate BIC and AIC
bic = gmm.bic(gmm_data)
aic = gmm.aic(gmm_data)

print(f"\nGaussian Mixture Model Results:")
print(f"  Number of components: 3")
print(f"  Covariance type: full")
print(f"  Random state: 42")
print(f"  Bayesian Information Criterion (BIC): {bic:.2f}")
print(f"  Akaike Information Criterion (AIC): {aic:.2f}")
print()

# ============================================================================
# ANALYSIS 2: QUANTILE REGRESSION (UTTAR PRADESH, START_HOUR=12)
# ============================================================================
print("=" * 80)
print("ANALYSIS 2: Quantile Regression (0.75 Quantile)")
print("=" * 80)

# Filter for Uttar Pradesh and start_hour == 12
up_data = final_dataset[(final_dataset['State'] == 'Uttar Pradesh') &
                        (final_dataset['start_hour'] == 12)].copy()

print(f"Uttar Pradesh data (start_hour=12) shape: {up_data.shape}")

if len(up_data) > 0:
    # Prepare data for quantile regression
    X = up_data[['Partial-Vaccinated']].copy()
    X['const'] = 1  # Add constant term
    y = up_data['tot_dose_1']

    # Perform quantile regression at 0.75 quantile
    qr_model = QuantReg(y, X[['const', 'Partial-Vaccinated']])
    qr_results = qr_model.fit(q=0.75)

    intercept = qr_results.params['const']
    slope = qr_results.params['Partial-Vaccinated']

    print(f"\nQuantile Regression Results (q=0.75):")
    print(f"  Filter: State='Uttar Pradesh', start_hour=12")
    print(f"  Model: tot_dose_1 ~ Partial-Vaccinated")
    print(f"  Slope coefficient (0.75 quantile): {slope:.5f}")
    print(f"  Intercept coefficient (0.75 quantile): {intercept:.2f}")
else:
    print("WARNING: No data available for Uttar Pradesh with start_hour=12")
print()

# ============================================================================
# ANALYSIS 3: KOLMOGOROV-SMIRNOV TEST
# ============================================================================
print("=" * 80)
print("ANALYSIS 3: Kolmogorov-Smirnov Test")
print("=" * 80)

# Split data based on start_hour
dose_one_before_12 = final_dataset[final_dataset['start_hour'] < 12]['dose_one'].dropna()
dose_one_after_12 = final_dataset[final_dataset['start_hour'] >= 12]['dose_one'].dropna()

print(f"Records with start_hour < 12: {len(dose_one_before_12)}")
print(f"Records with start_hour >= 12: {len(dose_one_after_12)}")

# Perform Kolmogorov-Smirnov test
ks_statistic, ks_pvalue = ks_2samp(dose_one_before_12, dose_one_after_12)

print(f"\nKolmogorov-Smirnov Test Results:")
print(f"  Comparing: dose_one distribution (start_hour < 12 vs >= 12)")
print(f"  KS test statistic: {ks_statistic:.4f}")
print(f"  P-value: {ks_pvalue:.6f}")
print()

# ============================================================================
# ANALYSIS 4: STL DECOMPOSITION
# ============================================================================
print("=" * 80)
print("ANALYSIS 4: STL Seasonal Decomposition")
print("=" * 80)

# Aggregate by date and sum count
time_series_data = final_dataset.groupby('date')['count'].sum().reset_index()
time_series_data = time_series_data.sort_values('date')

# Convert date to datetime
time_series_data['date'] = pd.to_datetime(time_series_data['date'])
time_series_data.set_index('date', inplace=True)

print(f"Time series data shape: {time_series_data.shape}")
print(f"Date range: {time_series_data.index.min()} to {time_series_data.index.max()}")

# Ensure we have enough data points for seasonal decomposition (need at least 2 periods = 14 days)
if len(time_series_data) >= 14:
    # Perform STL decomposition
    # Create a series with proper frequency
    ts_series = time_series_data['count']

    stl = STL(ts_series, seasonal=7, robust=True, period=7)
    stl_result = stl.fit()

    # Calculate variance of seasonal and residual components
    seasonal_variance = np.var(stl_result.seasonal)
    residual_variance = np.var(stl_result.resid)

    print(f"\nSTL Decomposition Results:")
    print(f"  Seasonal parameter: 7")
    print(f"  Robust: True")
    print(f"  Variance of seasonal component: {seasonal_variance:.2f}")
    print(f"  Variance of residual component: {residual_variance:.2f}")
else:
    print(f"WARNING: Insufficient data for STL decomposition (need >= 14, have {len(time_series_data)})")
print()

# ============================================================================
# ANALYSIS 5: AUGMENTED DICKEY-FULLER TEST (DELHI)
# ============================================================================
print("=" * 80)
print("ANALYSIS 5: Augmented Dickey-Fuller Test (Delhi)")
print("=" * 80)

# Aggregate by State and date
agg_data = final_dataset.groupby(['State', 'date']).agg({
    'tot_dose_1': 'sum',
    'count': 'sum'
}).reset_index()

# Calculate daily_efficiency
agg_data['daily_efficiency'] = agg_data['tot_dose_1'] / agg_data['count']

# Filter for Delhi
delhi_data = agg_data[agg_data['State'] == 'Delhi'].copy()
delhi_data = delhi_data.sort_values('date')

print(f"Delhi data shape: {delhi_data.shape}")

if len(delhi_data) > 3:  # Need at least 4 observations for ADF test
    # Perform ADF test
    adf_result = adfuller(delhi_data['daily_efficiency'].dropna())

    adf_statistic = adf_result[0]
    adf_pvalue = adf_result[1]

    print(f"\nAugmented Dickey-Fuller Test Results:")
    print(f"  State: Delhi")
    print(f"  Variable: daily_efficiency (tot_dose_1 / count)")
    print(f"  ADF test statistic: {adf_statistic:.3f}")
    print(f"  P-value: {adf_pvalue:.6f}")
    print(f"  Number of observations: {len(delhi_data)}")
else:
    print(f"WARNING: Insufficient data for ADF test (need > 3, have {len(delhi_data)})")
print()

# ============================================================================
# ANALYSIS 6: HEXBIN PLOT
# ============================================================================
print("=" * 80)
print("ANALYSIS 6: Generating Hexbin Plot")
print("=" * 80)

plt.figure(figsize=(12, 8))

# Create hexbin plot
hexbin = plt.hexbin(
    final_dataset['tot_dose_1'],
    final_dataset['count'],
    gridsize=30,
    cmap='YlOrRd',
    bins='log',  # Logarithmic color scale
    mincnt=1
)

plt.xlim(0, 250000000)
plt.ylim(0, 2000000)
plt.xlabel('Total Dose 1 (tot_dose_1)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Hexbin Plot: Total Dose 1 vs Count (Log Scale)', fontsize=14, fontweight='bold')
plt.colorbar(hexbin, label='Log10(Count of Points)')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save plot
plt.savefig('hexbin_plot.png', dpi=300, bbox_inches='tight')
print("Hexbin plot saved as 'hexbin_plot.png'")
print(f"  X-axis: tot_dose_1, range [0, 250000000]")
print(f"  Y-axis: count, range [0, 2000000]")
print(f"  Gridsize: 30")
print(f"  Color scale: Logarithmic")
print()

# ============================================================================
# SUMMARY OF ALL RESULTS
# ============================================================================
print("=" * 80)
print("SUMMARY OF ALL RESULTS")
print("=" * 80)
print()

print("1. GAUSSIAN MIXTURE MODEL:")
print(f"   BIC: {bic:.2f}")
print(f"   AIC: {aic:.2f}")
print()

print("2. QUANTILE REGRESSION (Uttar Pradesh, start_hour=12):")
if len(up_data) > 0:
    print(f"   Slope (0.75 quantile): {slope:.5f}")
    print(f"   Intercept (0.75 quantile): {intercept:.2f}")
else:
    print("   No data available")
print()

print("3. KOLMOGOROV-SMIRNOV TEST:")
print(f"   KS Statistic: {ks_statistic:.4f}")
print(f"   P-value: {ks_pvalue:.6f}")
print()

print("4. STL DECOMPOSITION:")
if len(time_series_data) >= 14:
    print(f"   Seasonal Variance: {seasonal_variance:.2f}")
    print(f"   Residual Variance: {residual_variance:.2f}")
else:
    print("   Insufficient data")
print()

print("5. AUGMENTED DICKEY-FULLER TEST (Delhi):")
if len(delhi_data) > 3:
    print(f"   ADF Statistic: {adf_statistic:.3f}")
    print(f"   P-value: {adf_pvalue:.6f}")
else:
    print("   Insufficient data")
print()

print("6. HEXBIN PLOT:")
print("   Generated and saved as 'hexbin_plot.png'")
print()

print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
