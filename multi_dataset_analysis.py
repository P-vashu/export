#!/usr/bin/env python3
"""
Multi-Dataset Policy Analysis Script
Performs comprehensive statistical analysis on demographics, communications, and economy data.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import spearmanr
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 80)
print("MULTI-DATASET POLICY ANALYSIS")
print("=" * 80)

# ============================================================================
# STEP 1: DATA LOADING AND MERGING
# ============================================================================
print("\n[1] Loading datasets...")

# Load the three datasets
demographics = pd.read_csv('demographics_data.csv')
communications = pd.read_csv('communications_data.csv')
economy = pd.read_csv('economy_data.csv')

print(f"Demographics: {demographics.shape[0]} rows, {demographics.shape[1]} columns")
print(f"Communications: {communications.shape[0]} rows, {communications.shape[1]} columns")
print(f"Economy: {economy.shape[0]} rows, {economy.shape[1]} columns")

# ============================================================================
# STEP 2: MERGE DATASETS
# ============================================================================
print("\n[2] Merging datasets...")

# First merge: demographics with communications
df = pd.merge(demographics, communications, on='Country', how='inner')
print(f"After merging demographics with communications: {df.shape[0]} rows")

# Second merge: with economy
df = pd.merge(df, economy, on='Country', how='inner')
print(f"After merging with economy: {df.shape[0]} rows")

# ============================================================================
# STEP 3: DATA TYPE CONVERSION
# ============================================================================
print("\n[3] Converting data types to numeric...")

# First, handle percentage columns (they contain % sign)
percentage_str_cols = [
    'Total_Literacy_Rate',
    'Male_Literacy_Rate',
    'Female_Literacy_Rate',
    'Population_Growth_Rate'
]

for col in percentage_str_cols:
    if col in df.columns:
        # Remove % sign and convert to numeric
        df[col] = pd.to_numeric(df[col].astype(str).str.rstrip('%'), errors='coerce')

# Convert other numeric columns, replacing non-numeric values with NaN
other_numeric_cols = [
    'Total_Population',
    'internet_users_total',
    'Real_GDP_per_Capita_USD',
    'Unemployment_Rate_percent',
    'Youth_Unemployment_Rate_percent',
    'broadband_fixed_subscriptions_total',
    'Median_Age',
    'Total_Fertility_Rate',
    'Public_Debt_percent_of_GDP'
]

for col in other_numeric_cols:
    if col in df.columns:
        # Remove commas and convert to numeric
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')

# ============================================================================
# STEP 4: CLEAN MISSING VALUES
# ============================================================================
print("\n[4] Cleaning missing values...")

# Required columns without missing values
required_cols = [
    'Total_Population',
    'internet_users_total',
    'Real_GDP_per_Capita_USD',
    'Unemployment_Rate_percent',
    'Youth_Unemployment_Rate_percent',
    'Total_Literacy_Rate',
    'broadband_fixed_subscriptions_total'
]

# Check initial nulls
print(f"Rows before cleaning: {len(df)}")
for col in required_cols:
    if col in df.columns:
        print(f"  {col}: {df[col].isna().sum()} missing")

# Remove rows with missing values in required columns
df_clean = df.dropna(subset=required_cols)
print(f"Rows after cleaning: {len(df_clean)}")

# ============================================================================
# STEP 5: CONVERT PERCENTAGES TO DECIMALS
# ============================================================================
print("\n[5] Converting percentages to decimals...")

# Convert percentage columns to decimal format (divide by 100)
percentage_cols = [
    'Population_Growth_Rate',
    'Total_Literacy_Rate',
    'Male_Literacy_Rate',
    'Female_Literacy_Rate'
]

for col in percentage_cols:
    if col in df_clean.columns:
        # Already numeric from previous conversion, just divide by 100
        df_clean[col] = df_clean[col] / 100
        print(f"  Converted {col}")

# ============================================================================
# STEP 6: CREATE DIGITAL_PENETRATION_RATE
# ============================================================================
print("\n[6] Creating Digital_Penetration_Rate...")

df_clean['Digital_Penetration_Rate'] = (
    df_clean['internet_users_total'] / df_clean['Total_Population']
)
print(f"  Digital_Penetration_Rate created")
print(f"  Range: {df_clean['Digital_Penetration_Rate'].min():.4f} to {df_clean['Digital_Penetration_Rate'].max():.4f}")

# ============================================================================
# STEP 7: CREATE HIGH_DIGITAL_ACCESS BINARY TREATMENT VARIABLE
# ============================================================================
print("\n[7] Creating High_Digital_Access binary variable...")

# Calculate median
median_dpr = df_clean['Digital_Penetration_Rate'].median()
print(f"  Median Digital_Penetration_Rate: {median_dpr:.6f}")

# Create binary variable: 1 if strictly above median, 0 otherwise
df_clean['High_Digital_Access'] = (
    df_clean['Digital_Penetration_Rate'] > median_dpr
).astype(int)

print(f"  High_Digital_Access distribution:")
print(f"    0 (below or equal to median): {(df_clean['High_Digital_Access'] == 0).sum()}")
print(f"    1 (above median): {(df_clean['High_Digital_Access'] == 1).sum()}")

# ============================================================================
# STEP 8: QUANTILE REGRESSION AT 50TH PERCENTILE
# ============================================================================
print("\n[8] Quantile Regression (50th percentile)...")

# Prepare data for quantile regression
X_quant = df_clean[['Digital_Penetration_Rate']].values
y_quant = df_clean['Real_GDP_per_Capita_USD'].values

# Add constant for intercept
X_quant_const = sm.add_constant(X_quant)

# Fit quantile regression at 50th percentile using interior point method
qr_model = QuantReg(y_quant, X_quant_const)
qr_results = qr_model.fit(q=0.5, method='interior-point')

# Get coefficient
qr_coef = qr_results.params[1]  # Index 1 is Digital_Penetration_Rate (0 is const)
print(f"  Coefficient of Digital_Penetration_Rate at 50th percentile: {qr_coef:.4f}")

# ============================================================================
# STEP 9: SPEARMAN RANK CORRELATION
# ============================================================================
print("\n[9] Spearman Rank Correlation...")

# Perform Spearman correlation (handles ties using average rank method by default)
spearman_corr, spearman_pval = spearmanr(
    df_clean['broadband_fixed_subscriptions_total'],
    df_clean['Youth_Unemployment_Rate_percent']
)

print(f"  Spearman correlation coefficient: {spearman_corr:.4f}")
print(f"  P-value: {spearman_pval:.6f}")

# ============================================================================
# STEP 10: PROPENSITY SCORE MATCHING WITH BOOTSTRAP
# ============================================================================
print("\n[10] Propensity Score Matching with Bootstrap...")

# Prepare data for PSM
psm_data = df_clean[[
    'High_Digital_Access',
    'Total_Literacy_Rate',
    'Unemployment_Rate_percent',
    'Total_Population',
    'Real_GDP_per_Capita_USD'
]].copy()

# Reset index to ensure continuous integers
psm_data = psm_data.reset_index(drop=True)

def estimate_propensity_scores(data):
    """Estimate propensity scores using logistic regression"""
    X = data[['Total_Literacy_Rate', 'Unemployment_Rate_percent', 'Total_Population']]
    y = data['High_Digital_Access']

    # Add constant
    X_const = sm.add_constant(X)

    # Fit logistic regression with IRLS algorithm
    logit_model = sm.Logit(y, X_const)
    # Use convergence tolerance of 1e-8 and max 25 iterations
    logit_result = logit_model.fit(method='newton', maxiter=25, tol=1e-8, disp=False)

    # Get propensity scores
    propensity_scores = logit_result.predict(X_const)

    return propensity_scores

def nearest_neighbor_matching(data_with_ps, caliper=0.1):
    """Perform nearest neighbor matching without replacement"""
    data = data_with_ps.copy()
    data = data.reset_index(drop=True)

    # Separate treated and control units
    treated = data[data['High_Digital_Access'] == 1].copy()
    control = data[data['High_Digital_Access'] == 0].copy()

    treated_indices = treated.index.tolist()
    control_indices = control.index.tolist()

    matched_pairs = []
    used_controls = set()

    # Match treated units in order they appear
    for t_idx in treated_indices:
        t_ps = data.loc[t_idx, 'propensity_score']

        # Find eligible controls within caliper
        eligible_controls = []
        for c_idx in control_indices:
            if c_idx not in used_controls:
                c_ps = data.loc[c_idx, 'propensity_score']
                if abs(t_ps - c_ps) <= caliper:
                    eligible_controls.append((c_idx, abs(t_ps - c_ps)))

        # If eligible controls exist, select the closest one (first if tied)
        if eligible_controls:
            # Sort by distance, then by index to get first appearing
            eligible_controls.sort(key=lambda x: (x[1], x[0]))
            matched_control_idx = eligible_controls[0][0]

            matched_pairs.append((t_idx, matched_control_idx))
            used_controls.add(matched_control_idx)

    return matched_pairs

def calculate_ate(data, matched_pairs):
    """Calculate Average Treatment Effect"""
    if len(matched_pairs) == 0:
        return np.nan

    treatment_effects = []
    for t_idx, c_idx in matched_pairs:
        te = data.loc[t_idx, 'Real_GDP_per_Capita_USD'] - data.loc[c_idx, 'Real_GDP_per_Capita_USD']
        treatment_effects.append(te)

    ate = np.mean(treatment_effects)
    return ate

# Estimate propensity scores on full data
print("  Estimating propensity scores...")
psm_data['propensity_score'] = estimate_propensity_scores(psm_data)

# Perform matching on full data
print("  Performing matching...")
matched_pairs = nearest_neighbor_matching(psm_data, caliper=0.1)
print(f"  Matched pairs: {len(matched_pairs)}")

# Calculate ATE
ate = calculate_ate(psm_data, matched_pairs)
print(f"  ATE: {ate:.2f}")

# Bootstrap for standard error
print("  Performing bootstrap (1000 iterations)...")
n_bootstrap = 1000
bootstrap_ates = []

np.random.seed(42)
for i in range(n_bootstrap):
    # Resample entire dataset with replacement
    bootstrap_sample = psm_data.sample(n=len(psm_data), replace=True)
    bootstrap_sample = bootstrap_sample.reset_index(drop=True)

    # Re-estimate propensity scores
    try:
        bootstrap_sample['propensity_score'] = estimate_propensity_scores(bootstrap_sample)

        # Re-perform matching
        boot_matched_pairs = nearest_neighbor_matching(bootstrap_sample, caliper=0.1)

        # Calculate ATE
        boot_ate = calculate_ate(bootstrap_sample, boot_matched_pairs)

        if not np.isnan(boot_ate):
            bootstrap_ates.append(boot_ate)
    except:
        # Skip this bootstrap iteration if estimation fails
        continue

# Calculate standard error
se_ate = np.std(bootstrap_ates)
print(f"  Standard Error: {se_ate:.3f}")
print(f"  Bootstrap iterations with valid ATE: {len(bootstrap_ates)}/{n_bootstrap}")

# ============================================================================
# STEP 11: MULTIPLE LINEAR REGRESSION WITH HC3
# ============================================================================
print("\n[11] Multiple Linear Regression with HC3...")

# Prepare data - need to drop rows with missing Public_Debt
mlr_data = df_clean[['High_Digital_Access', 'Median_Age', 'Public_Debt_percent_of_GDP']].dropna()
print(f"  Rows with complete data: {len(mlr_data)}")

X_mlr = mlr_data[['High_Digital_Access', 'Median_Age']]
y_mlr = mlr_data['Public_Debt_percent_of_GDP']

# Add constant for intercept
X_mlr_const = sm.add_constant(X_mlr)

# Fit OLS
mlr_model = sm.OLS(y_mlr, X_mlr_const)
mlr_results = mlr_model.fit(cov_type='HC3')

# Get coefficient of High_Digital_Access
hda_coef = mlr_results.params['High_Digital_Access']
print(f"  Coefficient of High_Digital_Access: {hda_coef:.3f}")
print(f"  Standard Error (HC3): {mlr_results.bse['High_Digital_Access']:.3f}")

# ============================================================================
# STEP 12: PRINCIPAL COMPONENT ANALYSIS
# ============================================================================
print("\n[12] Principal Component Analysis...")

# Select variables for PCA
pca_vars = [
    'Total_Literacy_Rate',
    'broadband_fixed_subscriptions_total',
    'Real_GDP_per_Capita_USD',
    'Unemployment_Rate_percent',
    'Youth_Unemployment_Rate_percent',
    'Total_Fertility_Rate',
    'Median_Age'
]

# Extract data
pca_data = df_clean[pca_vars].copy()

# Standardize using z-score normalization
scaler = StandardScaler()
pca_data_scaled = scaler.fit_transform(pca_data)

# Apply PCA
pca = PCA()
pca_transformed = pca.fit_transform(pca_data_scaled)

# Variance explained by first PC
pc1_variance = pca.explained_variance_ratio_[0]
print(f"  Variance explained by PC1: {pc1_variance:.3f}")

# ============================================================================
# STEP 13: DBSCAN CLUSTERING
# ============================================================================
print("\n[13] DBSCAN Clustering...")

# Use first two principal components
pca_2d = pca_transformed[:, :2]

# Apply DBSCAN with epsilon=0.5 and min_samples=5
dbscan = DBSCAN(eps=0.5, min_samples=5, metric='euclidean')
cluster_labels = dbscan.fit_predict(pca_2d)

# Count clusters (excluding noise points with label -1)
unique_labels = set(cluster_labels)
n_clusters = len([label for label in unique_labels if label != -1])
n_noise = list(cluster_labels).count(-1)

print(f"  Number of clusters (excluding noise): {n_clusters}")
print(f"  Number of noise points: {n_noise}")
print(f"  Cluster distribution: {dict(zip(*np.unique(cluster_labels, return_counts=True)))}")

# Calculate Silhouette Score (excluding noise points)
if n_clusters > 0:
    # Filter out noise points
    non_noise_mask = cluster_labels != -1
    if non_noise_mask.sum() > 0:
        pca_2d_non_noise = pca_2d[non_noise_mask]
        labels_non_noise = cluster_labels[non_noise_mask]

        # Calculate silhouette score
        if len(set(labels_non_noise)) > 1:
            silhouette = silhouette_score(pca_2d_non_noise, labels_non_noise, metric='euclidean')
        else:
            # Only one cluster found
            silhouette = silhouette_score(pca_2d_non_noise, labels_non_noise, metric='euclidean')

        print(f"  Silhouette Score: {silhouette:.4f}")
    else:
        print("  Silhouette Score: N/A (all points are noise)")
else:
    print("  Silhouette Score: N/A (no clusters found)")

# ============================================================================
# STEP 14: HEXBIN PLOT
# ============================================================================
print("\n[14] Generating Hexbin Plot...")

fig, ax = plt.subplots(figsize=(10, 8))

hexbin = ax.hexbin(
    df_clean['Real_GDP_per_Capita_USD'],
    df_clean['Youth_Unemployment_Rate_percent'],
    gridsize=20,
    cmap='YlOrRd',
    mincnt=1
)

ax.set_xlabel('Real GDP per Capita (USD)', fontsize=12)
ax.set_ylabel('Youth Unemployment Rate (%)', fontsize=12)
ax.set_title('Hexbin Plot: GDP per Capita vs Youth Unemployment', fontsize=14, fontweight='bold')

# Add colorbar
cbar = plt.colorbar(hexbin, ax=ax)
cbar.set_label('Count', fontsize=12)

plt.tight_layout()
plt.savefig('hexbin_plot.png', dpi=300, bbox_inches='tight')
print("  Hexbin plot saved as 'hexbin_plot.png'")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("ANALYSIS SUMMARY")
print("=" * 80)
print(f"\n1. Data Merging & Cleaning:")
print(f"   - Final dataset: {len(df_clean)} countries")
print(f"\n2. Quantile Regression (50th percentile):")
print(f"   - Coefficient: {qr_coef:.4f}")
print(f"\n3. Spearman Rank Correlation:")
print(f"   - Correlation: {spearman_corr:.4f}")
print(f"   - P-value: {spearman_pval:.6f}")
print(f"\n4. Propensity Score Matching:")
print(f"   - ATE: {ate:.2f}")
print(f"   - Standard Error: {se_ate:.3f}")
print(f"\n5. Multiple Linear Regression (HC3):")
print(f"   - Coefficient: {hda_coef:.3f}")
print(f"\n6. Principal Component Analysis:")
print(f"   - PC1 Variance: {pc1_variance:.3f}")
print(f"\n7. DBSCAN Clustering:")
print(f"   - Number of Clusters: {n_clusters}")
if n_clusters > 0 and non_noise_mask.sum() > 0:
    print(f"   - Silhouette Score: {silhouette:.4f}")
else:
    print(f"   - Silhouette Score: N/A")
print("\n" + "=" * 80)

print("\nAnalysis complete!")
