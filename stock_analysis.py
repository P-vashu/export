"""
Advanced Statistical Analysis of Stock Price Data
Analyzing GS, HD, and IBM historical stock prices with volatility patterns,
risk-adjusted returns, and inter-stock dependencies.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from scipy import stats
from scipy.stats import genpareto, norm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("="*80)
print("ADVANCED STOCK ANALYSIS: GS, HD, IBM")
print("="*80)

# ============================================================================
# 1. DATA PREPARATION
# ============================================================================
print("\n[1] LOADING AND PREPARING DATA")
print("-"*80)

# Load datasets
gs_df = pd.read_csv('GS_2006-01-01_to_2018-01-01.csv')
hd_df = pd.read_csv('HD_2006-01-01_to_2018-01-01.csv')
ibm_df = pd.read_csv('IBM_2006-01-01_to_2018-01-01.csv')

# Convert Date columns to datetime
gs_df['Date'] = pd.to_datetime(gs_df['Date'])
hd_df['Date'] = pd.to_datetime(hd_df['Date'])
ibm_df['Date'] = pd.to_datetime(ibm_df['Date'])

print(f"Original data loaded:")
print(f"  GS:  {len(gs_df)} records")
print(f"  HD:  {len(hd_df)} records")
print(f"  IBM: {len(ibm_df)} records")

# Filter datasets to specific date ranges
gs_filtered = gs_df[(gs_df['Date'] >= '2010-01-01') & (gs_df['Date'] <= '2015-12-31')].copy()
hd_filtered = hd_df[(hd_df['Date'] >= '2008-01-01') & (hd_df['Date'] <= '2014-12-31')].copy()
ibm_filtered = ibm_df[(ibm_df['Date'] >= '2009-01-01') & (ibm_df['Date'] <= '2016-12-31')].copy()

print(f"\nFiltered data:")
print(f"  GS  (2010-01-01 to 2015-12-31): {len(gs_filtered)} records")
print(f"  HD  (2008-01-01 to 2014-12-31): {len(hd_filtered)} records")
print(f"  IBM (2009-01-01 to 2016-12-31): {len(ibm_filtered)} records")

# Prepare datasets for merging
gs_merge = gs_filtered[['Date', 'Close', 'Volume']].rename(
    columns={'Close': 'GS_Close', 'Volume': 'GS_Volume'}
)
hd_merge = hd_filtered[['Date', 'Close', 'Volume']].rename(
    columns={'Close': 'HD_Close', 'Volume': 'HD_Volume'}
)
ibm_merge = ibm_filtered[['Date', 'Close', 'Volume']].rename(
    columns={'Close': 'IBM_Close', 'Volume': 'IBM_Volume'}
)

# Inner join on Date
merged_df = gs_merge.merge(hd_merge, on='Date', how='inner')
merged_df = merged_df.merge(ibm_merge, on='Date', how='inner')
merged_df = merged_df.sort_values('Date').reset_index(drop=True)

print(f"\nMerged dataset (inner join): {len(merged_df)} records")
print(f"Date range: {merged_df['Date'].min().strftime('%Y-%m-%d')} to {merged_df['Date'].max().strftime('%Y-%m-%d')}")

# ============================================================================
# 2. LOG RETURNS CALCULATION
# ============================================================================
print("\n[2] CALCULATING LOG RETURNS")
print("-"*80)

# Calculate log returns: ln(P_t / P_{t-1})
merged_df['GS_LogReturn'] = np.log(merged_df['GS_Close'] / merged_df['GS_Close'].shift(1))
merged_df['HD_LogReturn'] = np.log(merged_df['HD_Close'] / merged_df['HD_Close'].shift(1))
merged_df['IBM_LogReturn'] = np.log(merged_df['IBM_Close'] / merged_df['IBM_Close'].shift(1))

# Remove first row with NaN log returns
merged_df = merged_df.dropna().reset_index(drop=True)

print(f"Log returns calculated for {len(merged_df)} trading days")
print(f"\nLog Returns Summary Statistics:")
print(f"  GS  - Mean: {merged_df['GS_LogReturn'].mean():.6f}, Std: {merged_df['GS_LogReturn'].std():.6f}")
print(f"  HD  - Mean: {merged_df['HD_LogReturn'].mean():.6f}, Std: {merged_df['HD_LogReturn'].std():.6f}")
print(f"  IBM - Mean: {merged_df['IBM_LogReturn'].mean():.6f}, Std: {merged_df['IBM_LogReturn'].std():.6f}")

# ============================================================================
# 3. TIME-SERIES DECOMPOSITION (STL) FOR GOLDMAN SACHS
# ============================================================================
print("\n[3] STL DECOMPOSITION ON GOLDMAN SACHS CLOSE PRICES")
print("-"*80)

# Apply STL decomposition with seasonal period of 63 trading days
# STL requires explicit period specification
gs_close_values = merged_df['GS_Close'].values

stl = STL(gs_close_values, seasonal=63, period=63, robust=True)
result = stl.fit()

trend_component = result.trend
residual_component = result.resid

# Find date with maximum trend value (map back to dates)
max_trend_idx = np.argmax(trend_component)
max_trend_date = merged_df.loc[max_trend_idx, 'Date']
max_trend_value = trend_component[max_trend_idx]

print(f"STL Decomposition completed (seasonal period = 63)")
print(f"Date with maximum trend value: {max_trend_date.strftime('%Y-%m-%d')}")
print(f"Maximum trend value: ${max_trend_value:.2f}")

# ============================================================================
# 4. GAUSSIAN COPULA FOR HD AND IBM LOG RETURNS
# ============================================================================
print("\n[4] GAUSSIAN COPULA FOR HD AND IBM LOG RETURNS")
print("-"*80)

# Extract HD and IBM log returns
hd_returns = merged_df['HD_LogReturn'].values
ibm_returns = merged_df['IBM_LogReturn'].values

# Apply probability integral transform (empirical CDF)
def empirical_cdf_transform(data):
    """Transform data using empirical CDF (probability integral transform)"""
    n = len(data)
    ranks = stats.rankdata(data, method='average')
    # Use (rank - 0.5) / n to avoid 0 and 1
    uniform = (ranks - 0.5) / n
    return uniform

# Transform to uniform margins
u_hd = empirical_cdf_transform(hd_returns)
u_ibm = empirical_cdf_transform(ibm_returns)

# Transform to standard normal (inverse CDF)
z_hd = norm.ppf(u_hd)
z_ibm = norm.ppf(u_ibm)

# Estimate Gaussian copula correlation using Pearson correlation of transformed data
copula_correlation = np.corrcoef(z_hd, z_ibm)[0, 1]

print(f"Gaussian Copula Correlation Parameter: {copula_correlation:.3f}")

# ============================================================================
# 5. PRINCIPAL COMPONENT ANALYSIS (PCA)
# ============================================================================
print("\n[5] PRINCIPAL COMPONENT ANALYSIS ON LOG RETURNS")
print("-"*80)

# Prepare log returns matrix
returns_matrix = merged_df[['GS_LogReturn', 'HD_LogReturn', 'IBM_LogReturn']].values

# Standardize the returns (z-score)
scaler = StandardScaler()
returns_standardized = scaler.fit_transform(returns_matrix)

# Apply PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(returns_standardized)

# Extract metrics
variance_pc1 = pca.explained_variance_ratio_[0]
cumulative_variance_pc1_pc2 = pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1]
gs_loading_pc1 = pca.components_[0, 0]  # First component, first feature (GS)

print(f"Variance explained by PC1: {variance_pc1:.3f}")
print(f"Cumulative variance explained by PC1+PC2: {cumulative_variance_pc1_pc2:.3f}")
print(f"GS loading on PC1: {gs_loading_pc1:.3f}")
print(f"\nAll loadings on PC1:")
print(f"  GS:  {pca.components_[0, 0]:.3f}")
print(f"  HD:  {pca.components_[0, 1]:.3f}")
print(f"  IBM: {pca.components_[0, 2]:.3f}")

# ============================================================================
# 6. EXTREME VALUE THEORY (EVT) WITH GENERALIZED PARETO DISTRIBUTION
# ============================================================================
print("\n[6] EXTREME VALUE THEORY FOR IBM LOG RETURNS")
print("-"*80)

# Extract IBM log returns
ibm_log_returns = merged_df['IBM_LogReturn'].values

# Find 5th percentile threshold
threshold = np.percentile(ibm_log_returns, 5)
print(f"5th percentile threshold: {threshold:.6f}")

# Extract exceedances below threshold
below_threshold = ibm_log_returns[ibm_log_returns < threshold]
# Transform to positive exceedances
exceedances = np.abs(below_threshold - threshold)

print(f"Number of exceedances: {len(exceedances)}")

# Fit Generalized Pareto Distribution using MLE
shape, loc, scale = genpareto.fit(exceedances, floc=0)

print(f"GPD Shape Parameter (ξ): {shape:.3f}")
print(f"GPD Scale Parameter (σ): {scale:.3f}")

# ============================================================================
# 7. HIDDEN MARKOV MODEL (2-STATE GAUSSIAN) FOR GOLDMAN SACHS
# ============================================================================
print("\n[7] HIDDEN MARKOV MODEL FOR GOLDMAN SACHS LOG RETURNS")
print("-"*80)

# Prepare data for HMM (needs to be 2D array)
gs_returns = merged_df['GS_LogReturn'].values.reshape(-1, 1)

# Initialize and fit HMM
model = hmm.GaussianHMM(
    n_components=2,
    covariance_type="full",
    n_iter=100,
    tol=0.0001,
    random_state=42,
    init_params='stmc',
    params='stmc'
)

# Set initialization method to k-means
model.fit(gs_returns)

# Extract means and identify states
means = model.means_.flatten()
state_0_idx = np.argmin(means)  # State 0 has lower mean
state_1_idx = np.argmax(means)  # State 1 has higher mean

print(f"State 0 (lower mean): μ = {means[state_0_idx]:.6f}")
print(f"State 1 (higher mean): μ = {means[state_1_idx]:.6f}")

# Extract transition matrix
transmat = model.transmat_

# Get transition probabilities based on identified states
prob_0_to_1 = transmat[state_0_idx, state_1_idx]
prob_1_to_1 = transmat[state_1_idx, state_1_idx]

print(f"\nTransition Probabilities:")
print(f"  P(State 0 → State 1): {prob_0_to_1:.3f}")
print(f"  P(State 1 → State 1): {prob_1_to_1:.3f}")
print(f"\nFull Transition Matrix:")
print(f"  [[{transmat[0,0]:.3f}, {transmat[0,1]:.3f}],")
print(f"   [{transmat[1,0]:.3f}, {transmat[1,1]:.3f}]]")

# ============================================================================
# 8. VISUALIZATION: HEXBIN PLOT (HD vs IBM LOG RETURNS)
# ============================================================================
print("\n[8] CREATING HEXBIN VISUALIZATION")
print("-"*80)

fig, ax = plt.subplots(figsize=(10, 8))

hexbin = ax.hexbin(
    merged_df['HD_LogReturn'],
    merged_df['IBM_LogReturn'],
    gridsize=30,
    cmap='YlOrRd',
    mincnt=1
)

ax.set_xlabel('Home Depot Daily Log Returns', fontsize=12, fontweight='bold')
ax.set_ylabel('IBM Daily Log Returns', fontsize=12, fontweight='bold')
ax.set_title('Joint Distribution: HD vs IBM Log Returns\n(Hexbin Plot, gridsize=30)',
             fontsize=14, fontweight='bold', pad=20)

# Add colorbar
cbar = plt.colorbar(hexbin, ax=ax)
cbar.set_label('Count', fontsize=11, fontweight='bold')

# Add grid
ax.grid(True, alpha=0.3, linestyle='--')

# Add zero lines
ax.axhline(y=0, color='black', linewidth=0.8, alpha=0.5)
ax.axvline(x=0, color='black', linewidth=0.8, alpha=0.5)

plt.tight_layout()
plt.savefig('hexbin_hd_ibm_log_returns.png', dpi=300, bbox_inches='tight')
print("Hexbin plot saved: hexbin_hd_ibm_log_returns.png")

# ============================================================================
# SUMMARY REPORT
# ============================================================================
print("\n" + "="*80)
print("FINAL RESULTS SUMMARY")
print("="*80)

print(f"""
1. STL DECOMPOSITION (Goldman Sachs):
   Date with Maximum Trend: {max_trend_date.strftime('%Y-%m-%d')}

2. GAUSSIAN COPULA (HD-IBM):
   Correlation Parameter: {copula_correlation:.3f}

3. PRINCIPAL COMPONENT ANALYSIS:
   Variance Explained by PC1: {variance_pc1:.3f}
   Cumulative Variance (PC1+PC2): {cumulative_variance_pc1_pc2:.3f}
   GS Loading on PC1: {gs_loading_pc1:.3f}

4. EXTREME VALUE THEORY (IBM):
   GPD Shape Parameter: {shape:.3f}
   GPD Scale Parameter: {scale:.3f}

5. HIDDEN MARKOV MODEL (Goldman Sachs):
   P(State 0 → State 1): {prob_0_to_1:.3f}
   P(State 1 → State 1): {prob_1_to_1:.3f}

6. VISUALIZATION:
   Hexbin plot created: hexbin_hd_ibm_log_returns.png
""")

print("="*80)
print("ANALYSIS COMPLETE")
print("="*80)

# Save detailed results to CSV
results_summary = pd.DataFrame({
    'Metric': [
        'STL_MaxTrend_Date',
        'Copula_Correlation',
        'PCA_PC1_Variance',
        'PCA_PC1_PC2_Cumulative',
        'PCA_GS_Loading_PC1',
        'EVT_GPD_Shape',
        'EVT_GPD_Scale',
        'HMM_P_State0_to_State1',
        'HMM_P_State1_to_State1'
    ],
    'Value': [
        max_trend_date.strftime('%Y-%m-%d'),
        f'{copula_correlation:.3f}',
        f'{variance_pc1:.3f}',
        f'{cumulative_variance_pc1_pc2:.3f}',
        f'{gs_loading_pc1:.3f}',
        f'{shape:.3f}',
        f'{scale:.3f}',
        f'{prob_0_to_1:.3f}',
        f'{prob_1_to_1:.3f}'
    ]
})

results_summary.to_csv('analysis_results_summary.csv', index=False)
print("\nDetailed results saved to: analysis_results_summary.csv")
