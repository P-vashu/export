import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import STL
from hmmlearn import hmm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. DATA PREPARATION
# ============================================================================
print("="*80)
print("STOCK ANALYSIS: UTX, UNH, TRV")
print("="*80)

# Load datasets
utx = pd.read_csv('UTX_2006-01-01_to_2018-01-01.csv')
unh = pd.read_csv('UNH_2006-01-01_to_2018-01-01.csv')
trv = pd.read_csv('TRV_2006-01-01_to_2018-01-01.csv')

# Convert Date column to datetime format
utx['Date'] = pd.to_datetime(utx['Date'])
unh['Date'] = pd.to_datetime(unh['Date'])
trv['Date'] = pd.to_datetime(trv['Date'])

# Filter datasets to specified date ranges
utx_filtered = utx[(utx['Date'] >= '2010-01-01') & (utx['Date'] <= '2015-12-31')].copy()
unh_filtered = unh[(unh['Date'] >= '2008-01-01') & (unh['Date'] <= '2014-12-31')].copy()
trv_filtered = trv[(trv['Date'] >= '2009-01-01') & (trv['Date'] <= '2016-12-31')].copy()

print(f"\nFiltered data ranges:")
print(f"UTX: {utx_filtered['Date'].min()} to {utx_filtered['Date'].max()} ({len(utx_filtered)} records)")
print(f"UNH: {unh_filtered['Date'].min()} to {unh_filtered['Date'].max()} ({len(unh_filtered)} records)")
print(f"TRV: {trv_filtered['Date'].min()} to {trv_filtered['Date'].max()} ({len(trv_filtered)} records)")

# Prepare for merging - select relevant columns
utx_filtered = utx_filtered[['Date', 'Close', 'Volume']].rename(
    columns={'Close': 'UTX_Close', 'Volume': 'UTX_Volume'}
)
unh_filtered = unh_filtered[['Date', 'Close', 'Volume']].rename(
    columns={'Close': 'UNH_Close', 'Volume': 'UNH_Volume'}
)
trv_filtered = trv_filtered[['Date', 'Close', 'Volume']].rename(
    columns={'Close': 'TRV_Close', 'Volume': 'TRV_Volume'}
)

# Inner join on Date
merged = utx_filtered.merge(unh_filtered, on='Date', how='inner')
merged = merged.merge(trv_filtered, on='Date', how='inner')

print(f"\nMerged dataset: {len(merged)} records")
print(f"Date range: {merged['Date'].min()} to {merged['Date'].max()}")

# ============================================================================
# 2. LOG RETURNS CALCULATION
# ============================================================================
print("\n" + "="*80)
print("LOG RETURNS CALCULATION")
print("="*80)

# Calculate log returns
merged['UTX_LogReturn'] = np.log(merged['UTX_Close'] / merged['UTX_Close'].shift(1))
merged['UNH_LogReturn'] = np.log(merged['UNH_Close'] / merged['UNH_Close'].shift(1))
merged['TRV_LogReturn'] = np.log(merged['TRV_Close'] / merged['TRV_Close'].shift(1))

# Drop first row with NaN values
merged = merged.dropna().reset_index(drop=True)

print(f"\nLog returns calculated. Dataset after dropping NaN: {len(merged)} records")
print(f"\nLog Returns Summary:")
print(merged[['UTX_LogReturn', 'UNH_LogReturn', 'TRV_LogReturn']].describe())

# ============================================================================
# 3. STL DECOMPOSITION ON UTX CLOSE PRICES
# ============================================================================
print("\n" + "="*80)
print("STL DECOMPOSITION (UTX Close Prices)")
print("="*80)

# Create a time series for STL decomposition
# Note: We need the merged dataset with Date as index
utx_close_series = merged.set_index('Date')['UTX_Close']

# Apply STL decomposition with period = 63
stl = STL(utx_close_series, period=63)
result = stl.fit()

# Extract trend and residual components
trend = result.trend
residual = result.resid

# Find date with maximum trend value
max_trend_date = trend.idxmax()
max_trend_value = trend.max()

print(f"\nSTL Decomposition Results:")
print(f"Date with maximum trend: {max_trend_date.strftime('%Y-%m-%d')}")
print(f"Maximum trend value: {max_trend_value:.2f}")

# ============================================================================
# 4. GAUSSIAN COPULA (UNH-TRV DEPENDENCE)
# ============================================================================
print("\n" + "="*80)
print("GAUSSIAN COPULA (UNH-TRV Log Returns)")
print("="*80)

# Extract log returns for UNH and TRV
unh_returns = merged['UNH_LogReturn'].values
trv_returns = merged['TRV_LogReturn'].values

# Apply probability integral transform using empirical marginals
# Convert to uniform marginals using empirical CDF (rank-based)
n = len(unh_returns)
unh_ranks = stats.rankdata(unh_returns)
trv_ranks = stats.rankdata(trv_returns)

# Convert ranks to uniform [0,1] - using (rank-0.5)/n to avoid boundary issues
unh_uniform = (unh_ranks - 0.5) / n
trv_uniform = (trv_ranks - 0.5) / n

# Transform to standard normal using inverse normal CDF
unh_normal = stats.norm.ppf(unh_uniform)
trv_normal = stats.norm.ppf(trv_uniform)

# Estimate Gaussian copula correlation parameter
copula_corr = np.corrcoef(unh_normal, trv_normal)[0, 1]

print(f"\nGaussian Copula Correlation Parameter: {copula_corr:.3f}")

# ============================================================================
# 5. PRINCIPAL COMPONENT ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("PRINCIPAL COMPONENT ANALYSIS")
print("="*80)

# Create log returns matrix
log_returns_matrix = merged[['UTX_LogReturn', 'UNH_LogReturn', 'TRV_LogReturn']].values

# Standardize the data (mean=0, std=1)
scaler = StandardScaler()
standardized_returns = scaler.fit_transform(log_returns_matrix)

# Apply PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(standardized_returns)

# Extract variance explained
variance_pc1 = pca.explained_variance_ratio_[0]
cumulative_variance_pc1_pc2 = pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1]

# Extract loadings (components)
# First principal component loadings
pc1_loadings = pca.components_[0]
utx_loading_pc1 = pc1_loadings[0]  # UTX is first column

print(f"\nPCA Results:")
print(f"Variance explained by PC1: {variance_pc1:.3f}")
print(f"Cumulative variance (PC1+PC2): {cumulative_variance_pc1_pc2:.3f}")
print(f"UTX loading on PC1: {utx_loading_pc1:.3f}")
print(f"\nAll PC1 loadings: UTX={pc1_loadings[0]:.3f}, UNH={pc1_loadings[1]:.3f}, TRV={pc1_loadings[2]:.3f}")

# ============================================================================
# 6. EXTREME VALUE THEORY (TRV Log Returns)
# ============================================================================
print("\n" + "="*80)
print("EXTREME VALUE THEORY (Generalized Pareto Distribution)")
print("="*80)

# Extract TRV log returns
trv_log_returns = merged['TRV_LogReturn'].values

# Calculate 5th percentile threshold
threshold = np.percentile(trv_log_returns, 5)
print(f"\n5th percentile threshold: {threshold:.6f}")

# Identify exceedances below threshold
exceedances_mask = trv_log_returns < threshold
exceedances = trv_log_returns[exceedances_mask]

# Transform to positive exceedances
positive_exceedances = np.abs(exceedances - threshold)

print(f"Number of exceedances: {len(positive_exceedances)}")
print(f"Mean exceedance: {positive_exceedances.mean():.6f}")

# Fit Generalized Pareto Distribution using MLE
gpd_params = stats.genpareto.fit(positive_exceedances, floc=0)
shape_param = gpd_params[0]
loc_param = gpd_params[1]
scale_param = gpd_params[2]

print(f"\nGPD Parameters (MLE):")
print(f"Shape parameter: {shape_param:.3f}")
print(f"Scale parameter: {scale_param:.3f}")
print(f"Location parameter (fixed): {loc_param:.3f}")

# ============================================================================
# 7. HIDDEN MARKOV MODEL (UTX Log Returns)
# ============================================================================
print("\n" + "="*80)
print("HIDDEN MARKOV MODEL (2 States, Gaussian Emissions)")
print("="*80)

# Prepare data for HMM
utx_returns_hmm = merged['UTX_LogReturn'].values.reshape(-1, 1)

# Initialize and fit Gaussian HMM
model = hmm.GaussianHMM(
    n_components=2,
    covariance_type='full',
    n_iter=100,
    tol=0.0001,
    random_state=42,
    init_params='kmeans',
    params='stmc',
    implementation='log'
)

# Set k-means initialization parameters explicitly
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10, init='k-means++')
labels = kmeans.fit_predict(utx_returns_hmm)

# Compute initial means and covariances from k-means
means_init = np.array([utx_returns_hmm[labels == i].mean() for i in range(2)]).reshape(2, 1)
covars_init = np.array([np.cov(utx_returns_hmm[labels == i].T) for i in range(2)]).reshape(2, 1, 1)

# Set initial parameters
model.means_ = means_init
model.covars_ = covars_init

# Fit the model
model.fit(utx_returns_hmm)

# Extract means to identify states
state_means = model.means_.flatten()
state_0_idx = 0 if state_means[0] < state_means[1] else 1
state_1_idx = 1 - state_0_idx

# Extract transition matrix
transition_matrix = model.transmat_

# Probability of transitioning from State 0 to State 1
prob_0_to_1 = transition_matrix[state_0_idx, state_1_idx]

# Probability of remaining in State 1
prob_1_to_1 = transition_matrix[state_1_idx, state_1_idx]

print(f"\nHMM Results:")
print(f"State 0 mean (lower): {state_means[state_0_idx]:.6f}")
print(f"State 1 mean (higher): {state_means[state_1_idx]:.6f}")
print(f"\nTransition Matrix:")
print(transition_matrix)
print(f"\nP(State 0 -> State 1): {prob_0_to_1:.3f}")
print(f"P(State 1 -> State 1): {prob_1_to_1:.3f}")

# ============================================================================
# 8. HEXBIN PLOT (UNH vs TRV Log Returns)
# ============================================================================
print("\n" + "="*80)
print("CREATING HEXBIN PLOT")
print("="*80)

plt.figure(figsize=(10, 8))
plt.hexbin(
    merged['UNH_LogReturn'],
    merged['TRV_LogReturn'],
    gridsize=20,
    cmap='Blues',
    mincnt=1
)
plt.colorbar(label='Count')
plt.xlabel('UnitedHealth Group Daily Log Returns', fontsize=12)
plt.ylabel('Travelers Companies Daily Log Returns', fontsize=12)
plt.title('Hexbin Plot: UNH vs TRV Daily Log Returns', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('hexbin_unh_trv_returns.png', dpi=300, bbox_inches='tight')
print("\nHexbin plot saved as 'hexbin_unh_trv_returns.png'")

# ============================================================================
# SUMMARY OF RESULTS
# ============================================================================
print("\n" + "="*80)
print("SUMMARY OF RESULTS")
print("="*80)

print(f"""
1. STL DECOMPOSITION (UTX):
   - Date with maximum trend: {max_trend_date.strftime('%Y-%m-%d')}

2. GAUSSIAN COPULA (UNH-TRV):
   - Correlation parameter: {copula_corr:.3f}

3. PRINCIPAL COMPONENT ANALYSIS:
   - Variance explained by PC1: {variance_pc1:.3f}
   - Cumulative variance (PC1+PC2): {cumulative_variance_pc1_pc2:.3f}
   - UTX loading on PC1: {utx_loading_pc1:.3f}

4. EXTREME VALUE THEORY (TRV):
   - GPD Shape parameter: {shape_param:.3f}
   - GPD Scale parameter: {scale_param:.3f}

5. HIDDEN MARKOV MODEL (UTX):
   - P(State 0 -> State 1): {prob_0_to_1:.3f}
   - P(State 1 -> State 1): {prob_1_to_1:.3f}

6. VISUALIZATION:
   - Hexbin plot saved as 'hexbin_unh_trv_returns.png'
""")

print("="*80)
print("ANALYSIS COMPLETE")
print("="*80)
