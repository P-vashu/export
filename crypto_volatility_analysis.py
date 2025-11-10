"""
Cryptocurrency Volatility Analysis
Advanced statistical analysis on Bitcoin, BinanceCoin, and Aave datasets
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.tsa.stattools import grangercausalitytests
from scipy.stats import genpareto
from statsmodels.regression.quantile_regression import QuantReg
from scipy.fft import fft
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("CRYPTOCURRENCY VOLATILITY ANALYSIS")
print("="*80)

# ============================================================================
# 1. DATA LOADING AND PREPROCESSING
# ============================================================================
print("\n1. DATA LOADING AND PREPROCESSING")
print("-"*80)

# Load datasets
df_bitcoin = pd.read_csv('coin_Bitcoin.csv')
df_binance = pd.read_csv('coin_BinanceCoin.csv')
df_aave = pd.read_csv('coin_Aave.csv')

# Convert Date columns to datetime
df_bitcoin['Date'] = pd.to_datetime(df_bitcoin['Date'])
df_binance['Date'] = pd.to_datetime(df_binance['Date'])
df_aave['Date'] = pd.to_datetime(df_aave['Date'])

# Filter for October 5, 2020 to July 6, 2021
start_date = pd.to_datetime('2020-10-05')
end_date = pd.to_datetime('2021-07-06')

df_bitcoin_filtered = df_bitcoin[(df_bitcoin['Date'] >= start_date) &
                                 (df_bitcoin['Date'] <= end_date)].copy()
df_binance_filtered = df_binance[(df_binance['Date'] >= start_date) &
                                 (df_binance['Date'] <= end_date)].copy()
df_aave_filtered = df_aave[(df_aave['Date'] >= start_date) &
                           (df_aave['Date'] <= end_date)].copy()

print(f"Bitcoin records: {len(df_bitcoin_filtered)}")
print(f"BinanceCoin records: {len(df_binance_filtered)}")
print(f"Aave records: {len(df_aave_filtered)}")

# Calculate Daily_Return
df_bitcoin_filtered['Daily_Return'] = (df_bitcoin_filtered['Close'] -
                                        df_bitcoin_filtered['Close'].shift(1)) / \
                                        df_bitcoin_filtered['Close'].shift(1) * 100

df_binance_filtered['Daily_Return'] = (df_binance_filtered['Close'] -
                                        df_binance_filtered['Close'].shift(1)) / \
                                        df_binance_filtered['Close'].shift(1) * 100

df_aave_filtered['Daily_Return'] = (df_aave_filtered['Close'] -
                                     df_aave_filtered['Close'].shift(1)) / \
                                     df_aave_filtered['Close'].shift(1) * 100

# Remove first row with undefined Daily_Return
df_bitcoin_filtered = df_bitcoin_filtered.dropna(subset=['Daily_Return']).reset_index(drop=True)
df_binance_filtered = df_binance_filtered.dropna(subset=['Daily_Return']).reset_index(drop=True)
df_aave_filtered = df_aave_filtered.dropna(subset=['Daily_Return']).reset_index(drop=True)

print(f"\nAfter removing first row:")
print(f"Bitcoin records: {len(df_bitcoin_filtered)}")
print(f"BinanceCoin records: {len(df_binance_filtered)}")
print(f"Aave records: {len(df_aave_filtered)}")

# ============================================================================
# 2. SEASONAL DECOMPOSITION (Bitcoin)
# ============================================================================
print("\n2. SEASONAL DECOMPOSITION - BITCOIN")
print("-"*80)

# Set Date as index for decomposition
bitcoin_ts = df_bitcoin_filtered.set_index('Date')['Close']

# Perform seasonal decomposition
decomposition = seasonal_decompose(
    bitcoin_ts,
    model='multiplicative',
    period=30,
    extrapolate_trend='freq'
)

# Calculate statistics
trend_mean = decomposition.trend.mean()
residual_std = decomposition.resid.std()

print(f"Trend Component Mean: {trend_mean:.4f}")
print(f"Residual Component Std Dev: {residual_std:.4f}")

# ============================================================================
# 3. PCA ANALYSIS ON CONCATENATED DATA
# ============================================================================
print("\n3. PCA ANALYSIS ON CONCATENATED DATA")
print("-"*80)

# Vertically concatenate all three datasets
df_combined = pd.concat([df_bitcoin_filtered, df_binance_filtered, df_aave_filtered],
                        ignore_index=True)

print(f"Combined dataset size: {len(df_combined)}")

# Select columns for normalization
columns_to_normalize = ['High', 'Low', 'Open', 'Close', 'Volume']

# Apply z-score normalization
scaler = StandardScaler()
df_combined_normalized = df_combined.copy()
df_combined_normalized[columns_to_normalize] = scaler.fit_transform(
    df_combined[columns_to_normalize]
)

# Perform PCA
pca = PCA()
pca_result = pca.fit_transform(df_combined_normalized[columns_to_normalize])

# Get explained variance ratios
explained_variance_pc1 = pca.explained_variance_ratio_[0]
cumulative_variance_pc3 = pca.explained_variance_ratio_[:3].sum()

print(f"Explained Variance Ratio (PC1): {explained_variance_pc1:.4f}")
print(f"Cumulative Explained Variance (PC1-3): {cumulative_variance_pc3:.4f}")

# ============================================================================
# 4. GRANGER CAUSALITY TEST
# ============================================================================
print("\n4. GRANGER CAUSALITY TEST")
print("-"*80)

# Inner join on Date between Bitcoin and BinanceCoin Daily_Return
df_granger = pd.merge(
    df_bitcoin_filtered[['Date', 'Daily_Return']],
    df_binance_filtered[['Date', 'Daily_Return']],
    on='Date',
    how='inner',
    suffixes=('_Bitcoin', '_BinanceCoin')
)

print(f"Merged dataset size for Granger test: {len(df_granger)}")

# Prepare data for Granger test
granger_data = df_granger[['Daily_Return_Bitcoin', 'Daily_Return_BinanceCoin']].values

# Perform Granger causality test with lag 5
# Test if BinanceCoin Daily_Return Granger-causes Bitcoin Daily_Return
granger_results = grangercausalitytests(granger_data, maxlag=5, verbose=False)

# Extract F-statistic and p-value for lag 5 using SSR F-test
f_statistic = granger_results[5][0]['ssr_ftest'][0]
p_value = granger_results[5][0]['ssr_ftest'][1]

print(f"Granger Causality Test (Lag 5, SSR F-test):")
print(f"F-statistic: {f_statistic:.3f}")
print(f"P-value: {p_value:.6f}")

# ============================================================================
# 5. GENERALIZED PARETO DISTRIBUTION (Bitcoin Volume)
# ============================================================================
print("\n5. GENERALIZED PARETO DISTRIBUTION - BITCOIN VOLUME")
print("-"*80)

# Calculate 95th percentile threshold
threshold = df_bitcoin_filtered['Volume'].quantile(0.95)
print(f"95th percentile threshold: {threshold:.2f}")

# Get exceedances above threshold
exceedances = df_bitcoin_filtered['Volume'][df_bitcoin_filtered['Volume'] > threshold] - threshold

print(f"Number of exceedances: {len(exceedances)}")

# Fit Generalized Pareto Distribution with location=0 (using exceedances)
# MLE fitting
shape, loc_fit, scale = genpareto.fit(exceedances, floc=0)

print(f"GPD Shape Parameter: {shape:.4f}")
print(f"GPD Scale Parameter: {scale:.4f}")

# ============================================================================
# 6. QUANTILE REGRESSION (BinanceCoin)
# ============================================================================
print("\n6. QUANTILE REGRESSION - BINANCECOIN")
print("-"*80)

# Prepare data: Volume (independent), Close (dependent)
X = df_binance_filtered['Volume']
y = df_binance_filtered['Close']

# Add constant (intercept)
from statsmodels.tools import add_constant
X_with_const = add_constant(X)

# Fit quantile regression at quantiles 0.25, 0.50, 0.75
quantiles = [0.25, 0.50, 0.75]
results = {}

for q in quantiles:
    model = QuantReg(y, X_with_const)
    result = model.fit(q=q)
    results[q] = result
    print(f"\nQuantile {q}:")
    print(f"  Intercept: {result.params['const']:.6f}")
    print(f"  Slope: {result.params['Volume']:.6f}")

# Report slope at 0.50 quantile
slope_050 = results[0.50].params['Volume']
print(f"\nSlope Coefficient at 0.50 Quantile: {slope_050:.5f}")

# ============================================================================
# 7. SPECTRAL ANALYSIS (Aave Close)
# ============================================================================
print("\n7. SPECTRAL ANALYSIS - AAVE CLOSE")
print("-"*80)

# Get Close prices
aave_close = df_aave_filtered['Close'].values

# Subtract mean
aave_close_demeaned = aave_close - aave_close.mean()

# Number of data points
N = len(aave_close_demeaned)
print(f"Number of data points: {N}")

# Compute FFT
fft_result = fft(aave_close_demeaned)

# Compute periodogram: (1/N) * |FFT|^2
periodogram = (1 / N) * np.abs(fft_result) ** 2

# Frequencies (cycles per day, assuming daily data)
# Positive frequencies up to Nyquist frequency (0.5)
frequencies = np.fft.fftfreq(N, d=1)  # d=1 for daily sampling

# Keep only positive frequencies (exclude DC component at frequency 0)
positive_freq_mask = frequencies > 0
positive_frequencies = frequencies[positive_freq_mask]
positive_periodogram = periodogram[positive_freq_mask]

# Find frequency with highest power spectral density
max_psd_idx = np.argmax(positive_periodogram)
frequency_max_psd = positive_frequencies[max_psd_idx]
max_psd_value = positive_periodogram[max_psd_idx]

# Calculate periodicity in days
periodicity_days = 1 / frequency_max_psd if frequency_max_psd > 0 else np.inf

print(f"Frequency with Highest PSD: {frequency_max_psd:.6f} cycles/day")
print(f"Periodicity: {periodicity_days:.2f} days")
print(f"Maximum PSD Value: {max_psd_value:.6e}")

# ============================================================================
# 8. HEXBIN PLOT
# ============================================================================
print("\n8. GENERATING HEXBIN PLOT")
print("-"*80)

# Inner join Bitcoin and BinanceCoin on Date for Daily_Return
df_hexbin = pd.merge(
    df_bitcoin_filtered[['Date', 'Daily_Return']],
    df_binance_filtered[['Date', 'Daily_Return']],
    on='Date',
    how='inner',
    suffixes=('_Bitcoin', '_BinanceCoin')
)

print(f"Data points for hexbin plot: {len(df_hexbin)}")

# Create hexbin plot
plt.figure(figsize=(10, 8))
hexbin = plt.hexbin(
    df_hexbin['Daily_Return_Bitcoin'],
    df_hexbin['Daily_Return_BinanceCoin'],
    gridsize=30,
    cmap='YlOrRd',
    mincnt=1
)

plt.colorbar(hexbin, label='Count')
plt.xlabel('Bitcoin Daily Return (%)', fontsize=12)
plt.ylabel('BinanceCoin Daily Return (%)', fontsize=12)
plt.title('Hexbin Plot: Bitcoin vs BinanceCoin Daily Returns\n(Oct 5, 2020 - Jul 6, 2021)',
          fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('hexbin_bitcoin_binancecoin_returns.png', dpi=300, bbox_inches='tight')
print("Hexbin plot saved as 'hexbin_bitcoin_binancecoin_returns.png'")

# ============================================================================
# SUMMARY OF RESULTS
# ============================================================================
print("\n" + "="*80)
print("SUMMARY OF RESULTS")
print("="*80)

print("\n1. SEASONAL DECOMPOSITION (Bitcoin):")
print(f"   - Trend Component Mean: {trend_mean:.4f}")
print(f"   - Residual Component Std Dev: {residual_std:.4f}")

print("\n2. PCA ANALYSIS:")
print(f"   - Explained Variance Ratio (PC1): {explained_variance_pc1:.4f}")
print(f"   - Cumulative Explained Variance (PC1-3): {cumulative_variance_pc3:.4f}")

print("\n3. GRANGER CAUSALITY TEST (Bitcoin vs BinanceCoin, Lag 5):")
print(f"   - F-statistic: {f_statistic:.3f}")
print(f"   - P-value: {p_value:.6f}")

print("\n4. GENERALIZED PARETO DISTRIBUTION (Bitcoin Volume):")
print(f"   - Shape Parameter: {shape:.4f}")
print(f"   - Scale Parameter: {scale:.4f}")

print("\n5. QUANTILE REGRESSION (BinanceCoin):")
print(f"   - Slope Coefficient at 0.50 Quantile: {slope_050:.5f}")

print("\n6. SPECTRAL ANALYSIS (Aave):")
print(f"   - Frequency with Highest PSD: {frequency_max_psd:.6f} cycles/day")
print(f"   - Periodicity: {periodicity_days:.2f} days")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
