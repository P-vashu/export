#!/usr/bin/env python3
"""
Comprehensive Cryptocurrency Market Analysis
Author: Quantitative Analyst
Date: 2025-11-10

This script performs advanced statistical analysis on cryptocurrency datasets including:
- Bitcoin (BTC), BinanceCoin (BNB), and Aave
- Volatility analysis, cross-asset dependencies, and temporal patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import periodogram
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 80)
print("CRYPTOCURRENCY MARKET ANALYSIS")
print("=" * 80)

# ============================================================================
# TASK 1: DATA PREPARATION
# ============================================================================
print("\n[1] DATA PREPARATION")
print("-" * 80)

# Load datasets
bitcoin = pd.read_csv('coin_Bitcoin.csv')
binance = pd.read_csv('coin_BinanceCoin.csv')
aave = pd.read_csv('coin_Aave.csv')

# Convert Date columns to datetime
bitcoin['Date'] = pd.to_datetime(bitcoin['Date'])
binance['Date'] = pd.to_datetime(binance['Date'])
aave['Date'] = pd.to_datetime(aave['Date'])

# Filter date range: October 5, 2020 to July 6, 2021
start_date = pd.to_datetime('2020-10-05')
end_date = pd.to_datetime('2021-07-06')

bitcoin_filtered = bitcoin[(bitcoin['Date'] >= start_date) & (bitcoin['Date'] <= end_date)].copy()
binance_filtered = binance[(binance['Date'] >= start_date) & (binance['Date'] <= end_date)].copy()
aave_filtered = aave[(aave['Date'] >= start_date) & (aave['Date'] <= end_date)].copy()

# Create Daily_Return column: (Close - Previous Close) / Previous Close * 100
bitcoin_filtered['Daily_Return'] = (bitcoin_filtered['Close'] - bitcoin_filtered['Close'].shift(1)) / bitcoin_filtered['Close'].shift(1) * 100
binance_filtered['Daily_Return'] = (binance_filtered['Close'] - binance_filtered['Close'].shift(1)) / binance_filtered['Close'].shift(1) * 100
aave_filtered['Daily_Return'] = (aave_filtered['Close'] - aave_filtered['Close'].shift(1)) / aave_filtered['Close'].shift(1) * 100

# Remove first row where Daily_Return is undefined (NaN)
bitcoin_filtered = bitcoin_filtered.dropna(subset=['Daily_Return']).reset_index(drop=True)
binance_filtered = binance_filtered.dropna(subset=['Daily_Return']).reset_index(drop=True)
aave_filtered = aave_filtered.dropna(subset=['Daily_Return']).reset_index(drop=True)

print(f"Bitcoin records: {len(bitcoin_filtered)}")
print(f"BinanceCoin records: {len(binance_filtered)}")
print(f"Aave records: {len(aave_filtered)}")
print(f"Date range: {start_date.date()} to {end_date.date()}")

# ============================================================================
# TASK 2: SEASONAL DECOMPOSITION (BITCOIN)
# ============================================================================
print("\n[2] SEASONAL DECOMPOSITION - BITCOIN")
print("-" * 80)

# Apply seasonal decomposition to Bitcoin Close prices
# Multiplicative model, period=30, classical decomposition, extrapolate_trend='freq'
decomposition = seasonal_decompose(
    bitcoin_filtered['Close'],
    model='multiplicative',
    period=30,
    extrapolate_trend='freq'
)

# Calculate statistics
trend_mean = decomposition.trend.mean()
residual_std = decomposition.resid.std()

print(f"Mean of trend component: {trend_mean:.4f}")
print(f"Standard deviation of residual component: {residual_std:.4f}")

# ============================================================================
# TASK 3: PCA ANALYSIS
# ============================================================================
print("\n[3] PRINCIPAL COMPONENT ANALYSIS")
print("-" * 80)

# Add coin identifier before concatenation
bitcoin_pca = bitcoin_filtered.copy()
binance_pca = binance_filtered.copy()
aave_pca = aave_filtered.copy()

bitcoin_pca['Coin'] = 'Bitcoin'
binance_pca['Coin'] = 'BinanceCoin'
aave_pca['Coin'] = 'Aave'

# Vertically concatenate all three filtered datasets
combined = pd.concat([bitcoin_pca, binance_pca, aave_pca], ignore_index=True)

# Z-score normalization on specified columns
columns_to_normalize = ['High', 'Low', 'Open', 'Close', 'Volume']
scaler = StandardScaler()
combined[columns_to_normalize] = scaler.fit_transform(combined[columns_to_normalize])

# Perform PCA
pca = PCA()
pca_result = pca.fit_transform(combined[columns_to_normalize])

# Calculate explained variance
explained_variance_pc1 = pca.explained_variance_ratio_[0]
cumulative_variance_pc123 = pca.explained_variance_ratio_[:3].sum()

print(f"Explained variance ratio of PC1: {explained_variance_pc1:.4f}")
print(f"Cumulative explained variance of PC1-PC3: {cumulative_variance_pc123:.4f}")

# ============================================================================
# TASK 4: GRANGER CAUSALITY TEST
# ============================================================================
print("\n[4] GRANGER CAUSALITY TEST")
print("-" * 80)

# Inner join on Date between Bitcoin and BinanceCoin
merged_granger = pd.merge(
    bitcoin_filtered[['Date', 'Daily_Return']],
    binance_filtered[['Date', 'Daily_Return']],
    on='Date',
    suffixes=('_BTC', '_BNB')
)

# Prepare data for Granger test
granger_data = merged_granger[['Daily_Return_BTC', 'Daily_Return_BNB']].values

# Perform Granger causality test with lag=5
# Testing if Bitcoin Daily_Return Granger-causes BinanceCoin Daily_Return
maxlag = 5
granger_result = grangercausalitytests(granger_data[:, [1, 0]], maxlag=maxlag, verbose=False)

# Extract SSR F-test results for lag 5
f_statistic = granger_result[5][0]['ssr_ftest'][0]
p_value = granger_result[5][0]['ssr_ftest'][1]

print(f"Granger causality test (Bitcoin -> BinanceCoin, lag=5)")
print(f"F-statistic: {f_statistic:.3f}")
print(f"P-value: {p_value:.6f}")

# ============================================================================
# TASK 5: AUGMENTED DICKEY-FULLER TEST (AAVE)
# ============================================================================
print("\n[5] AUGMENTED DICKEY-FULLER TEST - AAVE")
print("-" * 80)

# Apply ADF test on Aave Close prices
# regression='c' for constant, autolag='AIC' with maxlag=10
adf_result = adfuller(
    aave_filtered['Close'],
    regression='c',
    autolag='AIC',
    maxlag=10
)

adf_statistic = adf_result[0]
adf_pvalue = adf_result[1]

print(f"ADF test statistic: {adf_statistic:.4f}")
print(f"P-value: {adf_pvalue:.6f}")

# ============================================================================
# TASK 6: CROSS-CORRELATION ANALYSIS
# ============================================================================
print("\n[6] CROSS-CORRELATION ANALYSIS")
print("-" * 80)

# Inner join on Date for Daily_Return
merged_corr = pd.merge(
    bitcoin_filtered[['Date', 'Daily_Return']],
    binance_filtered[['Date', 'Daily_Return']],
    on='Date',
    suffixes=('_BTC', '_BNB')
)

btc_returns = merged_corr['Daily_Return_BTC'].values
bnb_returns = merged_corr['Daily_Return_BNB'].values

# Calculate cross-correlation for lags -10 to +10
lags = range(-10, 11)
correlations = {}

for lag in lags:
    if lag < 0:
        # Negative lag: correlate BTC[t] with BNB[t - lag] = BNB[t + |lag|]
        btc_slice = btc_returns[:lag]
        bnb_slice = bnb_returns[-lag:]
    elif lag > 0:
        # Positive lag: correlate BTC[t] with BNB[t - lag]
        btc_slice = btc_returns[lag:]
        bnb_slice = bnb_returns[:-lag]
    else:
        # Zero lag
        btc_slice = btc_returns
        bnb_slice = bnb_returns

    if len(btc_slice) > 0 and len(bnb_slice) > 0:
        corr = np.corrcoef(btc_slice, bnb_slice)[0, 1]
        correlations[lag] = corr

# Find maximum absolute correlation
max_abs_corr = 0
max_lag = 0
for lag, corr in correlations.items():
    if abs(corr) > max_abs_corr:
        max_abs_corr = abs(corr)
        max_lag = lag
    elif abs(corr) == max_abs_corr and abs(lag) < abs(max_lag):
        # If tied, choose lag closest to zero
        max_lag = lag

print(f"Maximum absolute correlation: {max_abs_corr:.4f}")
print(f"Corresponding lag: {max_lag}")

# ============================================================================
# TASK 7: GENERALIZED PARETO DISTRIBUTION (BITCOIN VOLUME)
# ============================================================================
print("\n[7] GENERALIZED PARETO DISTRIBUTION - BITCOIN VOLUME")
print("-" * 80)

# Calculate 95th percentile of Bitcoin Volume
volume_95th = np.percentile(bitcoin_filtered['Volume'], 95)

# Extract exceedances above 95th percentile
exceedances = bitcoin_filtered['Volume'][bitcoin_filtered['Volume'] > volume_95th] - volume_95th

# Fit Generalized Pareto Distribution using MLE
# location parameter fixed at 0
try:
    from scipy.stats import genpareto

    # Fit GPD (location is 0, we fit shape and scale)
    shape, loc, scale = genpareto.fit(exceedances, floc=0)

    print(f"Shape parameter: {shape:.4f}")
    print(f"Scale parameter: {scale:.4f}")
except Exception as e:
    print(f"Error fitting GPD: {e}")

# ============================================================================
# TASK 8: QUANTILE REGRESSION (BINANCECOIN)
# ============================================================================
print("\n[8] QUANTILE REGRESSION - BINANCECOIN")
print("-" * 80)

# Quantile regression with Volume as independent, Close as dependent
try:
    import statsmodels.formula.api as smf
    from statsmodels.regression.quantile_regression import QuantReg
    import statsmodels.api as sm

    # Prepare data
    X = sm.add_constant(binance_filtered['Volume'])
    y = binance_filtered['Close']

    # Fit quantile regression at quantiles 0.25, 0.50, 0.75
    quantiles = [0.25, 0.50, 0.75]
    results = {}

    for q in quantiles:
        model = QuantReg(y, X)
        result = model.fit(q=q)
        results[q] = result

    # Report slope coefficient at 0.50 quantile
    slope_050 = results[0.50].params['Volume']

    print(f"Slope coefficient at 0.50 quantile: {slope_050:.5f}")
    print(f"  (Note: value is {slope_050:.8e} in scientific notation)")

    # Display all results
    for q in quantiles:
        print(f"\nQuantile {q}:")
        print(f"  Intercept: {results[q].params['const']:.8f}")
        print(f"  Slope: {results[q].params['Volume']:.5f} ({results[q].params['Volume']:.8e})")

except Exception as e:
    print(f"Error in quantile regression: {e}")

# ============================================================================
# TASK 9: SPECTRAL ANALYSIS (AAVE)
# ============================================================================
print("\n[9] SPECTRAL ANALYSIS - AAVE")
print("-" * 80)

# Subtract mean from Aave Close prices
close_demeaned = aave_filtered['Close'] - aave_filtered['Close'].mean()

# Compute periodogram using FFT
N = len(close_demeaned)
fft_result = np.fft.fft(close_demeaned)
power_spectrum = (1 / N) * np.abs(fft_result) ** 2

# Get frequencies (cycles per day, assuming daily data)
frequencies = np.fft.fftfreq(N, d=1)  # d=1 for daily sampling

# Consider only positive frequencies up to Nyquist (0.5)
# Exclude DC component (frequency = 0)
positive_freq_mask = (frequencies > 0) & (frequencies <= 0.5)
positive_frequencies = frequencies[positive_freq_mask]
positive_power = power_spectrum[positive_freq_mask]

# Find frequency with highest power spectral density
max_power_idx = np.argmax(positive_power)
peak_frequency = positive_frequencies[max_power_idx]
peak_power = positive_power[max_power_idx]

# Calculate periodicity in days
if peak_frequency > 0:
    periodicity_days = 1 / peak_frequency
else:
    periodicity_days = np.inf

print(f"Frequency with highest PSD: {peak_frequency:.6f} cycles/day")
print(f"Periodicity: {periodicity_days:.2f} days")
print(f"Peak power: {peak_power:.6e}")

# ============================================================================
# TASK 10: HEXBIN PLOT
# ============================================================================
print("\n[10] HEXBIN PLOT - BITCOIN vs BINANCECOIN DAILY RETURNS")
print("-" * 80)

# Inner join on Date for hexbin plot
merged_hexbin = pd.merge(
    bitcoin_filtered[['Date', 'Daily_Return']],
    binance_filtered[['Date', 'Daily_Return']],
    on='Date',
    suffixes=('_BTC', '_BNB')
)

# Create hexbin plot
plt.figure(figsize=(10, 8))
hexbin = plt.hexbin(
    merged_hexbin['Daily_Return_BTC'],
    merged_hexbin['Daily_Return_BNB'],
    gridsize=30,
    cmap='YlOrRd',
    mincnt=1
)
plt.colorbar(hexbin, label='Count')
plt.xlabel('Bitcoin Daily Return (%)', fontsize=12)
plt.ylabel('BinanceCoin Daily Return (%)', fontsize=12)
plt.title('Hexbin Plot: Bitcoin vs BinanceCoin Daily Returns', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('hexbin_btc_bnb_returns.png', dpi=300, bbox_inches='tight')
print("Hexbin plot saved as 'hexbin_btc_bnb_returns.png'")

# ============================================================================
# SUMMARY REPORT
# ============================================================================
print("\n" + "=" * 80)
print("ANALYSIS SUMMARY")
print("=" * 80)

summary = f"""
1. DATA PREPARATION:
   - Bitcoin records: {len(bitcoin_filtered)}
   - BinanceCoin records: {len(binance_filtered)}
   - Aave records: {len(aave_filtered)}
   - Date range: {start_date.date()} to {end_date.date()}

2. SEASONAL DECOMPOSITION (Bitcoin):
   - Mean of trend component: {trend_mean:.4f}
   - Std dev of residual component: {residual_std:.4f}

3. PRINCIPAL COMPONENT ANALYSIS:
   - Explained variance ratio (PC1): {explained_variance_pc1:.4f}
   - Cumulative variance (PC1-PC3): {cumulative_variance_pc123:.4f}

4. GRANGER CAUSALITY TEST (Bitcoin -> BinanceCoin):
   - F-statistic: {f_statistic:.3f}
   - P-value: {p_value:.6f}

5. AUGMENTED DICKEY-FULLER TEST (Aave):
   - Test statistic: {adf_statistic:.4f}
   - P-value: {adf_pvalue:.6f}

6. CROSS-CORRELATION ANALYSIS:
   - Maximum absolute correlation: {max_abs_corr:.4f}
   - Corresponding lag: {max_lag}

7. GENERALIZED PARETO DISTRIBUTION (Bitcoin Volume):
   - Shape parameter: {shape:.4f}
   - Scale parameter: {scale:.4f}

8. QUANTILE REGRESSION (BinanceCoin):
   - Slope coefficient at 0.50 quantile: {slope_050:.5f} ({slope_050:.8e})

9. SPECTRAL ANALYSIS (Aave):
   - Frequency with highest PSD: {peak_frequency:.6f} cycles/day
   - Periodicity: {periodicity_days:.2f} days

10. HEXBIN PLOT:
    - Generated: hexbin_btc_bnb_returns.png
"""

print(summary)

# Save summary to file
with open('analysis_summary.txt', 'w') as f:
    f.write("CRYPTOCURRENCY MARKET ANALYSIS - SUMMARY REPORT\n")
    f.write("=" * 80 + "\n")
    f.write(summary)

print("\nAnalysis complete! Summary saved to 'analysis_summary.txt'")
print("=" * 80)
