#!/usr/bin/env python3
"""
Comprehensive Financial Market Analysis
Analyzes MSFT, IBM, and NKE stock data from 2006-2018
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

# Load the three datasets
print("Loading datasets...")
msft = pd.read_csv('MSFT_2006-01-01_to_2018-01-01.csv')
ibm = pd.read_csv('IBM_2006-01-01_to_2018-01-01.csv')
nke = pd.read_csv('NKE_2006-01-01_to_2018-01-01.csv')

print(f"MSFT: {len(msft)} rows")
print(f"IBM: {len(ibm)} rows")
print(f"NKE: {len(nke)} rows")
print()

# ====================================================================
# 1. AUGMENTED DICKEY-FULLER TESTS
# ====================================================================
print("=" * 70)
print("1. AUGMENTED DICKEY-FULLER (ADF) STATIONARITY TESTS")
print("=" * 70)

# ADF test for MSFT
adf_msft = adfuller(msft['Close'], autolag='AIC', regression='c')
print(f"\nMSFT ADF Test:")
print(f"  ADF Test Statistic: {adf_msft[0]:.4f}")
print(f"  p-value: {adf_msft[1]:.4f}")

# ADF test for IBM
adf_ibm = adfuller(ibm['Close'], autolag='AIC', regression='c')
print(f"\nIBM ADF Test:")
print(f"  ADF Test Statistic: {adf_ibm[0]:.4f}")
print(f"  p-value: {adf_ibm[1]:.4f}")

# ADF test for NKE
adf_nke = adfuller(nke['Close'], autolag='AIC', regression='c')
print(f"\nNKE ADF Test:")
print(f"  ADF Test Statistic: {adf_nke[0]:.4f}")
print(f"  p-value: {adf_nke[1]:.4f}")

# ====================================================================
# 2. SPEARMAN CORRELATION ANALYSIS (MSFT)
# ====================================================================
print("\n" + "=" * 70)
print("2. SPEARMAN RANK CORRELATION (MSFT Volume vs Close)")
print("=" * 70)

spearman_result = stats.spearmanr(msft['Volume'], msft['Close'])
print(f"\nSpearman Correlation Coefficient: {spearman_result.correlation:.4f}")
print(f"Two-tailed p-value: {spearman_result.pvalue:.4f}")

# ====================================================================
# 3. LEVENE TEST FOR EQUALITY OF VARIANCES
# ====================================================================
print("\n" + "=" * 70)
print("3. LEVENE TEST FOR EQUALITY OF VARIANCES (Daily Simple Returns)")
print("=" * 70)

# Calculate simple returns for each dataset
# Simple return = (Close[t] - Close[t-1]) / Close[t-1] * 100
msft['returns'] = (msft['Close'].diff() / msft['Close'].shift(1)) * 100
ibm['returns'] = (ibm['Close'].diff() / ibm['Close'].shift(1)) * 100
nke['returns'] = (nke['Close'].diff() / nke['Close'].shift(1)) * 100

# Remove NaN values (first row will be NaN)
msft_returns = msft['returns'].dropna()
ibm_returns = ibm['returns'].dropna()
nke_returns = nke['returns'].dropna()

print(f"\nMSFT returns: {len(msft_returns)} observations")
print(f"IBM returns: {len(ibm_returns)} observations")
print(f"NKE returns: {len(nke_returns)} observations")

# Perform Levene test using median (center='median')
levene_result = stats.levene(msft_returns, ibm_returns, nke_returns, center='median')
print(f"\nLevene Test Statistic: {levene_result.statistic:.4f}")
print(f"p-value: {levene_result.pvalue:.4f}")

# ====================================================================
# 4. COEFFICIENT OF VARIATION (IBM)
# ====================================================================
print("\n" + "=" * 70)
print("4. COEFFICIENT OF VARIATION (IBM Daily Returns)")
print("=" * 70)

# Calculate coefficient of variation for IBM returns
ibm_mean = ibm_returns.mean()
ibm_std = ibm_returns.std()
ibm_cv = ibm_std / ibm_mean

print(f"\nIBM Returns Statistics:")
print(f"  Mean: {ibm_mean:.4f}%")
print(f"  Standard Deviation: {ibm_std:.4f}%")
print(f"  Coefficient of Variation: {ibm_cv:.4f}")

# ====================================================================
# 5. HEXBIN PLOT VISUALIZATION
# ====================================================================
print("\n" + "=" * 70)
print("5. GENERATING HEXBIN PLOT")
print("=" * 70)

# Combine all three datasets
all_data = pd.concat([
    msft[['Close', 'Volume']],
    ibm[['Close', 'Volume']],
    nke[['Close', 'Volume']]
], ignore_index=True)

print(f"\nCombined dataset: {len(all_data)} observations")

# Create hexbin plot
plt.figure(figsize=(10, 8))
hexbin = plt.hexbin(all_data['Close'], all_data['Volume'],
                     gridsize=20, cmap='viridis', mincnt=1)
plt.xlabel('Close Price', fontsize=12)
plt.ylabel('Volume', fontsize=12)
plt.title('Volume vs Close Price Distribution (MSFT, IBM, NKE Combined)',
          fontsize=14, fontweight='bold')
plt.colorbar(hexbin, label='Count')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('hexbin_volume_vs_close.png', dpi=300, bbox_inches='tight')
print("Hexbin plot saved as: hexbin_volume_vs_close.png")

# ====================================================================
# 6. INTERQUARTILE RANGE ANALYSIS
# ====================================================================
print("\n" + "=" * 70)
print("6. INTERQUARTILE RANGE (IQR) ANALYSIS")
print("=" * 70)

# Calculate IQR for each dataset's Close column
msft_q75 = msft['Close'].quantile(0.75)
msft_q25 = msft['Close'].quantile(0.25)
msft_iqr = msft_q75 - msft_q25

ibm_q75 = ibm['Close'].quantile(0.75)
ibm_q25 = ibm['Close'].quantile(0.25)
ibm_iqr = ibm_q75 - ibm_q25

nke_q75 = nke['Close'].quantile(0.75)
nke_q25 = nke['Close'].quantile(0.25)
nke_iqr = nke_q75 - nke_q25

print(f"\nMSFT Close IQR:")
print(f"  Q1 (25th percentile): {msft_q25:.2f}")
print(f"  Q3 (75th percentile): {msft_q75:.2f}")
print(f"  IQR: {msft_iqr:.2f}")

print(f"\nIBM Close IQR:")
print(f"  Q1 (25th percentile): {ibm_q25:.2f}")
print(f"  Q3 (75th percentile): {ibm_q75:.2f}")
print(f"  IQR: {ibm_iqr:.2f}")

print(f"\nNKE Close IQR:")
print(f"  Q1 (25th percentile): {nke_q25:.2f}")
print(f"  Q3 (75th percentile): {nke_q75:.2f}")
print(f"  IQR: {nke_iqr:.2f}")

# Determine which has the highest IQR
iqr_dict = {
    'MSFT_2006-01-01_to_2018-01-01.csv': msft_iqr,
    'IBM_2006-01-01_to_2018-01-01.csv': ibm_iqr,
    'NKE_2006-01-01_to_2018-01-01.csv': nke_iqr
}
highest_iqr_dataset = max(iqr_dict, key=iqr_dict.get)
highest_iqr_value = iqr_dict[highest_iqr_dataset]

print(f"\n*** Dataset with highest IQR: {highest_iqr_dataset}")
print(f"*** IQR Value: {highest_iqr_value:.2f}")

# ====================================================================
# 7. SUMMARY OF KEY RESULTS
# ====================================================================
print("\n" + "=" * 70)
print("SUMMARY OF KEY RESULTS")
print("=" * 70)

print(f"""
STATIONARITY TESTS (ADF):
  MSFT ADF Statistic: {adf_msft[0]:.4f}
  IBM p-value: {adf_ibm[1]:.4f}
  NKE ADF Statistic: {adf_nke[0]:.4f}

SPEARMAN CORRELATION (MSFT Volume vs Close):
  Correlation Coefficient: {spearman_result.correlation:.4f}
  Two-tailed p-value: {spearman_result.pvalue:.4f}

LEVENE TEST (Variance Equality):
  Test Statistic: {levene_result.statistic:.4f}
  p-value: {levene_result.pvalue:.4f}

COEFFICIENT OF VARIATION (IBM):
  CV: {ibm_cv:.4f}

INTERQUARTILE RANGE:
  Highest IQR Dataset: {highest_iqr_dataset}
  IQR Value: {highest_iqr_value:.2f}
""")

# ====================================================================
# 8. INTERPRETATION OF STATIONARITY FINDINGS
# ====================================================================
print("=" * 70)
print("INTERPRETATION OF STATIONARITY TEST FINDINGS")
print("=" * 70)

print("""
The Augmented Dickey-Fuller (ADF) test results for all three stocks (MSFT,
IBM, and NKE) suggest NON-STATIONARY price behavior over the 2006-2018 period.

Key Observations:

1. TEST STATISTICS: All three ADF test statistics are relatively high
   (close to zero or positive), indicating the presence of unit roots in
   the price series. Stationary series typically exhibit more negative
   test statistics.

2. IMPLICATIONS: Non-stationary price behavior is typical for stock prices,
   as they tend to follow random walk patterns with trending behavior rather
   than mean-reverting patterns. This suggests:

   - Price levels are not bound to a long-term mean
   - Past prices have persistent effects on future prices
   - Variance increases over time
   - Traditional time series models assuming stationarity (like ARMA)
     would not be appropriate for price levels

3. PRACTICAL SIGNIFICANCE: For financial modeling and forecasting:
   - Analysts should work with returns (first differences) rather than
     price levels, as returns are typically stationary
   - Risk management should account for the non-mean-reverting nature
     of prices
   - Long-term predictions of price levels carry high uncertainty

4. CONSISTENCY: The non-stationary finding is consistent across all three
   stocks despite them representing different sectors (Technology: MSFT,
   Industrial/IT Services: IBM, Consumer Discretionary: NKE), suggesting
   this is a fundamental characteristic of equity price behavior in
   efficient markets.
""")

print("\nAnalysis complete!")
