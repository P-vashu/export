"""
Quantitative Financial Analysis: Stock Dependency and Causality Analysis
Analyzing AAPL, AMZN, and AABA stock data from 2006-2018
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests, adfuller, kpss
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

# ============================================================================
# STEP 1: DATA LOADING AND PREPROCESSING
# ============================================================================

print("="*80)
print("STOCK DEPENDENCY ANALYSIS - DATA PREPROCESSING")
print("="*80)

# Load the three CSV files
aapl = pd.read_csv('AAPL_2006-01-01_to_2018-01-01.csv')
amzn = pd.read_csv('AMZN_2006-01-01_to_2018-01-01.csv')
aaba = pd.read_csv('AABA_2006-01-01_to_2018-01-01.csv')

# Convert Date column to datetime format
aapl['Date'] = pd.to_datetime(aapl['Date'])
amzn['Date'] = pd.to_datetime(amzn['Date'])
aaba['Date'] = pd.to_datetime(aaba['Date'])

# Extract Year and Month
aapl['Year'] = aapl['Date'].dt.year
aapl['Month'] = aapl['Date'].dt.month
amzn['Year'] = amzn['Date'].dt.year
amzn['Month'] = amzn['Date'].dt.month
aaba['Year'] = aaba['Date'].dt.year
aaba['Month'] = aaba['Date'].dt.month

# Merge Close columns into a single dataframe
merged_df = pd.merge(aapl[['Date', 'Close']], amzn[['Date', 'Close']],
                     on='Date', how='inner', suffixes=('_AAPL', '_AMZN'))
merged_df = pd.merge(merged_df, aaba[['Date', 'Close']],
                     on='Date', how='inner')
merged_df.columns = ['Date', 'AAPL_Close', 'AMZN_Close', 'AABA_Close']

# Calculate daily returns (percentage change as decimal proportion)
merged_df['AAPL_Return'] = merged_df['AAPL_Close'].pct_change()
merged_df['AMZN_Return'] = merged_df['AMZN_Close'].pct_change()
merged_df['AABA_Return'] = merged_df['AABA_Close'].pct_change()

# Remove all rows with missing values
merged_df = merged_df.dropna()

print(f"\nData shape after preprocessing: {merged_df.shape}")
print(f"Date range: {merged_df['Date'].min()} to {merged_df['Date'].max()}")
print(f"Number of trading days: {len(merged_df)}")
print("\nFirst few rows of daily returns:")
print(merged_df[['Date', 'AAPL_Return', 'AMZN_Return', 'AABA_Return']].head())

# ============================================================================
# STEP 2: GRANGER CAUSALITY TESTS
# ============================================================================

print("\n" + "="*80)
print("GRANGER CAUSALITY TESTS")
print("="*80)

# Granger causality test: AAPL -> AMZN
print("\n1. Granger Causality Test: AAPL daily returns -> AMZN daily returns (lag=5)")
print("-" * 80)
data_aapl_amzn = merged_df[['AMZN_Return', 'AAPL_Return']].values
granger_result_aapl_amzn = grangercausalitytests(data_aapl_amzn, maxlag=5, verbose=False)

# Extract F-statistic and p-value for lag 5
ssr_ftest = granger_result_aapl_amzn[5][0]['ssr_ftest']
f_stat_aapl_amzn = ssr_ftest[0]
p_value_aapl_amzn = ssr_ftest[1]

print(f"F-statistic: {f_stat_aapl_amzn:.4f}")
print(f"p-value: {p_value_aapl_amzn:.6f}")
print(f"Null Hypothesis: AAPL daily returns do NOT Granger-cause AMZN daily returns")
if p_value_aapl_amzn < 0.05:
    print("Result: REJECT null hypothesis (AAPL Granger-causes AMZN)")
else:
    print("Result: FAIL TO REJECT null hypothesis (no Granger causality)")

# Granger causality test: AMZN -> AABA
print("\n2. Granger Causality Test: AMZN daily returns -> AABA daily returns (lag=5)")
print("-" * 80)
data_amzn_aaba = merged_df[['AABA_Return', 'AMZN_Return']].values
granger_result_amzn_aaba = grangercausalitytests(data_amzn_aaba, maxlag=5, verbose=False)

# Extract F-statistic and p-value for lag 5
ssr_ftest = granger_result_amzn_aaba[5][0]['ssr_ftest']
f_stat_amzn_aaba = ssr_ftest[0]
p_value_amzn_aaba = ssr_ftest[1]

print(f"F-statistic: {f_stat_amzn_aaba:.4f}")
print(f"p-value: {p_value_amzn_aaba:.6f}")
print(f"Null Hypothesis: AMZN daily returns do NOT Granger-cause AABA daily returns")
if p_value_amzn_aaba < 0.05:
    print("Result: REJECT null hypothesis (AMZN Granger-causes AABA)")
else:
    print("Result: FAIL TO REJECT null hypothesis (no Granger causality)")

# ============================================================================
# STEP 3: STATIONARITY TESTS
# ============================================================================

print("\n" + "="*80)
print("STATIONARITY TESTS")
print("="*80)

# ADF test on AAPL daily returns
print("\n1. Augmented Dickey-Fuller (ADF) Test on AAPL Daily Returns")
print("-" * 80)
adf_result = adfuller(merged_df['AAPL_Return'], autolag='AIC')
adf_statistic = adf_result[0]
adf_pvalue = adf_result[1]

print(f"ADF Test Statistic: {adf_statistic:.4f}")
print(f"p-value: {adf_pvalue:.6f}")
print(f"Null Hypothesis: Series has a unit root (non-stationary)")
if adf_pvalue < 0.05:
    print("Result: REJECT null hypothesis (series is stationary)")
else:
    print("Result: FAIL TO REJECT null hypothesis (series is non-stationary)")

# KPSS test on AMZN daily returns
print("\n2. KPSS Test on AMZN Daily Returns")
print("-" * 80)
kpss_result = kpss(merged_df['AMZN_Return'], regression='c', nlags='auto')
kpss_statistic = kpss_result[0]
kpss_pvalue = kpss_result[1]

print(f"KPSS Test Statistic: {kpss_statistic:.4f}")
print(f"p-value: {kpss_pvalue:.6f}")
print(f"Null Hypothesis: Series is stationary")
if kpss_pvalue < 0.05:
    print("Result: REJECT null hypothesis (series is non-stationary)")
else:
    print("Result: FAIL TO REJECT null hypothesis (series is stationary)")

# ============================================================================
# STEP 4: COPULA AND RANK CORRELATION ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("COPULA AND RANK CORRELATION ANALYSIS")
print("="*80)

# Gaussian Copula correlation for AAPL & AABA daily returns
print("\n1. Gaussian Copula Correlation: AAPL & AABA Daily Returns")
print("-" * 80)

# Rank transformation (convert to uniform margins using empirical CDF)
aapl_returns = merged_df['AAPL_Return'].values
aaba_returns = merged_df['AABA_Return'].values

# Compute ranks and convert to uniform [0,1]
n = len(aapl_returns)
aapl_ranks = stats.rankdata(aapl_returns) / (n + 1)
aaba_ranks = stats.rankdata(aaba_returns) / (n + 1)

# Transform to standard normal (inverse normal transformation)
aapl_normal = stats.norm.ppf(aapl_ranks)
aaba_normal = stats.norm.ppf(aaba_ranks)

# Compute correlation of transformed data (Gaussian copula parameter)
copula_correlation = np.corrcoef(aapl_normal, aaba_normal)[0, 1]

print(f"Gaussian Copula Correlation Parameter: {copula_correlation:.4f}")

# Kendall's Tau test for AMZN & AABA daily returns
print("\n2. Kendall's Tau Rank Correlation Test: AMZN & AABA Daily Returns")
print("-" * 80)

amzn_returns = merged_df['AMZN_Return'].values
kendall_tau, kendall_pvalue = stats.kendalltau(amzn_returns, aaba_returns)

print(f"Kendall's Tau: {kendall_tau:.4f}")
print(f"p-value: {kendall_pvalue:.6f}")
if kendall_pvalue < 0.05:
    print("Result: Significant rank correlation between AMZN and AABA returns")
else:
    print("Result: No significant rank correlation")

# ============================================================================
# STEP 5: RIDGE REGRESSION
# ============================================================================

print("\n" + "="*80)
print("RIDGE REGRESSION ANALYSIS")
print("="*80)

print("\nRidge Regression: Predicting AABA Daily Returns")
print("Independent Variables: AAPL and AMZN daily returns")
print("Regularization: L2 with alpha=1.0")
print("-" * 80)

# Prepare data for Ridge regression
X = merged_df[['AAPL_Return', 'AMZN_Return']].values
y = merged_df['AABA_Return'].values

# Fit Ridge regression with alpha=1.0
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X, y)

# Extract coefficients
coef_aapl = ridge_model.coef_[0]
coef_amzn = ridge_model.coef_[1]
intercept = ridge_model.intercept_

# Calculate R-squared
y_pred = ridge_model.predict(X)
r_squared = r2_score(y, y_pred)

print(f"\nRidge Coefficient for AAPL daily returns: {coef_aapl:.6f}")
print(f"Ridge Coefficient for AMZN daily returns: {coef_amzn:.6f}")
print(f"Intercept: {intercept:.6f}")
print(f"R-squared: {r_squared:.4f}")

# ============================================================================
# STEP 6: HEXBIN PLOT
# ============================================================================

print("\n" + "="*80)
print("VISUALIZATION")
print("="*80)

print("\nCreating hexbin plot: AAPL vs AMZN daily returns (gridsize=30)")

plt.figure(figsize=(10, 8))
plt.hexbin(merged_df['AAPL_Return'], merged_df['AMZN_Return'],
           gridsize=30, cmap='YlOrRd', mincnt=1)
plt.colorbar(label='Count')
plt.xlabel('AAPL Daily Returns', fontsize=12)
plt.ylabel('AMZN Daily Returns', fontsize=12)
plt.title('Hexbin Plot: AAPL vs AMZN Daily Returns (2006-2017)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('hexbin_aapl_amzn_returns.png', dpi=300, bbox_inches='tight')
print("Hexbin plot saved as: hexbin_aapl_amzn_returns.png")

# ============================================================================
# STEP 7: INTERPRETATION
# ============================================================================

print("\n" + "="*80)
print("RIDGE REGRESSION INTERPRETATION")
print("="*80)

if abs(coef_aapl) > abs(coef_amzn):
    interpretation = f"AAPL exhibits a stronger predictive influence on AABA daily returns (coefficient: {coef_aapl:.6f}) compared to AMZN (coefficient: {coef_amzn:.6f})."
else:
    interpretation = f"AMZN exhibits a stronger predictive influence on AABA daily returns (coefficient: {coef_amzn:.6f}) compared to AAPL (coefficient: {coef_aapl:.6f})."

print(f"\n{interpretation}")

# ============================================================================
# SUMMARY REPORT
# ============================================================================

print("\n" + "="*80)
print("ANALYSIS SUMMARY REPORT")
print("="*80)

print(f"""
DATA PREPROCESSING:
- Total trading days analyzed: {len(merged_df)}
- Date range: {merged_df['Date'].min().strftime('%Y-%m-%d')} to {merged_df['Date'].max().strftime('%Y-%m-%d')}

GRANGER CAUSALITY TESTS (lag=5):
1. AAPL -> AMZN: F-statistic={f_stat_aapl_amzn:.4f}, p-value={p_value_aapl_amzn:.6f}
2. AMZN -> AABA: F-statistic={f_stat_amzn_aaba:.4f}, p-value={p_value_amzn_aaba:.6f}

STATIONARITY TESTS:
1. ADF Test (AAPL Returns): Statistic={adf_statistic:.4f}, p-value={adf_pvalue:.6f}
2. KPSS Test (AMZN Returns): Statistic={kpss_statistic:.4f}, p-value={kpss_pvalue:.6f}

COPULA AND RANK CORRELATION:
1. Gaussian Copula (AAPL & AABA): Correlation={copula_correlation:.4f}
2. Kendall's Tau (AMZN & AABA): Tau={kendall_tau:.4f}, p-value={kendall_pvalue:.6f}

RIDGE REGRESSION (alpha=1.0):
- AAPL coefficient: {coef_aapl:.6f}
- AMZN coefficient: {coef_amzn:.6f}
- R-squared: {r_squared:.4f}

INTERPRETATION:
{interpretation}
""")

print("="*80)
print("ANALYSIS COMPLETE")
print("="*80)
