"""
Tail-Risk Dynamics and Distributional Dependencies Analysis
Quantitative analysis of equity market datasets using EVT, stationarity tests,
and multivariate dependency measures.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import genpareto, kendalltau
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import kpss
import warnings
warnings.filterwarnings('ignore')

# Load datasets
print("="*80)
print("TAIL-RISK EQUITY ANALYSIS: 2006-2018")
print("="*80)

msft = pd.read_csv('MSFT_2006-01-01_to_2018-01-01.csv')
ibm = pd.read_csv('IBM_2006-01-01_to_2018-01-01.csv')
nke = pd.read_csv('NKE_2006-01-01_to_2018-01-01.csv')

print("\nDatasets loaded:")
print(f"  MSFT: {len(msft)} observations")
print(f"  IBM:  {len(ibm)} observations")
print(f"  NKE:  {len(nke)} observations")

# ============================================================================
# 1. EXTREME VALUE THEORY: Generalized Pareto Distribution for IBM Right Tail
# ============================================================================
print("\n" + "="*80)
print("1. EXTREME VALUE THEORY: GPD Fit for IBM Right Tail")
print("="*80)

# Calculate IBM daily returns
ibm_returns = 100 * (ibm['Close'].diff() / ibm['Close'].shift(1))
ibm_returns = ibm_returns.dropna()

print(f"\nIBM Daily Returns Statistics:")
print(f"  Mean:     {ibm_returns.mean():.4f}%")
print(f"  Std Dev:  {ibm_returns.std():.4f}%")
print(f"  Skewness: {ibm_returns.skew():.4f}")
print(f"  Kurtosis: {ibm_returns.kurtosis():.4f}")

# Calculate 95th percentile threshold
threshold_95 = np.percentile(ibm_returns, 95)
print(f"\n95th Percentile Threshold: {threshold_95:.4f}%")

# Extract exceedances above threshold
exceedances = ibm_returns[ibm_returns > threshold_95] - threshold_95
print(f"Number of exceedances: {len(exceedances)}")

# Fit Generalized Pareto Distribution using MLE
gpd_shape, gpd_loc, gpd_scale = genpareto.fit(exceedances, floc=0)

print(f"\nGeneralized Pareto Distribution Parameters:")
print(f"  Shape parameter (xi):    {gpd_shape:.4f}")
print(f"  Scale parameter (sigma): {gpd_scale:.4f}")

gpd_xi = gpd_shape
gpd_sigma = gpd_scale

# ============================================================================
# 2. MULTIVARIATE DEPENDENCY: Kendall Tau-b for NKE
# ============================================================================
print("\n" + "="*80)
print("2. MULTIVARIATE DEPENDENCY STRUCTURE: Kendall Tau-b (NKE)")
print("="*80)

# Compute Kendall tau-b between Volume and Close for NKE
kendall_result = kendalltau(nke['Volume'], nke['Close'], variant='b')
kendall_tau = kendall_result.statistic
kendall_pvalue = kendall_result.pvalue

print(f"\nNKE Volume vs Close Price:")
print(f"  Kendall tau-b coefficient: {kendall_tau:.4f}")
print(f"  Two-tailed p-value:        {kendall_pvalue:.4f}")
print(f"  Significance: {'***' if kendall_pvalue < 0.001 else '**' if kendall_pvalue < 0.01 else '*' if kendall_pvalue < 0.05 else 'ns'}")

# ============================================================================
# 3. STATIONARITY ASSESSMENT: KPSS Test for MSFT
# ============================================================================
print("\n" + "="*80)
print("3. TIME SERIES STATIONARITY: KPSS Test (MSFT)")
print("="*80)

# Calculate lag parameter: int(4 * (n^0.25) / 100)
n = len(msft)
lag_param = int(4 * (n ** 0.25) / 100)
print(f"\nSample size: {n}")
print(f"Lag parameter: {lag_param}")

# Perform KPSS test with constant and trend
kpss_stat, kpss_pvalue, kpss_lags, kpss_crit = kpss(msft['Close'],
                                                      regression='ct',
                                                      nlags=lag_param)

print(f"\nKPSS Test Results (constant and trend):")
print(f"  Test statistic: {kpss_stat:.4f}")
print(f"  P-value:        {kpss_pvalue:.4f}")
print(f"  Critical values: {kpss_crit}")
print(f"  Interpretation: {'Non-stationary (reject H0)' if kpss_pvalue < 0.05 else 'Stationary (fail to reject H0)'}")

# ============================================================================
# 4. TAIL RISK QUANTIFICATION: Percentile Ratios
# ============================================================================
print("\n" + "="*80)
print("4. TAIL RISK QUANTIFICATION: Percentile Ratios")
print("="*80)

def calculate_tail_risk_ratio(df, name):
    """Calculate tail risk ratio: P90 / |P10| for returns"""
    returns = 100 * (df['Close'].diff() / df['Close'].shift(1))
    returns = returns.dropna()

    p90 = np.percentile(returns, 90, method='linear')
    p10 = np.percentile(returns, 10, method='linear')

    ratio = p90 / abs(p10)

    print(f"\n{name}:")
    print(f"  90th percentile: {p90:.4f}%")
    print(f"  10th percentile: {p10:.4f}%")
    print(f"  Tail risk ratio (P90/|P10|): {ratio:.4f}")

    return ratio, returns

msft_ratio, msft_returns = calculate_tail_risk_ratio(msft, "MSFT")
ibm_ratio, _ = calculate_tail_risk_ratio(ibm, "IBM")
nke_ratio, nke_returns = calculate_tail_risk_ratio(nke, "NKE")

# Identify highest tail risk
ratios = {'MSFT': msft_ratio, 'IBM': ibm_ratio, 'NKE': nke_ratio}
highest_risk_stock = max(ratios, key=ratios.get)
highest_risk_value = ratios[highest_risk_stock]

print(f"\n{'='*80}")
print(f"HIGHEST TAIL RISK RATIO: {highest_risk_stock}")
print(f"Value: {highest_risk_value:.4f}")
print(f"{'='*80}")

# ============================================================================
# 5. AUTOCORRELATION STRUCTURE: Ljung-Box Test for IBM
# ============================================================================
print("\n" + "="*80)
print("5. VOLATILITY CLUSTERING: Ljung-Box Test on Squared Returns (IBM)")
print("="*80)

# Calculate squared returns for IBM
ibm_returns_calc = 100 * (ibm['Close'].diff() / ibm['Close'].shift(1))
ibm_returns_calc = ibm_returns_calc.dropna()
ibm_squared_returns = ibm_returns_calc ** 2

# Ljung-Box test with lag 10
lb_result = acorr_ljungbox(ibm_squared_returns, lags=[10], return_df=True)
lb_stat = lb_result['lb_stat'].values[0]  # Q-statistic for lag 10
lb_pvalue = lb_result['lb_pvalue'].values[0]  # p-value for lag 10

print(f"\nLjung-Box Test on Squared Returns (lag=10):")
print(f"  Q-statistic: {lb_stat:.4f}")
print(f"  P-value:     {lb_pvalue:.4f}")
print(f"  Interpretation: {'Significant volatility clustering (reject H0)' if lb_pvalue < 0.05 else 'No significant clustering (fail to reject H0)'}")

# ============================================================================
# 6. HEXBIN VISUALIZATION: Standardized Volume vs Close
# ============================================================================
print("\n" + "="*80)
print("6. HEXBIN VISUALIZATION: Standardized Volume vs Close")
print("="*80)

# Combine all datasets
msft_data = msft[['Close', 'Volume']].copy()
msft_data['Stock'] = 'MSFT'

ibm_data = ibm[['Close', 'Volume']].copy()
ibm_data['Stock'] = 'IBM'

nke_data = nke[['Close', 'Volume']].copy()
nke_data['Stock'] = 'NKE'

combined = pd.concat([msft_data, ibm_data, nke_data], ignore_index=True)

# Z-score standardization
combined['Close_std'] = (combined['Close'] - combined['Close'].mean()) / combined['Close'].std()
combined['Volume_std'] = (combined['Volume'] - combined['Volume'].mean()) / combined['Volume'].std()

print(f"\nCombined dataset: {len(combined)} observations")
print(f"Standardized Close - Mean: {combined['Close_std'].mean():.6f}, Std: {combined['Close_std'].std():.6f}")
print(f"Standardized Volume - Mean: {combined['Volume_std'].mean():.6f}, Std: {combined['Volume_std'].std():.6f}")

# Create hexbin plot
plt.figure(figsize=(12, 8))
hexbin = plt.hexbin(combined['Close_std'], combined['Volume_std'],
                     gridsize=20, cmap='YlOrRd', mincnt=1)
plt.colorbar(hexbin, label='Count of Observations per Hexagonal Cell')
plt.xlabel('Standardized Close Price (z-score)', fontsize=12)
plt.ylabel('Standardized Volume (z-score)', fontsize=12)
plt.title('Hexbin Plot: Standardized Volume vs Standardized Close Price\n(MSFT, IBM, NKE Combined: 2006-2018)',
          fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('hexbin_standardized_volume_close.png', dpi=300, bbox_inches='tight')
print("\nHexbin plot saved as: hexbin_standardized_volume_close.png")

# ============================================================================
# 7. INTERPRETATION: Tail Heaviness and Downside Risk
# ============================================================================
print("\n" + "="*80)
print("7. INTERPRETATION: Tail Heaviness and Portfolio Allocation")
print("="*80)

print(f"""
TAIL HEAVINESS ASSESSMENT:

1. GPD Shape Parameter (IBM):
   Xi = {gpd_xi:.4f}

   Interpretation: {'The shape parameter is positive, indicating a heavy-tailed distribution' if gpd_xi > 0 else 'The shape parameter suggests finite tail behavior'}
   {'with infinite variance (xi > 0.5), suggesting extreme tail events are probable.' if gpd_xi > 0.5 else 'but with finite moments, suggesting moderate tail risk.' if gpd_xi > 0 else ''}

2. Tail Risk Ratios (P90/|P10|):
   MSFT: {msft_ratio:.4f}
   IBM:  {ibm_ratio:.4f}
   NKE:  {nke_ratio:.4f}

   Highest Tail Risk: {highest_risk_stock} ({highest_risk_value:.4f})

   The tail risk ratio measures asymmetry between upside and downside movements.
   A ratio > 1 indicates stronger upside extremes than downside extremes.
   A ratio < 1 indicates stronger downside risk (more negative tail events).

COMPARATIVE DOWNSIDE RISK PROFILES:

Portfolio Allocation Implications During Market Stress:

1. **{highest_risk_stock}** exhibits the {'highest' if highest_risk_value == max(ratios.values()) else 'lowest'} tail asymmetry ratio ({highest_risk_value:.4f}),
   suggesting {'greater upside potential but also tail risk' if highest_risk_value > 1 else 'more pronounced downside risk in stress scenarios'}.

2. The GPD shape parameter of {gpd_xi:.4f} for IBM indicates {'heavy tails with substantial' if gpd_xi > 0.3 else 'moderate'}
   probability of extreme positive returns beyond the 95th percentile.

3. Volatility Clustering (Ljung-Box Q={lb_stat:.4f}, p={lb_pvalue:.4f}):
   {'Strong evidence of volatility clustering suggests return volatility is predictable' if lb_pvalue < 0.01 else 'Moderate evidence suggests some volatility persistence'}
   and risk is time-varying, requiring dynamic hedging strategies.

4. **Portfolio Recommendations:**
   - During market stress, stocks with lower tail risk ratios offer better downside protection
   - The positive GPD shape parameter suggests extreme events are more likely than normal distribution predicts
   - Diversification across these three stocks should account for varying tail behaviors
   - Risk parity or tail-risk parity allocation may be more appropriate than traditional mean-variance optimization

5. **Kendall Tau-b (NKE Volume-Price) = {kendall_tau:.4f}**:
   {'Significant positive rank correlation suggests volume increases with price' if kendall_tau > 0 and kendall_pvalue < 0.05 else 'Weak correlation between volume and price'}
   {'during trending markets, providing liquidity when needed.' if kendall_tau > 0 and kendall_pvalue < 0.05 else '.'}
""")

# ============================================================================
# SUMMARY REPORT
# ============================================================================
print("\n" + "="*80)
print("SUMMARY OF RESULTS")
print("="*80)

print(f"""
1. GPD Parameters (IBM Right Tail):
   Shape (xi):    {gpd_xi:.4f}
   Scale (sigma): {gpd_sigma:.4f}

2. Kendall Tau-b (NKE Volume vs Close):
   Coefficient:   {kendall_tau:.4f}
   P-value:       {kendall_pvalue:.4f}

3. KPSS Test (MSFT Close):
   Statistic:     {kpss_stat:.4f}
   P-value:       {kpss_pvalue:.4f}

4. Tail Risk Ratios (P90/|P10|):
   Highest Risk Stock: {highest_risk_stock}
   Ratio Value:        {highest_risk_value:.4f}

5. Ljung-Box Test (IBM Squared Returns, lag=10):
   Q-statistic:   {lb_stat:.4f}
   P-value:       {lb_pvalue:.4f}

6. Visualization:
   Hexbin plot saved: hexbin_standardized_volume_close.png
   Grid size: 20x20 (approximately 400 hexagonal bins)
""")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
