"""
Cryptocurrency Time Series Analysis: SARIMAX and Granger Causality
Analyzing Aave, Polkadot, and Uniswap
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import grangercausalitytests
import warnings
warnings.filterwarnings('ignore')

# Load the datasets
print("="*80)
print("CRYPTOCURRENCY TIME SERIES ANALYSIS")
print("="*80)

# Load data
aave = pd.read_csv('coin_Aave.csv')
polkadot = pd.read_csv('coin_Polkadot.csv')
uniswap = pd.read_csv('coin_Uniswap.csv')

# Convert Date columns to datetime
aave['Date'] = pd.to_datetime(aave['Date'])
polkadot['Date'] = pd.to_datetime(polkadot['Date'])
uniswap['Date'] = pd.to_datetime(uniswap['Date'])

# Select relevant columns
aave_subset = aave[['Date', 'Close']].rename(columns={'Close': 'Close_AAVE'})
polkadot_subset = polkadot[['Date', 'Close']].rename(columns={'Close': 'Close_DOT'})
uniswap_subset = uniswap[['Date', 'Close']].rename(columns={'Close': 'Close_UNI'})

print("\n1. MERGING DATASETS ON DATE (INNER JOIN)")
print("-"*80)

# Merge on Date using inner join
merged = aave_subset.merge(polkadot_subset, on='Date', how='inner')
merged = merged.merge(uniswap_subset, on='Date', how='inner')

# Sort by Date
merged = merged.sort_values('Date').reset_index(drop=True)

print(f"Original dataset sizes:")
print(f"  Aave: {len(aave)} rows")
print(f"  Polkadot: {len(polkadot)} rows")
print(f"  Uniswap: {len(uniswap)} rows")
print(f"\nMerged dataset: {len(merged)} rows")
print(f"Date range: {merged['Date'].min()} to {merged['Date'].max()}")

# Drop any rows with missing values
merged_clean = merged.dropna()
print(f"After dropping missing values: {len(merged_clean)} rows")

# Prepare SARIMAX data (raw Close prices)
print("\n2. PREPARING DATA FOR SARIMAX MODELING")
print("-"*80)

# Dependent variable: Aave Close price
y_sarimax = merged_clean['Close_AAVE'].values

# Exogenous variables: Polkadot and Uniswap Close prices
X_sarimax = merged_clean[['Close_DOT', 'Close_UNI']].values

print(f"Dependent variable (Aave Close): {len(y_sarimax)} observations")
print(f"Exogenous variables (DOT, UNI Close): shape {X_sarimax.shape}")

# Compute standardized log-returns
print("\n3. COMPUTING STANDARDIZED LOG-RETURNS")
print("-"*80)

def compute_standardized_log_returns(prices):
    """
    Compute standardized log-returns:
    1. Take natural logarithm
    2. Difference consecutive values
    3. Remove first observation
    4. Apply z-score normalization with ddof=1
    """
    # Step 1: Natural log
    log_prices = np.log(prices)

    # Step 2: Difference
    log_returns = np.diff(log_prices)

    # Step 3: First observation already removed by diff

    # Step 4: Z-score normalization with sample std (ddof=1)
    mean_lr = np.mean(log_returns)
    std_lr = np.std(log_returns, ddof=1)
    standardized_lr = (log_returns - mean_lr) / std_lr

    return standardized_lr

# Compute for each cryptocurrency
aave_log_returns = compute_standardized_log_returns(merged_clean['Close_AAVE'].values)
dot_log_returns = compute_standardized_log_returns(merged_clean['Close_DOT'].values)
uni_log_returns = compute_standardized_log_returns(merged_clean['Close_UNI'].values)

print(f"Aave standardized log-returns: {len(aave_log_returns)} observations")
print(f"  Mean: {np.mean(aave_log_returns):.6f}")
print(f"  Std (ddof=1): {np.std(aave_log_returns, ddof=1):.6f}")

print(f"\nPolkadot standardized log-returns: {len(dot_log_returns)} observations")
print(f"  Mean: {np.mean(dot_log_returns):.6f}")
print(f"  Std (ddof=1): {np.std(dot_log_returns, ddof=1):.6f}")

print(f"\nUniswap standardized log-returns: {len(uni_log_returns)} observations")
print(f"  Mean: {np.mean(uni_log_returns):.6f}")
print(f"  Std (ddof=1): {np.std(uni_log_returns, ddof=1):.6f}")

# Fit SARIMAX model
print("\n4. FITTING SARIMAX MODEL")
print("-"*80)
print("Model: SARIMAX(1,1,1)x(1,0,1,12)")
print("Dependent: Aave raw Close prices")
print("Exogenous: Polkadot and Uniswap raw Close prices")

# SARIMAX(1,1,1)x(1,0,1,12)
model = SARIMAX(
    y_sarimax,
    exog=X_sarimax,
    order=(1, 1, 1),
    seasonal_order=(1, 0, 1, 12),
    enforce_stationarity=False,
    enforce_invertibility=False
)

# Fit the model
results = model.fit(disp=False, maxiter=200)

print("\nSARIMAX Model Parameters (to 6 decimal places):")
print("-"*80)

# Extract parameters
params = results.params
param_names = results.param_names

# Store key parameters for later use
dot_coef = None
uni_coef = None
ar1_coef = None
ma1_coef = None
sar12_coef = None
sma12_coef = None
sigma2_value = None

# Find and display the key parameters
for i, name in enumerate(param_names):
    if name in ['x1', 'Close_DOT']:
        print(f"Close_DOT:        {params[i]:.6f}")
        dot_coef = params[i]
    elif name in ['x2', 'Close_UNI']:
        print(f"Close_UNI:        {params[i]:.6f}")
        uni_coef = params[i]
    elif name == 'ar.L1':
        print(f"AR(1):            {params[i]:.6f}")
        ar1_coef = params[i]
    elif name == 'ma.L1':
        print(f"MA(1):            {params[i]:.6f}")
        ma1_coef = params[i]
    elif name == 'ar.S.L12':
        print(f"Seasonal AR(12):  {params[i]:.6f}")
        sar12_coef = params[i]
    elif name == 'ma.S.L12':
        print(f"Seasonal MA(12):  {params[i]:.6f}")
        sma12_coef = params[i]
    elif name == 'sigma2':
        print(f"sigma2:           {params[i]:.6f}")
        sigma2_value = params[i]

# Granger causality test
print("\n5. GRANGER CAUSALITY TEST")
print("-"*80)
print("Null Hypothesis: Polkadot log-returns do NOT Granger-cause Uniswap log-returns")
print("Lag order: 5")

# Create dataframe for Granger test
# Format: first column is the response (UNI), subsequent columns are predictors (DOT)
granger_data = pd.DataFrame({
    'UNI_lr': uni_log_returns,
    'DOT_lr': dot_log_returns
})

# Perform Granger causality test
granger_results = grangercausalitytests(granger_data[['UNI_lr', 'DOT_lr']], maxlag=5, verbose=False)

# Extract results for lag 5
lag5_results = granger_results[5][0]
f_stat = lag5_results['ssr_ftest'][0]
p_value = lag5_results['ssr_ftest'][1]

print(f"\nF-statistic:      {f_stat:.3f}")
print(f"p-value:          {p_value:.5f}")

if p_value < 0.05:
    print(f"\nConclusion: Reject null hypothesis at 5% significance level.")
    print(f"Polkadot log-returns DO Granger-cause Uniswap log-returns.")
else:
    print(f"\nConclusion: Fail to reject null hypothesis at 5% significance level.")
    print(f"Polkadot log-returns do NOT Granger-cause Uniswap log-returns.")

# Generate hexbin plot
print("\n6. GENERATING HEXBIN PLOT")
print("-"*80)

plt.figure(figsize=(10, 8))
hexbin = plt.hexbin(dot_log_returns, uni_log_returns, gridsize=30, cmap='Blues', mincnt=1)
plt.colorbar(hexbin, label='Count')
plt.xlabel('Polkadot Standardized Log-Returns', fontsize=12)
plt.ylabel('Uniswap Standardized Log-Returns', fontsize=12)
plt.title('Hexbin Plot: Polkadot vs Uniswap Standardized Log-Returns', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('hexbin_polkadot_uniswap_logreturns.png', dpi=300, bbox_inches='tight')
print("Hexbin plot saved as 'hexbin_polkadot_uniswap_logreturns.png'")

# Trading insights
print("\n7. TRADING INSIGHTS")
print("="*80)

print("\nBased on SARIMAX Parameter Estimates:")
print("-"*40)

print(f"\n1. CROSS-ASSET RELATIONSHIPS:")
if dot_coef > 0:
    print(f"   - Polkadot has a POSITIVE relationship ({dot_coef:.6f}) with Aave")
    print(f"   - When DOT price increases by $1, Aave tends to increase by ${dot_coef:.4f}")
else:
    print(f"   - Polkadot has a NEGATIVE relationship ({dot_coef:.6f}) with Aave")
    print(f"   - When DOT price increases by $1, Aave tends to decrease by ${abs(dot_coef):.4f}")

if uni_coef > 0:
    print(f"   - Uniswap has a POSITIVE relationship ({uni_coef:.6f}) with Aave")
    print(f"   - When UNI price increases by $1, Aave tends to increase by ${uni_coef:.4f}")
else:
    print(f"   - Uniswap has a NEGATIVE relationship ({uni_coef:.6f}) with Aave")
    print(f"   - When UNI price increases by $1, Aave tends to decrease by ${abs(uni_coef):.4f}")

print(f"\n2. MOMENTUM ANALYSIS (AR coefficient):")
if abs(ar1_coef) > 0.5:
    print(f"   - Strong momentum effect ({ar1_coef:.6f})")
    if ar1_coef > 0:
        print(f"   - Positive price movements tend to persist")
    else:
        print(f"   - Mean reversion tendency (negative momentum)")
else:
    print(f"   - Moderate momentum effect ({ar1_coef:.6f})")

print(f"\n3. GRANGER CAUSALITY FINDINGS:")
if p_value < 0.05:
    print(f"   - Polkadot returns predict future Uniswap returns (F={f_stat:.3f}, p={p_value:.5f})")
    print(f"   - ACTIONABLE: Monitor DOT price movements for UNI trading signals")
    print(f"   - Lead-lag relationship exists with 5-day lag structure")
else:
    print(f"   - No significant predictive relationship from DOT to UNI (p={p_value:.5f})")
    print(f"   - DOT movements do not provide reliable signals for UNI")

print(f"\n4. PORTFOLIO DIVERSIFICATION:")
# Calculate correlation
correlation = np.corrcoef(dot_log_returns, uni_log_returns)[0, 1]
print(f"   - Correlation between DOT and UNI log-returns: {correlation:.4f}")
if abs(correlation) < 0.7:
    print(f"   - Moderate correlation suggests diversification benefits")
    print(f"   - Consider holding both assets to reduce portfolio volatility")
else:
    print(f"   - High correlation suggests limited diversification benefits")
    print(f"   - Assets tend to move together")

print(f"\n5. TRADING STRATEGIES:")
print(f"   a) Pairs Trading:")
if abs(correlation) > 0.6:
    print(f"      - DOT and UNI show moderate-to-high correlation ({correlation:.4f})")
    print(f"      - Potential for statistical arbitrage when spread deviates")
else:
    print(f"      - Lower correlation ({correlation:.4f}) limits pairs trading opportunities")

print(f"\n   b) Cross-Asset Momentum:")
if p_value < 0.05:
    print(f"      - Use DOT price signals to anticipate UNI movements")
    print(f"      - Consider 5-day lag structure in trading decisions")

print(f"\n   c) Hedging Strategy:")
print(f"      - Use DOT (coef: {dot_coef:.4f}) and UNI (coef: {uni_coef:.4f}) to hedge AAVE positions")
if dot_coef * uni_coef > 0:
    print(f"      - Both assets move in same direction relative to AAVE")
else:
    print(f"      - Assets provide natural hedge (opposite directions)")

print(f"\n6. RISK MANAGEMENT:")
print(f"   - Model residual variance (sigma2): {sigma2_value:.6f}")
print(f"   - Lower sigma2 indicates better model fit")
print(f"   - Use for position sizing and stop-loss calculations")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

# Summary statistics
print("\nSUMMARY STATISTICS:")
print(f"Dataset size: {len(merged_clean)} observations")
print(f"Date range: {merged_clean['Date'].min().strftime('%Y-%m-%d')} to {merged_clean['Date'].max().strftime('%Y-%m-%d')}")
print(f"\nModel: SARIMAX(1,1,1)x(1,0,1,12)")
print(f"Granger causality test: DOT -> UNI at lag 5")
print(f"Visualization: Hexbin plot saved")
