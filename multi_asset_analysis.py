#!/usr/bin/env python3
"""
Multi-Asset Volatility and Causality Investigation
Comprehensive analysis across equity, commodity, and fixed income asset classes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

# Set float precision
pd.options.display.float_format = '{:.10f}'.format

print("=" * 80)
print("MULTI-ASSET VOLATILITY AND CAUSALITY INVESTIGATION")
print("=" * 80)

# ============================================================================
# STEP 1: Data Merging and Log Return Calculation
# ============================================================================
print("\n[1] DATA MERGING AND PREPROCESSING")
print("-" * 80)

# Load datasets
mag7 = pd.read_csv('mag7.csv')
metals_etfs = pd.read_csv('metals_etfs.csv')
bond_etfs = pd.read_csv('bond_etfs.csv')

print(f"MAG7 dataset shape: {mag7.shape}")
print(f"Metals ETFs dataset shape: {metals_etfs.shape}")
print(f"Bond ETFs dataset shape: {bond_etfs.shape}")

# Merge on Date column (inner join)
merged = mag7.merge(metals_etfs, on='Date', how='inner')
merged = merged.merge(bond_etfs, on='Date', how='inner')
print(f"Merged dataset shape: {merged.shape}")

# Exclude rows where IBIT is missing
merged = merged[merged['IBIT'].notna()]
print(f"After excluding missing IBIT: {merged.shape}")

# Calculate log returns for all numeric columns
merged['Date'] = pd.to_datetime(merged['Date'])
merged = merged.sort_values('Date').reset_index(drop=True)

# Get all numeric columns except Date
numeric_cols = merged.select_dtypes(include=[np.number]).columns.tolist()

# Calculate log returns
for col in numeric_cols:
    merged[f'{col}_return'] = np.log(merged[col] / merged[col].shift(1)).astype('float64')

# Remove first row with undefined returns
merged = merged.iloc[1:].reset_index(drop=True)
print(f"After removing first row: {merged.shape}")

# Create bond_portfolio_return
bond_etf_cols = ['SGOV', 'SHV', 'SHY', 'BIL', 'IEI', 'IEF', 'VGIT', 'TLT', 'TLH', 'EDV', 'AGG', 'BND']
bond_return_cols = [f'{col}_return' for col in bond_etf_cols]
merged['bond_portfolio_return'] = merged[bond_return_cols].mean(axis=1).astype('float64')

print(f"Bond portfolio return created. Sample mean: {merged['bond_portfolio_return'].mean():.8f}")

# ============================================================================
# STEP 2: Propensity Score Matching
# ============================================================================
print("\n[2] PROPENSITY SCORE MATCHING ANALYSIS")
print("-" * 80)

# Define treatment variable
q75 = merged['bond_portfolio_return'].quantile(0.75, interpolation='linear')
q25 = merged['bond_portfolio_return'].quantile(0.25, interpolation='linear')

print(f"75th percentile of bond_portfolio_return: {q75:.8f}")
print(f"25th percentile of bond_portfolio_return: {q25:.8f}")

# Filter for treatment and control
psm_data = merged[
    (merged['bond_portfolio_return'] > q75) | (merged['bond_portfolio_return'] < q25)
].copy()

psm_data['treatment'] = (psm_data['bond_portfolio_return'] > q75).astype(int)
psm_data = psm_data.sort_values('Date').reset_index(drop=True)

print(f"Treatment group size: {psm_data['treatment'].sum()}")
print(f"Control group size: {(psm_data['treatment'] == 0).sum()}")

# Outcome and covariate
psm_data['outcome'] = psm_data['NVDA_return'].astype('float64')
psm_data['covariate'] = psm_data['GLD_return'].astype('float64')

# Fit logistic regression for propensity scores
X = psm_data[['covariate']].values
y = psm_data['treatment'].values

lr = LogisticRegression(solver='lbfgs', max_iter=1000)
lr.fit(X, y)
propensity_scores = lr.predict_proba(X)[:, 1]
psm_data['propensity_score'] = propensity_scores

# Calculate logit of propensity scores
logit_ps = np.log(propensity_scores / (1 - propensity_scores))
caliper = 0.20 * np.std(logit_ps)

print(f"Caliper width: {caliper:.8f}")

# One-to-one nearest neighbor matching without replacement
treated_indices = psm_data[psm_data['treatment'] == 1].index.tolist()
control_indices = psm_data[psm_data['treatment'] == 0].index.tolist()

matched_pairs = []
used_controls = set()

for treated_idx in treated_indices:
    treated_logit = np.log(psm_data.loc[treated_idx, 'propensity_score'] /
                          (1 - psm_data.loc[treated_idx, 'propensity_score']))

    min_distance = float('inf')
    best_control = None

    for control_idx in control_indices:
        if control_idx in used_controls:
            continue

        control_logit = np.log(psm_data.loc[control_idx, 'propensity_score'] /
                              (1 - psm_data.loc[control_idx, 'propensity_score']))
        distance = abs(treated_logit - control_logit)

        if distance <= caliper:
            # If distances are nearly identical (within 1e-10), prefer smaller index
            if abs(distance - min_distance) < 1e-10:
                if control_idx < best_control:
                    best_control = control_idx
                    min_distance = distance
            elif distance < min_distance:
                min_distance = distance
                best_control = control_idx

    if best_control is not None:
        matched_pairs.append((treated_idx, best_control))
        used_controls.add(best_control)

num_matched_pairs = len(matched_pairs)
print(f"Number of successfully matched pairs: {num_matched_pairs}")

# Calculate ATT
if num_matched_pairs > 0:
    treated_outcomes = [psm_data.loc[t, 'outcome'] for t, c in matched_pairs]
    control_outcomes = [psm_data.loc[c, 'outcome'] for t, c in matched_pairs]
    att = np.mean(treated_outcomes) - np.mean(control_outcomes)
    print(f"Average Treatment Effect on the Treated (ATT): {att:.4f}")
else:
    print("No matched pairs found")

# ============================================================================
# STEP 3: ARIMA Forecasting on TLT
# ============================================================================
print("\n[3] ARIMA(2,0,2) FORECASTING ON TLT")
print("-" * 80)

tlt_returns = merged['TLT_return'].dropna().values
n_obs = len(tlt_returns)
train_size = int(np.floor(0.75 * n_obs))

train_data = tlt_returns[:train_size]
validation_data = tlt_returns[train_size:]

print(f"Total observations: {n_obs}")
print(f"Training set size: {train_size}")
print(f"Validation set size: {len(validation_data)}")

# Fit ARIMA(2,0,2)
arima_model = ARIMA(train_data, order=(2, 0, 2))
arima_fit = arima_model.fit(method_kwargs={'maxiter': 500})

# Generate forecasts
forecasts = arima_fit.forecast(steps=len(validation_data))

# Calculate MAPE (avoid division by zero)
# Only calculate MAPE for non-zero actual values
non_zero_mask = np.abs(validation_data) > 1e-10
if non_zero_mask.sum() > 0:
    mape_values = np.abs((validation_data[non_zero_mask] - forecasts[non_zero_mask]) /
                         validation_data[non_zero_mask]) * 100
    mape = np.mean(mape_values)
else:
    # If all values are near zero, use mean absolute error instead
    mape = np.mean(np.abs(validation_data - forecasts)) * 100

print(f"Mean Absolute Percentage Error (MAPE): {mape:.3f}%")

# ============================================================================
# STEP 4: Mixed Effects Model with Volatility Regime
# ============================================================================
print("\n[4] MIXED EFFECTS MODEL WITH VOLATILITY REGIME")
print("-" * 80)

# Calculate rolling 20-day std of SLV returns
mixed_data = merged.copy()
mixed_data['slv_rolling_std'] = mixed_data['SLV_return'].rolling(
    window=20, min_periods=20
).std()

# Exclude first 19 observations
mixed_data = mixed_data.iloc[19:].reset_index(drop=True)
print(f"After excluding first 19 observations: {mixed_data.shape}")

# Create volatility_regime
median_std = mixed_data['slv_rolling_std'].median()
mixed_data['volatility_regime'] = np.where(
    mixed_data['slv_rolling_std'] > median_std, 'high', 'low'
)

print(f"Median rolling std: {median_std:.8f}")
print(f"High regime count: {(mixed_data['volatility_regime'] == 'high').sum()}")
print(f"Low regime count: {(mixed_data['volatility_regime'] == 'low').sum()}")

# Create year-month grouping
mixed_data['year_month'] = mixed_data['Date'].dt.strftime('%Y-%m')

# Prepare data for mixed effects model
mixed_data['META_return_clean'] = mixed_data['META_return'].astype('float64')

# Create design matrix with intercept
# We want 'low' as reference, so create dummies and drop 'low'
exog_df = pd.get_dummies(mixed_data['volatility_regime'], drop_first=False, dtype='float64')
if 'low' in exog_df.columns:
    exog_df = exog_df.drop('low', axis=1)
exog_df.insert(0, 'Intercept', 1.0)

# Convert to proper dtypes
endog = mixed_data['META_return_clean'].values.astype('float64')
exog = exog_df.values.astype('float64')
groups = mixed_data['year_month'].values

# Fit mixed effects model with REML
model = MixedLM(
    endog=endog,
    exog=exog,
    groups=groups
)

result = model.fit(reml=True)

# Extract fixed effect coefficient for 'high'
# The column name after drop_first=True should be 'high' since 'low' is dropped
# Index 0 is Intercept, Index 1 is 'high' coefficient
exog_names = exog_df.columns.tolist()
print(f"Exog variable names: {exog_names}")

if len(result.params) > 1:
    fixed_effect_high = result.params[1]  # 'high' is the second parameter
else:
    # Fallback: search by name
    param_names = result.params.index.tolist()
    high_param_name = [p for p in param_names if 'high' in p.lower()][0]
    fixed_effect_high = result.params[high_param_name]

print(f"Fixed effect coefficient for volatility_regime[high]: {fixed_effect_high:.5f}")

# Calculate ICC
if isinstance(result.cov_re, np.ndarray):
    random_effect_var = float(result.cov_re.item())
else:
    random_effect_var = float(result.cov_re.iloc[0, 0])
residual_var = result.scale
total_var = random_effect_var + residual_var
icc = random_effect_var / total_var

print(f"Random effect variance: {random_effect_var:.8f}")
print(f"Residual variance: {residual_var:.8f}")
print(f"Intraclass Correlation Coefficient (ICC): {icc:.5f}")

# ============================================================================
# STEP 5: Granger Causality Test
# ============================================================================
print("\n[5] GRANGER CAUSALITY TEST: AAPL -> AGG")
print("-" * 80)

# Prepare data for Granger test
granger_data = merged[['AAPL_return', 'AGG_return']].dropna()
granger_data.columns = ['AAPL', 'AGG']

# Perform Granger causality test with lag order 3
gc_result = grangercausalitytests(granger_data[['AGG', 'AAPL']], maxlag=3, verbose=False)

# Extract F-statistic and p-value for lag 3
f_stat = gc_result[3][0]['ssr_ftest'][0]
p_value = gc_result[3][0]['ssr_ftest'][1]

print(f"Granger Causality F-statistic (lag=3): {f_stat:.5f}")
print(f"Granger Causality p-value: {p_value:.6f}")

# ============================================================================
# STEP 6: Anderson-Darling Normality Test
# ============================================================================
print("\n[6] ANDERSON-DARLING TEST FOR NORMALITY ON TSLA")
print("-" * 80)

tsla_returns = merged['TSLA_return'].dropna().values

# Perform Anderson-Darling test
ad_result = stats.anderson(tsla_returns, dist='norm')
ad_statistic = ad_result.statistic

print(f"Anderson-Darling test statistic: {ad_statistic:.5f}")

# ============================================================================
# STEP 7: Hexbin Plot
# ============================================================================
print("\n[7] GENERATING HEXBIN PLOT")
print("-" * 80)

plot_data = merged[['BND_return', 'MSFT_return']].dropna()

plt.figure(figsize=(10, 8))
plt.hexbin(
    plot_data['BND_return'],
    plot_data['MSFT_return'],
    gridsize=30,
    cmap='viridis',
    mincnt=1
)
plt.colorbar(label='Count')
plt.xlabel('BND Log Returns', fontsize=12)
plt.ylabel('MSFT Log Returns', fontsize=12)
plt.title('Hexbin Plot: BND vs MSFT Log Returns', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('hexbin_bnd_msft.png', dpi=300, bbox_inches='tight')
print("Hexbin plot saved as: hexbin_bnd_msft.png")

# ============================================================================
# SUMMARY OF RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY OF RESULTS")
print("=" * 80)
print(f"1. ATT (Propensity Score Matching): {att:.4f}")
print(f"2. Number of Matched Pairs: {num_matched_pairs}")
print(f"3. ARIMA MAPE on TLT Validation Set: {mape:.3f}%")
print(f"4. Fixed Effect Coefficient (volatility_regime[high]): {fixed_effect_high:.5f}")
print(f"5. Intraclass Correlation Coefficient (ICC): {icc:.5f}")
print(f"6. Granger Causality F-statistic (AAPL->AGG, lag=3): {f_stat:.5f}")
print(f"7. Granger Causality p-value: {p_value:.6f}")
print(f"8. Anderson-Darling Test Statistic (TSLA): {ad_statistic:.5f}")
print("=" * 80)

print("\nAnalysis completed successfully!")
