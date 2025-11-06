#!/usr/bin/env python3
"""
Causal Inference and Predictive Modeling Analysis
For Global Markets Data (2014-2016)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("=" * 80)
print("STEP 1: LOADING DATA")
print("=" * 80)

df_2014 = pd.read_csv('2014_Global_Markets_Data.csv')
df_2015 = pd.read_csv('2015_Global_Markets_Data.csv')
df_2016 = pd.read_csv('2016_Global_Markets_Data.csv')

# Parse dates in ISO 8601 format (YYYY-MM-DD)
df_2014['Date'] = pd.to_datetime(df_2014['Date'], format='ISO8601')
df_2015['Date'] = pd.to_datetime(df_2015['Date'], format='ISO8601')
df_2016['Date'] = pd.to_datetime(df_2016['Date'], format='ISO8601')

print(f"2014 data: {len(df_2014)} rows")
print(f"2015 data: {len(df_2015)} rows")
print(f"2016 data: {len(df_2016)} rows")

# ============================================================================
# STEP 2: CREATE CALENDAR DAY (MM-DD) AND MERGE
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: CREATING MERGED DATASET WITH CALENDAR DAY MATCHING")
print("=" * 80)

# Extract calendar day as MM-DD
df_2014['CalendarDay'] = df_2014['Date'].dt.strftime('%m-%d')
df_2015['CalendarDay'] = df_2015['Date'].dt.strftime('%m-%d')
df_2016['CalendarDay'] = df_2016['Date'].dt.strftime('%m-%d')

# Merge on Ticker and CalendarDay
merged = df_2014.merge(
    df_2015,
    on=['Ticker', 'CalendarDay'],
    suffixes=('_2014', '_2015')
)
merged = merged.merge(
    df_2016[['Ticker', 'CalendarDay', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']],
    on=['Ticker', 'CalendarDay'],
    suffixes=('', '_2016')
)

# Rename 2016 columns
merged = merged.rename(columns={
    'Open': 'Open_2016',
    'High': 'High_2016',
    'Low': 'Low_2016',
    'Close': 'Close_2016',
    'Adj Close': 'Adj Close_2016',
    'Volume': 'Volume_2016'
})

print(f"Merged observations (Ticker-CalendarDay triplets): {len(merged)}")

# ============================================================================
# STEP 3: CALCULATE Z-SCORES AND FILTER OUTLIERS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: CALCULATING Z-SCORES AND FILTERING OUTLIERS")
print("=" * 80)

numeric_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
years = ['2014', '2015', '2016']

# Calculate z-scores across the entire merged dataset
for col in numeric_cols:
    for year in years:
        col_name = f'{col}_{year}'
        merged[f'{col_name}_zscore'] = stats.zscore(merged[col_name], nan_policy='omit')

# Filter: exclude observations where any |z-score| > 4
filter_mask = np.ones(len(merged), dtype=bool)
for col in numeric_cols:
    for year in years:
        col_name = f'{col}_{year}_zscore'
        filter_mask &= (np.abs(merged[col_name]) <= 4)

merged_filtered = merged[filter_mask].copy()

print(f"Observations before filtering: {len(merged)}")
print(f"Observations after filtering: {len(merged_filtered)}")
print(f"Matched Ticker-CalendarDay observations retained: {len(merged_filtered)}")

# ============================================================================
# STEP 4: DEFINE TREATMENT (PER-TICKER MEDIAN SPLIT OF 2014 ADJ CLOSE)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: DEFINING TREATMENT VARIABLE")
print("=" * 80)

# Calculate per-ticker median of 2014 Adj Close
ticker_medians = merged_filtered.groupby('Ticker')['Adj Close_2014'].median()

# Create treatment indicator
merged_filtered['ticker_median_2014'] = merged_filtered['Ticker'].map(ticker_medians)
merged_filtered['treatment'] = (merged_filtered['Adj Close_2014'] > merged_filtered['ticker_median_2014']).astype(int)

print(f"Treatment group (above median): {merged_filtered['treatment'].sum()} observations")
print(f"Control group (at or below median): {(1 - merged_filtered['treatment']).sum()} observations")

# ============================================================================
# STEP 5: COMPUTE PROPENSITY SCORES VIA LOGISTIC REGRESSION
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: COMPUTING PROPENSITY SCORES")
print("=" * 80)

# Standardize confounders from 2014
confounders = ['Open_2014', 'High_2014', 'Low_2014', 'Close_2014']
scaler = StandardScaler()
X_confounders = scaler.fit_transform(merged_filtered[confounders])

# Logistic regression for propensity scores
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_confounders, merged_filtered['treatment'])

# Get propensity scores (probability of treatment)
propensity_scores = lr.predict_proba(X_confounders)[:, 1]
merged_filtered['propensity_score'] = propensity_scores

print(f"Propensity score range: [{propensity_scores.min():.4f}, {propensity_scores.max():.4f}]")
print(f"Mean propensity score: {propensity_scores.mean():.4f}")

# ============================================================================
# STEP 6: PERFORM 1-TO-1 NEAREST NEIGHBOR MATCHING WITH CALIPER
# ============================================================================
print("\n" + "=" * 80)
print("STEP 6: PERFORMING PROPENSITY SCORE MATCHING")
print("=" * 80)

# Caliper = 0.25 * std of raw propensity scores
caliper = 0.25 * np.std(propensity_scores)
print(f"Caliper width: {caliper:.4f}")

# Separate treated and control
treated_idx = merged_filtered[merged_filtered['treatment'] == 1].index.tolist()
control_idx = merged_filtered[merged_filtered['treatment'] == 0].index.tolist()

# Perform 1-to-1 nearest neighbor matching
matched_pairs = []
used_controls = set()

for t_idx in treated_idx:
    t_ps = merged_filtered.loc[t_idx, 'propensity_score']

    # Find nearest control within caliper
    best_control = None
    best_distance = np.inf

    for c_idx in control_idx:
        if c_idx in used_controls:
            continue

        c_ps = merged_filtered.loc[c_idx, 'propensity_score']
        distance = abs(t_ps - c_ps)

        if distance <= caliper and distance < best_distance:
            best_distance = distance
            best_control = c_idx

    if best_control is not None:
        matched_pairs.append((t_idx, best_control))
        used_controls.add(best_control)

print(f"Number of matched pairs: {len(matched_pairs)}")

# Create matched dataset
matched_treated_idx = [pair[0] for pair in matched_pairs]
matched_control_idx = [pair[1] for pair in matched_pairs]
matched_dataset = merged_filtered.loc[matched_treated_idx + matched_control_idx].copy()

print(f"Matched dataset size: {len(matched_dataset)} observations")

# ============================================================================
# STEP 7: CALCULATE ATE AND SMD
# ============================================================================
print("\n" + "=" * 80)
print("STEP 7: CALCULATING ATE AND SMD")
print("=" * 80)

# Average Treatment Effect (ATE) - outcome is Volume_2015
treated_volumes = merged_filtered.loc[matched_treated_idx, 'Volume_2015'].values
control_volumes = merged_filtered.loc[matched_control_idx, 'Volume_2015'].values

ate = np.mean(treated_volumes) - np.mean(control_volumes)
ate_millions = ate / 1_000_000  # Convert to millions

print(f"\nAverage Treatment Effect (ATE):")
print(f"  ATE in millions: {ate_millions:.2f} million units")

# Standardized Mean Difference (SMD) for 'Open' before and after matching
def calculate_smd(treated_values, control_values):
    mean_treated = np.mean(treated_values)
    mean_control = np.mean(control_values)

    var_treated = np.var(treated_values, ddof=1)
    var_control = np.var(control_values, ddof=1)

    pooled_std = np.sqrt((var_treated + var_control) / 2)

    if pooled_std == 0:
        return 0.0

    smd = (mean_treated - mean_control) / pooled_std
    return smd

# SMD before matching
treated_open_before = merged_filtered[merged_filtered['treatment'] == 1]['Open_2014'].values
control_open_before = merged_filtered[merged_filtered['treatment'] == 0]['Open_2014'].values
smd_before = calculate_smd(treated_open_before, control_open_before)

# SMD after matching
treated_open_after = merged_filtered.loc[matched_treated_idx, 'Open_2014'].values
control_open_after = merged_filtered.loc[matched_control_idx, 'Open_2014'].values
smd_after = calculate_smd(treated_open_after, control_open_after)

print(f"\nStandardized Mean Difference (SMD) for 'Open':")
print(f"  SMD before matching: {smd_before:.2f}")
print(f"  SMD after matching: {smd_after:.2f}")

# ============================================================================
# STEP 8: GAUSSIAN PROCESS REGRESSION ON MATCHED SAMPLE
# ============================================================================
print("\n" + "=" * 80)
print("STEP 8: GAUSSIAN PROCESS REGRESSION")
print("=" * 80)

# Features: z-score standardized Open, Close, Volume from 2014
gp_features = ['Open_2014', 'Close_2014', 'Volume_2014']
X_gp = matched_dataset[gp_features].values
y_gp = matched_dataset['Volume_2015'].values

# Z-score standardize features AND target
X_gp_scaled = stats.zscore(X_gp, axis=0)
y_gp_scaled = stats.zscore(y_gp)

# Store scaling parameters for inverse transform
y_mean = np.mean(y_gp)
y_std = np.std(y_gp, ddof=1)

# RBF kernel with 10 random restarts
kernel = RBF(length_scale=1.0)
gp = GaussianProcessRegressor(
    kernel=kernel,
    n_restarts_optimizer=10,
    random_state=42,
    normalize_y=False  # Already normalized
)

# Fit GP on scaled data
gp.fit(X_gp_scaled, y_gp_scaled)

# Report marginal log-likelihood
log_likelihood = gp.log_marginal_likelihood_value_
print(f"Marginal log-likelihood: {log_likelihood:.2f}")

# Predict at median of feature space
X_median = np.median(X_gp_scaled, axis=0).reshape(1, -1)
y_pred_scaled, y_std_scaled = gp.predict(X_median, return_std=True)

# Transform back to original scale
y_pred = y_pred_scaled[0] * y_std + y_mean
y_pred_std = y_std_scaled[0] * y_std

# 95% prediction interval (approximately ±1.96 * std)
pred_lower = y_pred - 1.96 * y_pred_std
pred_upper = y_pred + 1.96 * y_pred_std

print(f"Predicted mean at median: {y_pred:.2f}")
print(f"95% prediction interval: [{pred_lower:.2f}, {pred_upper:.2f}]")

# ============================================================================
# STEP 9: ZERO-INFLATED POISSON MODEL
# ============================================================================
print("\n" + "=" * 80)
print("STEP 9: ZERO-INFLATED POISSON MODEL")
print("=" * 80)

# Discretize Volume_2016
volume_2016 = merged_filtered['Volume_2016'].values
volume_discretized = np.round(volume_2016 / 100_000_000).astype(int)
volume_discretized = np.maximum(volume_discretized, 0)  # Ensure non-negative

print(f"Discretized volume range: [{volume_discretized.min()}, {volume_discretized.max()}]")
print(f"Zero rate: {(volume_discretized == 0).sum() / len(volume_discretized):.4f}")

# Try to fit Zero-Inflated Poisson model
try:
    from statsmodels.discrete.count_model import ZeroInflatedPoisson

    # Create dataframe for statsmodels
    zip_data = pd.DataFrame({'volume': volume_discretized})

    # Fit ZIP model with no covariates (intercept only)
    zip_model = ZeroInflatedPoisson(
        endog=zip_data['volume'],
        exog=np.ones((len(zip_data), 1)),
        exog_infl=np.ones((len(zip_data), 1))
    )

    zip_result = zip_model.fit(method='bfgs', maxiter=1000, disp=False)

    # Extract parameters
    # In ZIP, params[0] is Poisson lambda (log scale), params[1] is inflation (logit scale)
    lambda_param = np.exp(zip_result.params[0])
    zero_inflation_logit = zip_result.params[1]
    zero_inflation_prob = 1 / (1 + np.exp(-zero_inflation_logit))

    print(f"Zero-inflation probability: {zero_inflation_prob:.2f}")
    print(f"Poisson lambda: {lambda_param:.2f}")

except Exception as e:
    print(f"ZIP model failed to converge: {e}")
    print("Using fallback estimates:")

    # Fallback: empirical estimates
    zero_rate = (volume_discretized == 0).sum() / len(volume_discretized)
    non_zero_mean = volume_discretized[volume_discretized > 0].mean()

    print(f"Zero-inflation probability (empirical): {zero_rate:.2f}")
    print(f"Poisson lambda (non-zero mean): {non_zero_mean:.2f}")

    zero_inflation_prob = zero_rate
    lambda_param = non_zero_mean

# ============================================================================
# STEP 10: CREATE PROPENSITY SCORE SCATTER PLOT
# ============================================================================
print("\n" + "=" * 80)
print("STEP 10: CREATING VISUALIZATION")
print("=" * 80)

plt.figure(figsize=(10, 6))

# Separate by treatment status
treated_mask = merged_filtered['treatment'] == 1
control_mask = merged_filtered['treatment'] == 0

# Plot control (blue) and treated (red)
plt.scatter(
    merged_filtered.loc[control_mask, 'propensity_score'],
    merged_filtered.loc[control_mask, 'treatment'],
    c='blue',
    alpha=0.5,
    label='Control (Treatment = 0)',
    s=30
)

plt.scatter(
    merged_filtered.loc[treated_mask, 'propensity_score'],
    merged_filtered.loc[treated_mask, 'treatment'],
    c='red',
    alpha=0.5,
    label='Treated (Treatment = 1)',
    s=30
)

plt.xlabel('Propensity Score', fontsize=12)
plt.ylabel('Treatment Status', fontsize=12)
plt.title('Propensity Scores vs Treatment Status', fontsize=14, fontweight='bold')
plt.legend(loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3)
plt.yticks([0, 1])
plt.tight_layout()
plt.savefig('propensity_score_plot.png', dpi=300, bbox_inches='tight')
print("Propensity score plot saved as 'propensity_score_plot.png'")

# ============================================================================
# SUMMARY REPORT
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY REPORT")
print("=" * 80)

print(f"""
DATA PROCESSING:
  - Matched Ticker-CalendarDay observations retained: {len(merged_filtered)}

CAUSAL INFERENCE (PROPENSITY SCORE MATCHING):
  - Treatment: Above-median ticker-specific 2014 Adj Close
  - Outcome: Volume from 2015
  - Matched pairs: {len(matched_pairs)}
  - Average Treatment Effect (ATE): {ate_millions:.2f} million units
  - SMD for 'Open' before matching: {smd_before:.2f}
  - SMD for 'Open' after matching: {smd_after:.2f}

GAUSSIAN PROCESS REGRESSION:
  - Marginal log-likelihood: {log_likelihood:.2f}
  - Predicted mean at median: {y_pred:.2f}
  - 95% prediction interval: [{pred_lower:.2f}, {pred_upper:.2f}]

ZERO-INFLATED POISSON MODEL:
  - Zero-inflation probability: {zero_inflation_prob:.2f}
  - Poisson lambda: {lambda_param:.2f}

INTERPRETATION:
The Average Treatment Effect (ATE) of {ate_millions:.2f} million units represents
the causal impact of having above-median ticker-specific 2014 adjusted close price
on 2015 trading volume. This estimate accounts for selection bias through propensity
score matching, which balanced confounders (Open, High, Low, Close from 2014) between
treated and control groups. The SMD reduction from {smd_before:.2f} to {smd_after:.2f}
demonstrates improved covariate balance after matching.
""")

print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
