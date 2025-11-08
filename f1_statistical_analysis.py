"""
Formula 1 Multi-Dataset Statistical Modeling Analysis
Comprehensive analysis with strict reproducibility requirements
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from imblearn.over_sampling import SMOTE
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.tsa.api import VAR
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("="*80)
print("FORMULA 1 MULTI-DATASET STATISTICAL MODELING ANALYSIS")
print("="*80)

# ============================================================================
# PHASE 1: DATA LOADING AND MERGING
# ============================================================================
print("\n" + "="*80)
print("PHASE 1: DATA LOADING AND MERGING")
print("="*80)

# Load datasets
drivers_df = pd.read_csv('drivers_updated.csv')
teams_df = pd.read_csv('teams_updated.csv')
winners_df = pd.read_csv('winners.csv')

print(f"\nInitial dataset sizes:")
print(f"  - drivers_updated.csv: {len(drivers_df)} rows")
print(f"  - teams_updated.csv: {len(teams_df)} rows")
print(f"  - winners.csv: {len(winners_df)} rows")

# Extract year from winners.Date
winners_df['Date'] = pd.to_datetime(winners_df['Date'])
winners_df['year'] = winners_df['Date'].dt.year

# Aggregate winners by Name Code and year to count wins
win_counts = winners_df.groupby(['Name Code', 'year']).size().reset_index(name='win_count')
print(f"\nWin counts aggregated: {len(win_counts)} unique driver-year combinations")

# Merge drivers_df with win_counts
drivers_merged = drivers_df.merge(
    win_counts,
    left_on=['Code', 'year'],
    right_on=['Name Code', 'year'],
    how='left'
)

# Fill NaN win_count with 0 (drivers who didn't win)
drivers_merged['win_count'] = drivers_merged['win_count'].fillna(0).astype(int)

# Drop the redundant 'Name Code' column if it exists
if 'Name Code' in drivers_merged.columns:
    drivers_merged = drivers_merged.drop(columns=['Name Code'])

total_merged_rows = len(drivers_merged)
print(f"\n>>> RESULT 1: Total rows in final merged dataset: {total_merged_rows}")

# ============================================================================
# PHASE 2: DATA CLEANING
# ============================================================================
print("\n" + "="*80)
print("PHASE 2: DATA CLEANING")
print("="*80)

# Convert Pos to numeric
drivers_merged['Pos'] = pd.to_numeric(drivers_merged['Pos'], errors='coerce')

# Count rows before dropping missing Car values
rows_before_drop = len(drivers_merged)
missing_car_count = drivers_merged['Car'].isna().sum()

# Drop rows with missing Car values
drivers_merged = drivers_merged.dropna(subset=['Car'])

rows_after_drop = len(drivers_merged)
rows_dropped = rows_before_drop - rows_after_drop
percentage_dropped = (rows_dropped / rows_before_drop) * 100

print(f"\nRows before dropping missing Car: {rows_before_drop}")
print(f"Rows with missing Car: {missing_car_count}")
print(f"Rows after dropping missing Car: {rows_after_drop}")
print(f"Rows dropped: {rows_dropped}")
print(f"\n>>> RESULT 2: Percentage of rows dropped: {percentage_dropped:.2f}%")

# ============================================================================
# PHASE 3: KOLMOGOROV-SMIRNOV TEST
# ============================================================================
print("\n" + "="*80)
print("PHASE 3: KOLMOGOROV-SMIRNOV TWO-SAMPLE TEST")
print("="*80)

# Group A: Pos <= 3
group_a = drivers_merged[drivers_merged['Pos'] <= 3]['PTS'].dropna()

# Group B: Pos > 3
group_b = drivers_merged[drivers_merged['Pos'] > 3]['PTS'].dropna()

print(f"\nGroup A (Pos <= 3): {len(group_a)} samples")
print(f"Group B (Pos > 3): {len(group_b)} samples")

# Apply KS test
ks_statistic, ks_pvalue = ks_2samp(group_a, group_b)

print(f"\n>>> RESULT 3a: KS test statistic: {ks_statistic:.4f}")
print(f">>> RESULT 3b: KS test p-value: {ks_pvalue:.4f}")

# ============================================================================
# PHASE 4: SMOTE OVERSAMPLING
# ============================================================================
print("\n" + "="*80)
print("PHASE 4: SMOTE OVERSAMPLING")
print("="*80)

# Calculate 75th percentile
pts_75th = drivers_merged['PTS'].quantile(0.75)
print(f"\n75th percentile of PTS: {pts_75th}")

# Create binary target
drivers_merged['high_performer'] = (drivers_merged['PTS'] > pts_75th).astype(int)

print(f"\nClass distribution before SMOTE:")
print(drivers_merged['high_performer'].value_counts())

# Prepare features for SMOTE
feature_cols = ['PTS', 'Pos', 'win_count']
X = drivers_merged[feature_cols].dropna()
y = drivers_merged.loc[X.index, 'high_performer']

original_count = len(X)
class_counts_before = y.value_counts()

print(f"\nOriginal dataset size: {original_count}")
print(f"Class 0: {class_counts_before.get(0, 0)}")
print(f"Class 1: {class_counts_before.get(1, 0)}")

# Apply SMOTE
smote = SMOTE(random_state=42, sampling_strategy=1.0)
X_resampled, y_resampled = smote.fit_resample(X, y)

resampled_count = len(X_resampled)
synthetic_samples = resampled_count - original_count

print(f"\nResampled dataset size: {resampled_count}")
print(f"Synthetic samples generated: {synthetic_samples}")
print(f"\n>>> RESULT 4: Total count of synthetic samples generated: {synthetic_samples}")

# ============================================================================
# PHASE 5: QUANTILE REGRESSION
# ============================================================================
print("\n" + "="*80)
print("PHASE 5: QUANTILE REGRESSION")
print("="*80)

# Aggregate race counts per year from winners dataset
annual_races = winners_df.groupby('year').size().reset_index(name='race_count')
print(f"\nAnnual races aggregated: {len(annual_races)} years")

# Merge teams with annual_races
teams_merged = teams_df.merge(annual_races, on='year', how='inner')

# Create normalized_PTS
teams_merged['normalized_PTS'] = teams_merged['PTS'] / teams_merged['race_count']

print(f"\nTeams merged with annual races: {len(teams_merged)} rows")
print(f"Normalized PTS created")

# Prepare for Quantile Regression
y_qr = teams_merged['normalized_PTS']
X_qr = teams_merged['year']
X_qr = sm.add_constant(X_qr)

# Fit Quantile Regression at 0.90 quantile
qr_model = QuantReg(y_qr, X_qr)
qr_results = qr_model.fit(q=0.90)

intercept_coef = qr_results.params['const']
slope_coef = qr_results.params['year']

print(f"\nQuantile Regression (q=0.90) Results:")
print(f">>> RESULT 5a: Intercept coefficient: {intercept_coef:.3f}")
print(f">>> RESULT 5b: Slope coefficient for year: {slope_coef:.3f}")

# ============================================================================
# PHASE 6: VECTOR AUTOREGRESSION
# ============================================================================
print("\n" + "="*80)
print("PHASE 6: VECTOR AUTOREGRESSION")
print("="*80)

# Aggregate annual driver PTS sum
annual_driver_pts = drivers_merged.groupby('year')['PTS'].sum().reset_index()
annual_driver_pts.columns = ['year', 'driver_pts_sum']

# Aggregate annual team PTS sum
annual_team_pts = teams_df.groupby('year')['PTS'].sum().reset_index()
annual_team_pts.columns = ['year', 'team_pts_sum']

# Merge both time series
time_series_data = annual_driver_pts.merge(annual_team_pts, on='year', how='inner')
time_series_data = time_series_data.sort_values('year')

print(f"\nTime series data prepared: {len(time_series_data)} years")

# Prepare data for VAR (exclude year column)
var_data = time_series_data[['driver_pts_sum', 'team_pts_sum']]

# Fit VAR model with lag order 2
var_model = VAR(var_data)
var_results = var_model.fit(maxlags=2, ic=None)

# Get coefficient matrix
# For lag 1, the coefficient matrix is at index 0 in coefs
# Row 0 = driver_pts_sum equation
# Column 2 would be the second variable's lag 1 coefficient
# Actually, coefs[0] is lag 1, coefs[1] is lag 2
# Each has shape (n_vars, n_vars)

# Extract coefficient at row 0, column 1 for lag 1 (team's PTS lag-1 effect on driver PTS)
# In VAR, coefs[0] is coefficients for lag 1
# Row 0 is the equation for first variable (driver_pts_sum)
# Column 1 is the coefficient for second variable (team_pts_sum)

# Wait, the user said "row index 0, column index 2"
# Let me check the structure of the coefficient matrix
# var_results.coefs is a list of coefficient matrices, one for each lag
# var_results.coefs[0] is for lag 1, var_results.coefs[1] is for lag 2

# The user wants: "coefficient at row index 0, column index 2"
# This is a bit ambiguous. Let me think about the structure.
# If we have 2 variables and lag order 2, the full coefficient matrix
# should have dimensions related to both lags

# Actually, I think the user wants the flattened parameter matrix
# Let me check var_results.params which gives all parameters

# Let's use the params array which is flattened
# Or use the coefs attribute which is a 3D array (lag, n_vars, n_vars)

# For clarification, I'll extract from the params or coefs
# The user specifically says "row index 0, column index 2"
# This likely refers to the params matrix reshaped

# Let me get the AIC first
aic = var_results.aic

# For the coefficient, let's examine the structure
# var_results.params has all parameters including intercept
# The shape depends on how it's organized

# I think the safest interpretation is:
# - The coefficient matrix for all lags combined
# - Or the params matrix

# Let me print the structure to understand it better
print(f"\nVAR Model Summary:")
print(f"Number of observations: {var_results.nobs}")
print(f"Number of lags: {var_results.k_ar}")
print(f"AIC: {aic:.2f}")

# The params DataFrame structure:
# Rows: const, L1.driver_pts_sum, L1.team_pts_sum, L2.driver_pts_sum, L2.team_pts_sum
# Columns: driver_pts_sum (equation 1), team_pts_sum (equation 2)

# To access coefficient at row index 0, column index 2:
# We need to transpose params to get (n_equations, n_parameters) shape
# Then [0, 2] means: equation 0 (driver_pts_sum), parameter 2 (L1.team_pts_sum)

params_transposed = var_results.params.values.T  # Shape: (2, 5)
# Row 0: driver_pts_sum equation coefficients
# Row 1: team_pts_sum equation coefficients
# Columns: [0]=const, [1]=L1.driver_pts_sum, [2]=L1.team_pts_sum, [3]=L2.driver_pts_sum, [4]=L2.team_pts_sum

# Extract coefficient at row 0, column 2
# This represents the team's PTS lag-1 term effect on driver PTS
coefficient_0_2 = params_transposed[0, 2]

print(f"\n>>> RESULT 6a: Coefficient at row 0, column 2: {coefficient_0_2:.4f}")
print(f">>> RESULT 6b: Akaike Information Criterion: {aic:.2f}")

# ============================================================================
# PHASE 7: HEXBIN VISUALIZATION
# ============================================================================
print("\n" + "="*80)
print("PHASE 7: HEXBIN VISUALIZATION")
print("="*80)

# Prepare data for hexbin plot
plot_data = drivers_merged[['PTS', 'win_count']].dropna()

# Create hexbin plot
fig, ax = plt.subplots(figsize=(10, 6))
hexbin = ax.hexbin(plot_data['PTS'], plot_data['win_count'],
                    gridsize=30, cmap='YlOrRd', mincnt=1)

ax.set_xlabel('Championship Points', fontsize=12, fontweight='bold')
ax.set_ylabel('Race Victories', fontsize=12, fontweight='bold')
ax.set_title('F1 Driver Performance: Championship Points vs Race Victories',
             fontsize=14, fontweight='bold', pad=20)

# Add colorbar
cbar = plt.colorbar(hexbin, ax=ax)
cbar.set_label('Count', fontsize=10)

# Save the plot
plt.tight_layout()
plt.savefig('f1_hexbin_plot.png', dpi=300, bbox_inches='tight')
print("\nHexbin plot saved as 'f1_hexbin_plot.png'")

# ============================================================================
# SUMMARY OF RESULTS
# ============================================================================
print("\n" + "="*80)
print("SUMMARY OF ALL RESULTS")
print("="*80)

print(f"""
PHASE 1 - DATA MERGING:
  Total rows in final merged dataset: {total_merged_rows}

PHASE 2 - DATA CLEANING:
  Percentage of rows dropped due to missing Car values: {percentage_dropped:.2f}%

PHASE 3 - KOLMOGOROV-SMIRNOV TEST:
  KS test statistic: {ks_statistic:.4f}
  KS test p-value: {ks_pvalue:.4f}

PHASE 4 - SMOTE OVERSAMPLING:
  Total count of synthetic samples generated: {synthetic_samples}

PHASE 5 - QUANTILE REGRESSION:
  Intercept coefficient: {intercept_coef:.3f}
  Slope coefficient for year: {slope_coef:.3f}

PHASE 6 - VECTOR AUTOREGRESSION:
  Coefficient at row 0, column 2: {coefficient_0_2:.4f}
  Akaike Information Criterion: {aic:.2f}

PHASE 7 - VISUALIZATION:
  Hexbin plot generated and saved
""")

print("="*80)
print("ANALYSIS COMPLETE")
print("="*80)
