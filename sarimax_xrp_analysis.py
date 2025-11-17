#!/usr/bin/env python3
"""
SARIMAX Time Series Analysis: XRP Price Prediction
Analyzing the influence of Litecoin and NEM prices on XRP prices
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("SARIMAX Time Series Analysis: XRP Price Prediction")
print("="*80)
print()

# Step 1: Load datasets
print("Step 1: Loading datasets...")
df_xrp = pd.read_csv('coin_XRP.csv')
df_ltc = pd.read_csv('coin_Litecoin.csv')
df_xem = pd.read_csv('coin_NEM.csv')

print(f"  - XRP dataset: {len(df_xrp)} records")
print(f"  - Litecoin dataset: {len(df_ltc)} records")
print(f"  - NEM dataset: {len(df_xem)} records")
print()

# Step 2: Prepare datasets for merging
print("Step 2: Preparing datasets for merging...")
df_xrp['Date'] = pd.to_datetime(df_xrp['Date'])
df_ltc['Date'] = pd.to_datetime(df_ltc['Date'])
df_xem['Date'] = pd.to_datetime(df_xem['Date'])

# Select only Date and Close columns and rename Close columns
df_xrp_clean = df_xrp[['Date', 'Close']].rename(columns={'Close': 'Close_XRP'})
df_ltc_clean = df_ltc[['Date', 'Close']].rename(columns={'Close': 'Close_LTC'})
df_xem_clean = df_xem[['Date', 'Close']].rename(columns={'Close': 'Close_XEM'})

print(f"  - XRP date range: {df_xrp['Date'].min()} to {df_xrp['Date'].max()}")
print(f"  - Litecoin date range: {df_ltc['Date'].min()} to {df_ltc['Date'].max()}")
print(f"  - NEM date range: {df_xem['Date'].min()} to {df_xem['Date'].max()}")
print()

# Step 3: Merge datasets via inner join
print("Step 3: Merging datasets on Date column (inner join)...")
df_merged = df_xrp_clean.merge(df_ltc_clean, on='Date', how='inner')
df_merged = df_merged.merge(df_xem_clean, on='Date', how='inner')

print(f"  - Merged dataset: {len(df_merged)} records")
print(f"  - Date range: {df_merged['Date'].min()} to {df_merged['Date'].max()}")
print()

# Step 4: Handle missing values
print("Step 4: Checking and handling missing values...")
print(f"  - Missing values before cleaning:")
print(f"    Close_XRP: {df_merged['Close_XRP'].isna().sum()}")
print(f"    Close_LTC: {df_merged['Close_LTC'].isna().sum()}")
print(f"    Close_XEM: {df_merged['Close_XEM'].isna().sum()}")

df_merged = df_merged.dropna()
print(f"  - Records after removing missing values: {len(df_merged)}")
print()

# Sort by date
df_merged = df_merged.sort_values('Date').reset_index(drop=True)

# Display sample data
print("Sample of merged dataset:")
print(df_merged.head(10))
print()

# Step 5: Prepare data for SARIMAX modeling
print("Step 5: Preparing data for SARIMAX modeling...")
y = df_merged['Close_XRP']
exog = df_merged[['Close_LTC', 'Close_XEM']]

print(f"  - Endogenous variable (y): Close_XRP, shape: {y.shape}")
print(f"  - Exogenous variables (X): Close_LTC, Close_XEM, shape: {exog.shape}")
print(f"  - Mean Close_LTC: {exog['Close_LTC'].mean():.4f}")
print(f"  - Mean Close_XEM: {exog['Close_XEM'].mean():.4f}")
print()

# Step 6: Fit SARIMAX model
print("Step 6: Fitting SARIMAX model...")
print("  - Model parameters:")
print("    order: (1, 1, 1)")
print("    seasonal_order: (1, 0, 1, 12)")
print("    method: lbfgs")
print("    maxiter: 50")
print("    disp: False")
print("    enforce_stationarity: True")
print("    enforce_invertibility: True")
print()

model = SARIMAX(
    y,
    exog=exog,
    order=(1, 1, 1),
    seasonal_order=(1, 0, 1, 12),
    enforce_stationarity=True,
    enforce_invertibility=True
)

results = model.fit(
    method='lbfgs',
    maxiter=50,
    disp=False
)

print("  Model fitting completed!")
print()

# Step 7: Extract and report parameter estimates
print("="*80)
print("SARIMAX PARAMETER ESTIMATES")
print("="*80)
print()

# Get parameter estimates
params = results.params
print("Model Parameters (rounded to 4 decimal places):")
print("-" * 60)

# Exogenous variables coefficients
coef_ltc = params['Close_LTC']
coef_xem = params['Close_XEM']
print(f"Close_LTC coefficient:       {coef_ltc:.4f}")
print(f"Close_XEM coefficient:       {coef_xem:.4f}")

# AR, MA, and Seasonal components
ar1 = params['ar.L1']
ma1 = params['ma.L1']
sar12 = params['ar.S.L12']
sma12 = params['ma.S.L12']
sigma2 = params['sigma2']

print(f"AR(1) coefficient:           {ar1:.4f}")
print(f"MA(1) coefficient:           {ma1:.4f}")
print(f"Seasonal AR(12) coefficient: {sar12:.4f}")
print(f"Seasonal MA(12) coefficient: {sma12:.4f}")
print(f"Residual variance (sigma2):  {sigma2:.4f}")
print()

# Step 8: Forecast next 10 days
print("="*80)
print("FORECASTING XRP CLOSE PRICES FOR NEXT 10 DAYS")
print("="*80)
print()

# Create exogenous variables for forecasting using mean values
mean_ltc = exog['Close_LTC'].mean()
mean_xem = exog['Close_XEM'].mean()

print(f"Using mean values for forecasting:")
print(f"  - Mean Close_LTC: {mean_ltc:.4f}")
print(f"  - Mean Close_XEM: {mean_xem:.4f}")
print()

exog_forecast = pd.DataFrame({
    'Close_LTC': [mean_ltc] * 10,
    'Close_XEM': [mean_xem] * 10
})

forecast = results.forecast(steps=10, exog=exog_forecast)

print("Forecasted XRP Close prices for next 10 days:")
print("-" * 60)
for i, price in enumerate(forecast, 1):
    print(f"  Day {i:2d}: ${price:.4f}")

highest_forecast = forecast.max()
print()
print(f"HIGHEST FORECASTED VALUE: ${highest_forecast:.4f}")
print()

# Step 9: Get fitted values
print("="*80)
print("MODEL FIT EVALUATION")
print("="*80)
print()

fitted_values = results.fittedvalues
actual_values = y

print(f"  - Number of fitted values: {len(fitted_values)}")
print(f"  - Number of actual values: {len(actual_values)}")
print()

# Align the data - fitted values may have NaN at start due to differencing
# Remove NaN values from both arrays
mask = ~np.isnan(fitted_values)
aligned_fitted = fitted_values[mask].reset_index(drop=True)
aligned_actual = actual_values[mask].reset_index(drop=True)

print(f"  - After removing NaN values:")
print(f"    Fitted values: {len(aligned_fitted)}")
print(f"    Actual values: {len(aligned_actual)}")
print()

# Calculate some fit statistics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(aligned_actual, aligned_fitted)
mae = mean_absolute_error(aligned_actual, aligned_fitted)
rmse = np.sqrt(mse)

try:
    r2 = r2_score(aligned_actual, aligned_fitted)
    print(f"  - Mean Squared Error (MSE): {mse:.6f}")
    print(f"  - Root Mean Squared Error (RMSE): {rmse:.6f}")
    print(f"  - Mean Absolute Error (MAE): {mae:.6f}")
    print(f"  - R² Score: {r2:.4f}")
except:
    print(f"  - Mean Squared Error (MSE): {mse:.6f}")
    print(f"  - Root Mean Squared Error (RMSE): {rmse:.6f}")
    print(f"  - Mean Absolute Error (MAE): {mae:.6f}")

print()

# Step 10: Create hexbin plot
print("="*80)
print("CREATING VISUALIZATION")
print("="*80)
print()

print("Creating hexbin plot of Actual vs Fitted XRP Close prices...")

fig, ax = plt.subplots(figsize=(10, 8))

# Create hexbin plot
hexbin = ax.hexbin(
    aligned_actual,
    aligned_fitted,
    gridsize=30,
    cmap='YlOrRd',
    mincnt=1
)

# Add colorbar
cb = plt.colorbar(hexbin, ax=ax)
cb.set_label('Count', fontsize=12)

# Add diagonal line (perfect fit)
min_val = min(aligned_actual.min(), aligned_fitted.min())
max_val = max(aligned_actual.max(), aligned_fitted.max())
ax.plot([min_val, max_val], [min_val, max_val], 'b--', linewidth=2, label='Perfect Fit')

# Labels and title
ax.set_xlabel('Actual XRP Close Price ($)', fontsize=12, fontweight='bold')
ax.set_ylabel('Fitted XRP Close Price ($)', fontsize=12, fontweight='bold')
ax.set_title('SARIMAX Model: Actual vs Fitted XRP Close Prices\nHexbin Density Plot',
             fontsize=14, fontweight='bold', pad=20)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sarimax_hexbin_plot.png', dpi=300, bbox_inches='tight')
print("  - Hexbin plot saved as: sarimax_hexbin_plot.png")
print()

# Step 11: Insights on exogenous variable influence
print("="*80)
print("INSIGHTS ON EXOGENOUS VARIABLE INFLUENCE")
print("="*80)
print()

print("Analysis of Coefficient Magnitudes:")
print("-" * 60)
print(f"  Close_LTC coefficient: {coef_ltc:.4f}")
print(f"  Close_XEM coefficient: {coef_xem:.4f}")
print()

abs_ltc = abs(coef_ltc)
abs_xem = abs(coef_xem)

print(f"  Absolute values:")
print(f"    |Close_LTC|: {abs_ltc:.4f}")
print(f"    |Close_XEM|: {abs_xem:.4f}")
print()

if abs_ltc > abs_xem:
    ratio = abs_ltc / abs_xem if abs_xem != 0 else float('inf')
    stronger = "Litecoin (LTC)"
    print(f"  CONCLUSION: {stronger} has a STRONGER influence on XRP price movements")
    print(f"  - Litecoin's coefficient is {ratio:.2f}x larger in absolute magnitude")
else:
    ratio = abs_xem / abs_ltc if abs_ltc != 0 else float('inf')
    stronger = "NEM (XEM)"
    print(f"  CONCLUSION: {stronger} has a STRONGER influence on XRP price movements")
    print(f"  - NEM's coefficient is {ratio:.2f}x larger in absolute magnitude")

print()
print("Interpretation:")
print("-" * 60)

if coef_ltc > 0:
    print(f"  - Litecoin has a POSITIVE relationship with XRP")
    print(f"    When LTC price increases by $1, XRP price increases by ${coef_ltc:.4f}")
else:
    print(f"  - Litecoin has a NEGATIVE relationship with XRP")
    print(f"    When LTC price increases by $1, XRP price decreases by ${abs(coef_ltc):.4f}")

if coef_xem > 0:
    print(f"  - NEM has a POSITIVE relationship with XRP")
    print(f"    When XEM price increases by $1, XRP price increases by ${coef_xem:.4f}")
else:
    print(f"  - NEM has a NEGATIVE relationship with XRP")
    print(f"    When XEM price increases by $1, XRP price decreases by ${abs(coef_xem):.4f}")

print()
print("Statistical Significance:")
print("-" * 60)

# Get p-values
pvalues = results.pvalues
pval_ltc = pvalues['Close_LTC']
pval_xem = pvalues['Close_XEM']

print(f"  - Close_LTC p-value: {pval_ltc:.4f}", end="")
if pval_ltc < 0.05:
    print(" (Statistically significant at 5% level)")
else:
    print(" (Not statistically significant at 5% level)")

print(f"  - Close_XEM p-value: {pval_xem:.4f}", end="")
if pval_xem < 0.05:
    print(" (Statistically significant at 5% level)")
else:
    print(" (Not statistically significant at 5% level)")

print()
print("="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print()

# Save summary to file
with open('sarimax_analysis_summary.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("SARIMAX TIME SERIES ANALYSIS: XRP PRICE PREDICTION\n")
    f.write("="*80 + "\n\n")

    f.write("MODEL PARAMETERS\n")
    f.write("-"*60 + "\n")
    f.write(f"Close_LTC coefficient:       {coef_ltc:.4f}\n")
    f.write(f"Close_XEM coefficient:       {coef_xem:.4f}\n")
    f.write(f"AR(1) coefficient:           {ar1:.4f}\n")
    f.write(f"MA(1) coefficient:           {ma1:.4f}\n")
    f.write(f"Seasonal AR(12) coefficient: {sar12:.4f}\n")
    f.write(f"Seasonal MA(12) coefficient: {sma12:.4f}\n")
    f.write(f"Residual variance (sigma2):  {sigma2:.4f}\n\n")

    f.write("FORECAST RESULTS\n")
    f.write("-"*60 + "\n")
    f.write(f"Highest forecasted value: ${highest_forecast:.4f}\n\n")

    f.write("INSIGHTS\n")
    f.write("-"*60 + "\n")
    if abs_ltc > abs_xem:
        f.write(f"Litecoin (LTC) has stronger influence on XRP price movements\n")
        f.write(f"LTC coefficient is {ratio:.2f}x larger in absolute magnitude\n")
    else:
        f.write(f"NEM (XEM) has stronger influence on XRP price movements\n")
        f.write(f"XEM coefficient is {ratio:.2f}x larger in absolute magnitude\n")

print("Summary saved to: sarimax_analysis_summary.txt")
