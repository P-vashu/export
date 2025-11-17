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

# Step 1: Load datasets
df_xrp = pd.read_csv('coin_XRP.csv')
df_ltc = pd.read_csv('coin_Litecoin.csv')
df_xem = pd.read_csv('coin_NEM.csv')

# Step 2: Prepare datasets for merging
df_xrp['Date'] = pd.to_datetime(df_xrp['Date'])
df_ltc['Date'] = pd.to_datetime(df_ltc['Date'])
df_xem['Date'] = pd.to_datetime(df_xem['Date'])

# Select only Date and Close columns and rename Close columns
df_xrp_clean = df_xrp[['Date', 'Close']].rename(columns={'Close': 'Close_XRP'})
df_ltc_clean = df_ltc[['Date', 'Close']].rename(columns={'Close': 'Close_LTC'})
df_xem_clean = df_xem[['Date', 'Close']].rename(columns={'Close': 'Close_XEM'})

# Step 3: Merge datasets via inner join
df_merged = df_xrp_clean.merge(df_ltc_clean, on='Date', how='inner')
df_merged = df_merged.merge(df_xem_clean, on='Date', how='inner')

# Step 4: Handle missing values
df_merged = df_merged.dropna()

# Sort by date
df_merged = df_merged.sort_values('Date').reset_index(drop=True)

# Step 5: Prepare data for SARIMAX modeling
y = df_merged['Close_XRP']
exog = df_merged[['Close_LTC', 'Close_XEM']]

# Step 6: Fit SARIMAX model

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

# Step 7: Extract and report parameter estimates
print("="*80)
print("SARIMAX PARAMETER ESTIMATES (rounded to 4 decimals)")
print("="*80)

# Get parameter estimates
params = results.params

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
print("FORECAST RESULTS")
print("="*80)

# Create exogenous variables for forecasting using mean values
mean_ltc = exog['Close_LTC'].mean()
mean_xem = exog['Close_XEM'].mean()

exog_forecast = pd.DataFrame({
    'Close_LTC': [mean_ltc] * 10,
    'Close_XEM': [mean_xem] * 10
})

forecast = results.forecast(steps=10, exog=exog_forecast)
highest_forecast = forecast.max()

print(f"Highest forecasted value (10-day): ${highest_forecast:.4f}")
print()

# Step 9: Get fitted values for hexbin plot
fitted_values = results.fittedvalues
actual_values = y

# Align the data - remove NaN values
mask = ~np.isnan(fitted_values)
aligned_fitted = fitted_values[mask].reset_index(drop=True)
aligned_actual = actual_values[mask].reset_index(drop=True)

# Step 10: Create hexbin plot
print("="*80)
print("HEXBIN VISUALIZATION")
print("="*80)

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
print("Hexbin plot saved: sarimax_hexbin_plot.png")
print()

# Step 11: Insights on exogenous variable influence
print("="*80)
print("INSIGHTS ON EXOGENOUS VARIABLE INFLUENCE")
print("="*80)

abs_ltc = abs(coef_ltc)
abs_xem = abs(coef_xem)

if abs_ltc > abs_xem:
    ratio = abs_ltc / abs_xem if abs_xem != 0 else float('inf')
    stronger = "Litecoin (LTC)"
    print(f"Stronger influence: {stronger}")
    print(f"  - LTC coefficient is {ratio:.2f}x larger in absolute magnitude")
else:
    ratio = abs_xem / abs_ltc if abs_ltc != 0 else float('inf')
    stronger = "NEM (XEM)"
    print(f"Stronger influence: {stronger}")
    print(f"  - XEM coefficient is {ratio:.2f}x larger in absolute magnitude")

print()
if coef_ltc > 0:
    print(f"Litecoin: POSITIVE relationship (LTC +$1 → XRP +${coef_ltc:.4f})")
else:
    print(f"Litecoin: NEGATIVE relationship (LTC +$1 → XRP -${abs(coef_ltc):.4f})")

if coef_xem > 0:
    print(f"NEM: POSITIVE relationship (XEM +$1 → XRP +${coef_xem:.4f})")
else:
    print(f"NEM: NEGATIVE relationship (XEM +$1 → XRP -${abs(coef_xem):.4f})")

print()
print("="*80)
