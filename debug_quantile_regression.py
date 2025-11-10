"""
Debug Quantile Regression Issue
"""

import pandas as pd
import numpy as np
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.tools import add_constant

# Load BinanceCoin data
df_binance = pd.read_csv('coin_BinanceCoin.csv')
df_binance['Date'] = pd.to_datetime(df_binance['Date'])

# Filter for October 5, 2020 to July 6, 2021
start_date = pd.to_datetime('2020-10-05')
end_date = pd.to_datetime('2021-07-06')
df_binance_filtered = df_binance[(df_binance['Date'] >= start_date) &
                                 (df_binance['Date'] <= end_date)].copy()

# Calculate Daily_Return
df_binance_filtered['Daily_Return'] = (df_binance_filtered['Close'] -
                                        df_binance_filtered['Close'].shift(1)) / \
                                        df_binance_filtered['Close'].shift(1) * 100

# Remove first row with undefined Daily_Return
df_binance_filtered = df_binance_filtered.dropna(subset=['Daily_Return']).reset_index(drop=True)

print("BinanceCoin Data Summary:")
print(f"Number of records: {len(df_binance_filtered)}")
print(f"\nVolume statistics:")
print(df_binance_filtered['Volume'].describe())
print(f"\nClose statistics:")
print(df_binance_filtered['Close'].describe())

# Check for zero or missing values
print(f"\nZero values in Volume: {(df_binance_filtered['Volume'] == 0).sum()}")
print(f"Missing values in Volume: {df_binance_filtered['Volume'].isna().sum()}")
print(f"Missing values in Close: {df_binance_filtered['Close'].isna().sum()}")

# Prepare data: Volume (independent), Close (dependent)
X = df_binance_filtered['Volume']
y = df_binance_filtered['Close']

# Add constant (intercept)
X_with_const = add_constant(X)

print("\n" + "="*60)
print("QUANTILE REGRESSION RESULTS")
print("="*60)

# Fit quantile regression at quantiles 0.25, 0.50, 0.75
quantiles = [0.25, 0.50, 0.75]
for q in quantiles:
    model = QuantReg(y, X_with_const)
    result = model.fit(q=q)
    print(f"\nQuantile {q}:")
    print(result.summary())
    print(f"  Intercept: {result.params['const']:.10f}")
    print(f"  Slope (Volume): {result.params['Volume']:.10e}")

# Try with correlation analysis
print("\n" + "="*60)
print("CORRELATION ANALYSIS")
print("="*60)
correlation = df_binance_filtered[['Volume', 'Close']].corr()
print(correlation)
