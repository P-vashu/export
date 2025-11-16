import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')

# Load the three datasets
print("Loading datasets...")
eth_df = pd.read_csv('coin_Ethereum.csv')
xlm_df = pd.read_csv('coin_Stellar.csv')
btc_df = pd.read_csv('coin_Bitcoin.csv')

# Select Date and Close columns for each dataset
eth_data = eth_df[['Date', 'Close']].copy()
xlm_data = xlm_df[['Date', 'Close']].copy()
btc_data = btc_df[['Date', 'Close']].copy()

# Rename Close columns to be specific
eth_data.rename(columns={'Close': 'Close_ETH'}, inplace=True)
xlm_data.rename(columns={'Close': 'Close_XLM'}, inplace=True)
btc_data.rename(columns={'Close': 'Close_BTC'}, inplace=True)

# Merge datasets on Date column using inner join
print("Merging datasets...")
merged_df = eth_data.merge(xlm_data, on='Date', how='inner')
merged_df = merged_df.merge(btc_data, on='Date', how='inner')

# Remove any missing values
print(f"Shape before removing missing values: {merged_df.shape}")
merged_df = merged_df.dropna()
print(f"Shape after removing missing values: {merged_df.shape}")

# Convert Date to datetime and sort
merged_df['Date'] = pd.to_datetime(merged_df['Date'])
merged_df = merged_df.sort_values('Date').reset_index(drop=True)

print(f"\nDataset date range: {merged_df['Date'].min()} to {merged_df['Date'].max()}")
print(f"Total observations: {len(merged_df)}")

# Prepare data for SARIMAX
endog = merged_df['Close_ETH']  # Dependent variable
exog = merged_df[['Close_XLM', 'Close_BTC']]  # Exogenous variables

# Fit SARIMAX model
print("\nFitting SARIMAX model...")
print("Order: (1, 1, 1)")
print("Seasonal Order: (1, 0, 1, 12)")
print("Method: lbfgs")
print("Max iterations: 50")
print("Enforce stationarity: True")
print("Enforce invertibility: True")

model = SARIMAX(
    endog,
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

print("\n" + "="*80)
print("SARIMAX MODEL PARAMETER ESTIMATES")
print("="*80)

# Extract and report parameter estimates
params = results.params
print(f"\nExogenous Variables:")
print(f"  Close_XLM coefficient: {params['Close_XLM']:.4f}")
print(f"  Close_BTC coefficient: {params['Close_BTC']:.4f}")

print(f"\nARIMA Parameters:")
print(f"  AR(1) coefficient: {params['ar.L1']:.4f}")
print(f"  MA(1) coefficient: {params['ma.L1']:.4f}")

print(f"\nSeasonal Parameters:")
print(f"  Seasonal AR(12) coefficient: {params['ar.S.L12']:.4f}")
print(f"  Seasonal MA(12) coefficient: {params['ma.S.L12']:.4f}")

print(f"\nResidual Variance:")
print(f"  sigma2: {params['sigma2']:.4f}")

print("\n" + "="*80)

# Get fitted values
fitted_values = results.fittedvalues

# Forecast for next 10 days
print("\nForecasting next 10 days...")

# Calculate mean values of exogenous variables
mean_xlm = merged_df['Close_XLM'].mean()
mean_btc = merged_df['Close_BTC'].mean()

print(f"Mean Close_XLM: {mean_xlm:.4f}")
print(f"Mean Close_BTC: {mean_btc:.4f}")

# Create exogenous data for forecast (10 days with mean values)
exog_forecast = pd.DataFrame({
    'Close_XLM': [mean_xlm] * 10,
    'Close_BTC': [mean_btc] * 10
})

# Generate forecast
forecast = results.forecast(steps=10, exog=exog_forecast)

print("\nForecasted Ethereum Close Prices (Next 10 Days):")
for i, value in enumerate(forecast, 1):
    print(f"  Day {i}: {value:.4f}")

highest_forecast = forecast.max()
print(f"\nHighest Forecasted Value: {highest_forecast:.4f}")

print("\n" + "="*80)

# Visualization: Actual vs Fitted
print("\nCreating visualization...")

plt.figure(figsize=(14, 7))
plt.plot(merged_df['Date'], merged_df['Close_ETH'], label='Actual', linewidth=1.5, alpha=0.8)
plt.plot(merged_df['Date'], fitted_values, label='Fitted', linewidth=1.5, alpha=0.8)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price (USD)', fontsize=12)
plt.legend(fontsize=11)
plt.title('Ethereum Close Price: Actual vs Fitted (SARIMAX Model)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ethereum_sarimax_actual_vs_fitted.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'ethereum_sarimax_actual_vs_fitted.png'")

# Display the plot
plt.show()

print("\nAnalysis complete!")
