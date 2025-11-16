import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')

# Load the three datasets
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
merged_df = eth_data.merge(xlm_data, on='Date', how='inner')
merged_df = merged_df.merge(btc_data, on='Date', how='inner')

# Remove any missing values
merged_df = merged_df.dropna()

# Convert Date to datetime and sort
merged_df['Date'] = pd.to_datetime(merged_df['Date'])
merged_df = merged_df.sort_values('Date').reset_index(drop=True)

# Prepare data for SARIMAX
endog = merged_df['Close_ETH']  # Dependent variable
exog = merged_df[['Close_XLM', 'Close_BTC']]  # Exogenous variables

# Fit SARIMAX model
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

# Extract and report parameter estimates
params = results.params
print("SARIMAX Parameter Estimates:")
print(f"Close_XLM: {params['Close_XLM']:.4f}")
print(f"Close_BTC: {params['Close_BTC']:.4f}")
print(f"AR(1): {params['ar.L1']:.4f}")
print(f"MA(1): {params['ma.L1']:.4f}")
print(f"Seasonal AR(12): {params['ar.S.L12']:.4f}")
print(f"Seasonal MA(12): {params['ma.S.L12']:.4f}")
print(f"sigma2: {params['sigma2']:.4f}")

# Get fitted values
fitted_values = results.fittedvalues

# Forecast for next 10 days
# Calculate mean values of exogenous variables
mean_xlm = merged_df['Close_XLM'].mean()
mean_btc = merged_df['Close_BTC'].mean()

# Create exogenous data for forecast (10 days with mean values)
exog_forecast = pd.DataFrame({
    'Close_XLM': [mean_xlm] * 10,
    'Close_BTC': [mean_btc] * 10
})

# Generate forecast
forecast = results.forecast(steps=10, exog=exog_forecast)

highest_forecast = forecast.max()
print(f"\nHighest Forecasted Value: {highest_forecast:.4f}")

# Visualization: Actual vs Fitted
plt.figure(figsize=(14, 7))
plt.plot(merged_df['Date'], merged_df['Close_ETH'], label='Actual', linewidth=1.5, alpha=0.8)
plt.plot(merged_df['Date'], fitted_values, label='Fitted', linewidth=1.5, alpha=0.8)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price (USD)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ethereum_sarimax_actual_vs_fitted.png', dpi=300, bbox_inches='tight')
plt.close()
