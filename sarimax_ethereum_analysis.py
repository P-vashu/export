import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("SARIMAX TIME SERIES MODELING: Ethereum Price Analysis")
print("="*80)

# Step 1: Load the three datasets
print("\n[Step 1] Loading datasets...")
df_eth = pd.read_csv('coin_Ethereum.csv')
df_xlm = pd.read_csv('coin_Stellar.csv')
df_btc = pd.read_csv('coin_Bitcoin.csv')

print(f"Ethereum dataset: {len(df_eth)} records")
print(f"Stellar dataset: {len(df_xlm)} records")
print(f"Bitcoin dataset: {len(df_btc)} records")

# Step 2: Prepare datasets for merging
print("\n[Step 2] Preparing datasets for merge...")

# Extract Date and Close price for each cryptocurrency
eth_data = df_eth[['Date', 'Close']].copy()
eth_data.rename(columns={'Close': 'Close_ETH'}, inplace=True)

xlm_data = df_xlm[['Date', 'Close']].copy()
xlm_data.rename(columns={'Close': 'Close_XLM'}, inplace=True)

btc_data = df_btc[['Date', 'Close']].copy()
btc_data.rename(columns={'Close': 'Close_BTC'}, inplace=True)

# Step 3: Merge datasets on Date column using inner join
print("\n[Step 3] Merging datasets on Date column (inner join)...")
merged_data = eth_data.merge(xlm_data, on='Date', how='inner')
merged_data = merged_data.merge(btc_data, on='Date', how='inner')

print(f"Merged dataset: {len(merged_data)} records (overlapping dates)")

# Step 4: Remove missing values
print("\n[Step 4] Checking and removing missing values...")
print(f"Missing values before removal:\n{merged_data.isnull().sum()}")
merged_data = merged_data.dropna()
print(f"Records after removing missing values: {len(merged_data)}")

# Convert Date to datetime
merged_data['Date'] = pd.to_datetime(merged_data['Date'])
merged_data = merged_data.sort_values('Date').reset_index(drop=True)

print(f"\nDate range: {merged_data['Date'].min()} to {merged_data['Date'].max()}")
print(f"\nFinal dataset shape: {merged_data.shape}")
print(f"\nFirst few rows:")
print(merged_data.head())

# Step 5: Prepare data for SARIMAX modeling
print("\n[Step 5] Preparing data for SARIMAX modeling...")

# Endogenous variable (target)
y = merged_data['Close_ETH']

# Exogenous variables (predictors)
X = merged_data[['Close_XLM', 'Close_BTC']]

print(f"Endogenous variable (Close_ETH): {len(y)} observations")
print(f"Exogenous variables: {X.shape}")

# Step 6: Fit SARIMAX model
print("\n[Step 6] Fitting SARIMAX model...")
print("Model parameters:")
print("  order = (1, 1, 1)")
print("  seasonal_order = (1, 0, 1, 12)")
print("  method = 'lbfgs'")
print("  maxiter = 50")
print("  enforce_stationarity = True")
print("  enforce_invertibility = True")

model = SARIMAX(
    y,
    exog=X,
    order=(1, 1, 1),
    seasonal_order=(1, 0, 1, 12),
    enforce_stationarity=True,
    enforce_invertibility=True
)

# Fit the model
results = model.fit(
    method='lbfgs',
    maxiter=50,
    disp=False
)

print("\nModel fitting completed successfully!")

# Step 7: Extract and report SARIMAX parameter estimates
print("\n" + "="*80)
print("SARIMAX PARAMETER ESTIMATES (Rounded to 4 Decimal Places)")
print("="*80)

# Get parameter names and values
params = results.params
param_names = params.index.tolist()

# Report exogenous coefficients
print("\nExogenous Variable Coefficients:")
print(f"  Close_XLM coefficient: {params['Close_XLM']:.4f}")
print(f"  Close_BTC coefficient: {params['Close_BTC']:.4f}")

# Report AR and MA parameters
print("\nARIMA Parameters:")
if 'ar.L1' in param_names:
    print(f"  AR(1) coefficient: {params['ar.L1']:.4f}")
if 'ma.L1' in param_names:
    print(f"  MA(1) coefficient: {params['ma.L1']:.4f}")

# Report Seasonal parameters
print("\nSeasonal Parameters:")
if 'ar.S.L12' in param_names:
    print(f"  Seasonal AR(12) coefficient: {params['ar.S.L12']:.4f}")
if 'ma.S.L12' in param_names:
    print(f"  Seasonal MA(12) coefficient: {params['ma.S.L12']:.4f}")

# Report residual variance (sigma2)
print("\nResidual Variance:")
print(f"  sigma2: {params['sigma2']:.4f}")

print("\n" + "="*80)
print("Complete Parameter Summary:")
print("="*80)
for param_name in param_names:
    print(f"  {param_name}: {params[param_name]:.4f}")

# Step 8: Generate fitted values
print("\n[Step 8] Generating fitted values...")
fitted_values = results.fittedvalues

# Step 9: Forecast for next 10 days
print("\n[Step 9] Forecasting Ethereum Close prices for next 10 days...")

# Calculate mean values for exogenous variables
mean_xlm = X['Close_XLM'].mean()
mean_btc = X['Close_BTC'].mean()

print(f"\nMean values used for exogenous variables:")
print(f"  Mean Close_XLM: {mean_xlm:.4f}")
print(f"  Mean Close_BTC: {mean_btc:.4f}")

# Create exogenous data for forecast (10 days with mean values)
exog_forecast = np.array([[mean_xlm, mean_btc]] * 10)
exog_forecast_df = pd.DataFrame(exog_forecast, columns=['Close_XLM', 'Close_BTC'])

# Generate forecast
forecast = results.forecast(steps=10, exog=exog_forecast_df)

print("\n" + "="*80)
print("10-DAY ETHEREUM PRICE FORECAST (Rounded to 4 Decimal Places)")
print("="*80)
for i, value in enumerate(forecast, 1):
    print(f"  Day {i}: ${value:.4f}")

# Step 10: Visualize actual vs fitted values
print("\n[Step 10] Creating visualization...")

plt.figure(figsize=(14, 7))
plt.plot(merged_data['Date'], y, label='Actual', color='blue', linewidth=1.5, alpha=0.7)
plt.plot(merged_data['Date'], fitted_values, label='Fitted', color='red', linewidth=1.5, alpha=0.7)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price (USD)', fontsize=12)
plt.title('Actual vs Fitted Ethereum Close Prices (SARIMAX Model)', fontsize=14, fontweight='bold')
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the plot
plt.savefig('ethereum_sarimax_actual_vs_fitted.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved as 'ethereum_sarimax_actual_vs_fitted.png'")

# Display the plot
plt.show()

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

# Save results to a text file
with open('sarimax_results_summary.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("SARIMAX TIME SERIES MODELING: Ethereum Price Analysis\n")
    f.write("="*80 + "\n\n")

    f.write("MODEL SPECIFICATION:\n")
    f.write("  order = (1, 1, 1)\n")
    f.write("  seasonal_order = (1, 0, 1, 12)\n")
    f.write("  method = 'lbfgs'\n")
    f.write("  maxiter = 50\n")
    f.write("  enforce_stationarity = True\n")
    f.write("  enforce_invertibility = True\n\n")

    f.write("="*80 + "\n")
    f.write("PARAMETER ESTIMATES (Rounded to 4 Decimal Places)\n")
    f.write("="*80 + "\n\n")

    f.write("Exogenous Variable Coefficients:\n")
    f.write(f"  Close_XLM coefficient: {params['Close_XLM']:.4f}\n")
    f.write(f"  Close_BTC coefficient: {params['Close_BTC']:.4f}\n\n")

    f.write("ARIMA Parameters:\n")
    if 'ar.L1' in param_names:
        f.write(f"  AR(1) coefficient: {params['ar.L1']:.4f}\n")
    if 'ma.L1' in param_names:
        f.write(f"  MA(1) coefficient: {params['ma.L1']:.4f}\n\n")

    f.write("Seasonal Parameters:\n")
    if 'ar.S.L12' in param_names:
        f.write(f"  Seasonal AR(12) coefficient: {params['ar.S.L12']:.4f}\n")
    if 'ma.S.L12' in param_names:
        f.write(f"  Seasonal MA(12) coefficient: {params['ma.S.L12']:.4f}\n\n")

    f.write("Residual Variance:\n")
    f.write(f"  sigma2: {params['sigma2']:.4f}\n\n")

    f.write("="*80 + "\n")
    f.write("10-DAY ETHEREUM PRICE FORECAST (USD)\n")
    f.write("="*80 + "\n\n")
    f.write(f"Mean Close_XLM: {mean_xlm:.4f}\n")
    f.write(f"Mean Close_BTC: {mean_btc:.4f}\n\n")

    for i, value in enumerate(forecast, 1):
        f.write(f"  Day {i}: ${value:.4f}\n")

    f.write("\n" + "="*80 + "\n")

print("\nResults summary saved to 'sarimax_results_summary.txt'")
print("\nAll tasks completed successfully!")
