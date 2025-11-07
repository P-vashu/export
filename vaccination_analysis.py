import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.stats import ks_2samp
from statsmodels.tsa.seasonal import STL
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.tsa.stattools import grangercausalitytests
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. LOAD DATASETS
# ============================================================================
print("="*80)
print("STEP 1: Loading Datasets")
print("="*80)

state_df = pd.read_csv('state.csv')
vaccination_df = pd.read_csv('vaccination.csv')
time_df = pd.read_csv('time.csv')

print(f"State dataset shape: {state_df.shape}")
print(f"Vaccination dataset shape: {vaccination_df.shape}")
print(f"Time dataset shape: {time_df.shape}")

# ============================================================================
# 2. MERGE DATASETS
# ============================================================================
print("\n" + "="*80)
print("STEP 2: Merging Datasets")
print("="*80)

# Merge state.csv with vaccination.csv on date (inner join)
merged_state_vac = pd.merge(state_df, vaccination_df, on='date', how='inner')
print(f"Merged state_vac shape: {merged_state_vac.shape}")

# Merge merged_state_vac with time.csv on date (inner join)
final_dataset = pd.merge(merged_state_vac, time_df, on='date', how='inner')
print(f"Final dataset shape: {final_dataset.shape}")

# ============================================================================
# 3. DATA CLEANING
# ============================================================================
print("\n" + "="*80)
print("STEP 3: Data Cleaning")
print("="*80)

# Remove rows with missing values in specified columns
print(f"Shape before cleaning: {final_dataset.shape}")
final_dataset = final_dataset.dropna(subset=['Total-Vaccinated', 'tot_dose_1', 'count', 'dose_one'])
print(f"Shape after cleaning: {final_dataset.shape}")

# ============================================================================
# 4. FEATURE ENGINEERING
# ============================================================================
print("\n" + "="*80)
print("STEP 4: Feature Engineering")
print("="*80)

# Extract starting hour from time_slot
final_dataset['start_hour'] = final_dataset['time_slot'].str.split(':').str[0].astype(int)
print(f"Start hour extracted. Sample values: {final_dataset['start_hour'].unique()[:10]}")

# Standardize Partial-Vaccinated and Total-Vaccinated using z-score normalization
scaler = StandardScaler()
final_dataset['Partial_Vaccinated_std'] = scaler.fit_transform(final_dataset[['Partial-Vaccinated']])
final_dataset['Total_Vaccinated_std'] = scaler.fit_transform(final_dataset[['Total-Vaccinated']])
print("Standardization completed for Partial-Vaccinated and Total-Vaccinated")

# ============================================================================
# 5. GAUSSIAN MIXTURE MODEL
# ============================================================================
print("\n" + "="*80)
print("STEP 5: Gaussian Mixture Model Analysis")
print("="*80)

# Apply GMM with 4 components
X_gmm = final_dataset[['Partial_Vaccinated_std', 'Total_Vaccinated_std']].values
gmm = GaussianMixture(
    n_components=4,
    covariance_type='full',
    init_params='kmeans',
    random_state=42
)
gmm.fit(X_gmm)

# Report BIC and log-likelihood
bic = gmm.bic(X_gmm)
log_likelihood = gmm.score(X_gmm) * len(X_gmm)  # score returns mean log-likelihood

print(f"Bayesian Information Criterion (BIC): {bic:.2f}")
print(f"Log-likelihood: {log_likelihood:.3f}")

# ============================================================================
# 6. QUANTILE REGRESSION FOR MAHARASHTRA
# ============================================================================
print("\n" + "="*80)
print("STEP 6: Quantile Regression (Maharashtra, start_hour 11-15, 0.90 quantile)")
print("="*80)

# Filter data for Maharashtra and start_hour between 11-15
maharashtra_data = final_dataset[
    (final_dataset['State'] == 'Maharashtra') &
    (final_dataset['start_hour'] >= 11) &
    (final_dataset['start_hour'] <= 15)
].copy()

print(f"Maharashtra filtered dataset shape: {maharashtra_data.shape}")

if len(maharashtra_data) > 0:
    # Prepare data for quantile regression
    y = maharashtra_data['Total-Vaccinated']
    X = maharashtra_data[['Partial-Vaccinated', 'count']]
    X = pd.DataFrame(X)
    X.insert(0, 'const', 1.0)  # Add constant term

    # Perform quantile regression at 0.90 quantile
    qr_model = QuantReg(y, X)
    qr_results = qr_model.fit(q=0.90)

    # Extract coefficients
    intercept_090 = qr_results.params['const']
    partial_vac_coef_090 = qr_results.params['Partial-Vaccinated']
    count_coef_090 = qr_results.params['count']

    print(f"Intercept at 0.90 quantile: {intercept_090:.2f}")
    print(f"Partial-Vaccinated slope at 0.90 quantile: {partial_vac_coef_090:.6f}")
    print(f"Count slope at 0.90 quantile: {count_coef_090:.7f}")
else:
    print("No data available for Maharashtra with start_hour 11-15")

# ============================================================================
# 7. KOLMOGOROV-SMIRNOV TEST (Uttar Pradesh vs Bihar)
# ============================================================================
print("\n" + "="*80)
print("STEP 7: Kolmogorov-Smirnov Test (Uttar Pradesh vs Bihar)")
print("="*80)

# Aggregate by State and date
aggregated = final_dataset.groupby(['State', 'date']).agg({
    'tot_dose_1': 'sum',
    'count': 'sum'
}).reset_index()

# Calculate vaccination_rate
aggregated['vaccination_rate'] = aggregated['tot_dose_1'] / aggregated['count']

# Filter for Uttar Pradesh and Bihar
up_data = aggregated[aggregated['State'] == 'Uttar Pradesh']['vaccination_rate'].dropna()
bihar_data = aggregated[aggregated['State'] == 'Bihar']['vaccination_rate'].dropna()

print(f"Uttar Pradesh sample size: {len(up_data)}")
print(f"Bihar sample size: {len(bihar_data)}")

if len(up_data) > 0 and len(bihar_data) > 0:
    # Perform KS test
    ks_statistic, ks_pvalue = ks_2samp(up_data, bihar_data)

    print(f"Kolmogorov-Smirnov test statistic: {ks_statistic:.5f}")
    print(f"P-value: {ks_pvalue:.8f}")
else:
    print("Insufficient data for KS test")

# ============================================================================
# 8. STL SEASONAL DECOMPOSITION
# ============================================================================
print("\n" + "="*80)
print("STEP 8: STL Seasonal Decomposition")
print("="*80)

# Aggregate by date
time_series_data = final_dataset.groupby('date').agg({
    'count': 'sum',
    'dose_one': 'sum'
}).reset_index().sort_values('date')

print(f"Time series data shape: {time_series_data.shape}")

# Convert date to datetime and set as index
time_series_data['date'] = pd.to_datetime(time_series_data['date'])
time_series_data = time_series_data.set_index('date').sort_index()

# Check if we have enough data points (need at least 2 full periods)
if len(time_series_data) >= 14:
    # Perform STL decomposition on count
    stl = STL(
        time_series_data['count'],
        seasonal=7,
        period=7,
        trend=21,
        robust=True,
        seasonal_deg=1
    )
    stl_result = stl.fit()

    # Calculate variances
    seasonal_var = np.var(stl_result.seasonal)
    trend_var = np.var(stl_result.trend)

    # Calculate ratio and max residual
    if trend_var > 0:
        variance_ratio = seasonal_var / trend_var
    else:
        variance_ratio = np.inf

    max_residual = np.max(stl_result.resid)

    print(f"Seasonal variance: {seasonal_var:.2f}")
    print(f"Trend variance: {trend_var:.2f}")
    print(f"Ratio of seasonal variance to trend variance: {variance_ratio:.4f}")
    print(f"Maximum value of residual component: {max_residual:.2f}")
else:
    print(f"Insufficient data points for STL decomposition (need at least 14, have {len(time_series_data)})")

# ============================================================================
# 9. GRANGER CAUSALITY TEST (Tamil Nadu)
# ============================================================================
print("\n" + "="*80)
print("STEP 9: Granger Causality Test (Tamil Nadu)")
print("="*80)

# Filter for Tamil Nadu
tamil_nadu_data = final_dataset[final_dataset['State'] == 'Tamil Nadu'].copy()
print(f"Tamil Nadu dataset shape: {tamil_nadu_data.shape}")

if len(tamil_nadu_data) > 0:
    # Aggregate by date for Tamil Nadu
    tn_time_series = tamil_nadu_data.groupby('date').agg({
        'count': 'sum',
        'dose_one': 'sum'
    }).reset_index().sort_values('date')

    print(f"Tamil Nadu time series shape: {tn_time_series.shape}")

    # Need at least 4 observations for maxlag=3
    if len(tn_time_series) >= 8:
        # Prepare data for Granger test (dose_one is dependent, count is independent)
        granger_data = tn_time_series[['dose_one', 'count']].values

        # Perform Granger causality test
        try:
            granger_results = grangercausalitytests(granger_data, maxlag=3, verbose=False)

            # Extract results for lag 3
            lag3_results = granger_results[3][0]
            f_stat = lag3_results['ssr_ftest'][0]
            p_value = lag3_results['ssr_ftest'][1]

            print(f"Granger causality test (lag 3):")
            print(f"F-statistic: {f_stat:.4f}")
            print(f"P-value: {p_value:.7f}")
        except Exception as e:
            print(f"Error in Granger causality test: {e}")
    else:
        print(f"Insufficient data points for Granger test with lag 3 (need at least 8, have {len(tn_time_series)})")
else:
    print("No data available for Tamil Nadu")

# ============================================================================
# 10. CONTOUR PLOT
# ============================================================================
print("\n" + "="*80)
print("STEP 10: Generating Contour Plot")
print("="*80)

# Filter data for start_hour between 9-17
contour_data = final_dataset[
    (final_dataset['start_hour'] >= 9) &
    (final_dataset['start_hour'] <= 17)
].copy()

print(f"Contour plot data shape: {contour_data.shape}")

if len(contour_data) > 0:
    # Create figure
    plt.figure(figsize=(12, 8))

    # Extract data
    x = contour_data['Partial-Vaccinated']
    y = contour_data['tot_dose_1']
    z = contour_data['count']

    # Create grid for interpolation
    from scipy.interpolate import griddata

    # Define grid
    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate z values on grid
    zi = griddata((x, y), z, (xi, yi), method='linear')

    # Create contour plot
    contour = plt.contourf(xi, yi, zi, levels=15, cmap='viridis', alpha=0.8)
    plt.colorbar(contour, label='Count Density')

    # Add contour lines
    plt.contour(xi, yi, zi, levels=15, colors='black', alpha=0.3, linewidths=0.5)

    plt.xlabel('Partial-Vaccinated', fontsize=12)
    plt.ylabel('tot_dose_1', fontsize=12)
    plt.title('Contour Plot: Density of Count\n(start_hour 9-17)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    plt.savefig('vaccination_contour_plot.png', dpi=300, bbox_inches='tight')
    print("Contour plot saved as 'vaccination_contour_plot.png'")
    plt.close()
else:
    print("No data available for contour plot")

# ============================================================================
# SUMMARY REPORT
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS COMPLETE - SUMMARY OF RESULTS")
print("="*80)
print("\n1. GAUSSIAN MIXTURE MODEL:")
print(f"   - BIC: {bic:.2f}")
print(f"   - Log-likelihood: {log_likelihood:.3f}")

print("\n2. QUANTILE REGRESSION (Maharashtra, 0.90 quantile):")
if len(maharashtra_data) > 0:
    print(f"   - Intercept: {intercept_090:.2f}")
    print(f"   - Partial-Vaccinated slope: {partial_vac_coef_090:.6f}")
    print(f"   - Count slope: {count_coef_090:.7f}")

print("\n3. KOLMOGOROV-SMIRNOV TEST (UP vs Bihar):")
if len(up_data) > 0 and len(bihar_data) > 0:
    print(f"   - KS statistic: {ks_statistic:.5f}")
    print(f"   - P-value: {ks_pvalue:.8f}")

print("\n4. STL DECOMPOSITION:")
if len(time_series_data) >= 14:
    print(f"   - Variance ratio (seasonal/trend): {variance_ratio:.4f}")
    print(f"   - Max residual: {max_residual:.2f}")

print("\n5. GRANGER CAUSALITY TEST (Tamil Nadu, lag 3):")
if len(tamil_nadu_data) > 0 and len(tn_time_series) >= 8:
    print(f"   - F-statistic: {f_stat:.4f}")
    print(f"   - P-value: {p_value:.7f}")

print("\n6. CONTOUR PLOT:")
print(f"   - Generated successfully and saved as 'vaccination_contour_plot.png'")

print("\n" + "="*80)
print("ALL ANALYSES COMPLETED SUCCESSFULLY")
print("="*80)
