#!/usr/bin/env python3
"""
Aviation Safety Statistical Analysis
Principal Aviation Safety Analyst - International Civil Aviation Organization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import kendalltau, kruskal, mannwhitneyu
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_val_score
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import statsmodels.api as sm
from statsmodels.formula.api import ols
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)

print("="*80)
print("AVIATION SAFETY STATISTICAL ANALYSIS")
print("International Civil Aviation Organization")
print("="*80)

# ============================================================================
# PART 1: DATA LOADING AND PREPROCESSING
# ============================================================================

print("\n[1] Loading datasets...")
airlines = pd.read_csv('airlines.csv')
airports = pd.read_csv('airports.csv')
airplanes = pd.read_csv('airplanes.csv')

print(f"   - Airlines: {len(airlines)} records")
print(f"   - Airports: {len(airports)} records")
print(f"   - Airplanes: {len(airplanes)} records")

# ============================================================================
# PART 2: AIRPORTS FILTERING AND TRANSFORMATION
# ============================================================================

print("\n[2] Filtering airports (DST in ['E', 'U', 'N'])...")
airports_filtered = airports[airports['DST'].isin(['E', 'U', 'N'])].copy()
print(f"   - Filtered airports: {len(airports_filtered)} records")

print("\n[3] Converting Timezone column to numeric...")
def convert_timezone(tz):
    if pd.isna(tz) or tz == '\\N':
        return -999
    try:
        return float(tz)
    except:
        return -999

airports_filtered['Timezone'] = airports_filtered['Timezone'].apply(convert_timezone)
print(f"   - Timezone conversion complete")

# ============================================================================
# PART 3: AIRLINES CLEANING
# ============================================================================

print("\n[4] Cleaning airlines dataset...")
# Remove records where both ICAO and Callsign contain \N
mask = ~((airlines['ICAO'] == '\\N') & (airlines['Callsign'] == '\\N'))
airlines_clean = airlines[mask].copy()
print(f"   - After removing ICAO=\\N AND Callsign=\\N: {len(airlines_clean)} records")

# Retain only Active = 'Y' (case-sensitive)
airlines_clean = airlines_clean[airlines_clean['Active'] == 'Y'].copy()
print(f"   - After filtering Active='Y': {len(airlines_clean)} records")

# ============================================================================
# PART 4: AGGREGATION AND JOINING
# ============================================================================

print("\n[5] Aggregating filtered airports by Country...")
airports_agg = airports_filtered.groupby('Country').agg(
    median_altitude=('Altitude', 'median'),
    std_latitude=('Latitude', 'std'),
    airport_count=('Airport ID', 'count')
).reset_index()
print(f"   - Aggregated to {len(airports_agg)} countries")

print("\n[6] Inner joining aggregated airports and airlines on Country...")
merged = pd.merge(airports_agg, airlines_clean, on='Country', how='inner')
print(f"   - Merged dataset: {len(merged)} records")

print("\n[7] Grouping airplanes by IATA code...")
airplanes_grouped = airplanes.groupby('IATA code').size().reset_index(name='count')
print(f"   - Airplane IATA groups: {len(airplanes_grouped)} unique codes")

# ============================================================================
# PART 5: KENDALL TAU-B CORRELATION
# ============================================================================

print("\n[8] Computing Kendall tau-b correlation...")
print("   (median Altitude vs std Latitude, airport_count > 10)")

agg_subset = airports_agg[airports_agg['airport_count'] > 10].copy()
print(f"   - Countries with > 10 airports: {len(agg_subset)}")

tau, p_value = kendalltau(
    agg_subset['median_altitude'],
    agg_subset['std_latitude'],
    method='asymptotic'
)

print(f"\n   KENDALL TAU-B RESULTS:")
print(f"   - Correlation coefficient: {tau:.4f}")
print(f"   - P-value: {p_value:.4f}")

# ============================================================================
# PART 6: GAUSSIAN MIXTURE MODEL CLUSTERING
# ============================================================================

print("\n[9] Performing Gaussian Mixture Model clustering...")
print("   (2 components, full covariance, seed 42)")

# Filter for non-negative altitude
gmm_data = airports_filtered[airports_filtered['Altitude'] >= 0].copy()
print(f"   - Airports with non-negative altitude: {len(gmm_data)}")

# Prepare features: Latitude, Longitude, sqrt(Altitude)
gmm_features = gmm_data[['Latitude', 'Longitude', 'Altitude']].copy()
gmm_features['Altitude_sqrt'] = np.sqrt(gmm_features['Altitude'])
X_gmm = gmm_features[['Latitude', 'Longitude', 'Altitude_sqrt']].values

# Standardize
scaler_gmm = StandardScaler()
X_gmm_scaled = scaler_gmm.fit_transform(X_gmm)

# Fit GMM
gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
labels_gmm = gmm.fit_predict(X_gmm_scaled)

# Compute silhouette score
silhouette_avg = silhouette_score(X_gmm_scaled, labels_gmm)

# Component with max sample count
unique, counts = np.unique(labels_gmm, return_counts=True)
max_label = unique[np.argmax(counts)]

print(f"\n   GMM CLUSTERING RESULTS:")
print(f"   - Average silhouette coefficient: {silhouette_avg:.4f}")
print(f"   - Component label with max sample count: {max_label:.4f}")

# ============================================================================
# PART 7: ELASTIC NET REGRESSION
# ============================================================================

print("\n[10] Running Elastic Net regression...")
print("    (alpha=0.5, L1_ratio=0.7, polynomial degree 2, no interactions)")

# Use filtered airports with non-negative altitude
elastic_data = airports_filtered[airports_filtered['Altitude'] >= 0].copy()
print(f"    - Training samples: {len(elastic_data)}")

# Features: Timezone and Latitude
X_elastic = elastic_data[['Timezone', 'Latitude']].values
y_elastic = elastic_data['Altitude'].values

# Create polynomial features of degree 2 without interaction terms
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
X_poly = poly.fit_transform(X_elastic)

# Get feature names for identification
feature_names = poly.get_feature_names_out(['Timezone', 'Latitude'])
print(f"    - Polynomial features: {list(feature_names)}")

# Elastic Net model
elastic_net = ElasticNet(alpha=0.5, l1_ratio=0.7, random_state=42, max_iter=10000)

# 5-fold cross-validation for MAE
cv_scores = cross_val_score(elastic_net, X_poly, y_elastic,
                            cv=5, scoring='neg_mean_absolute_error')
mean_mae = -cv_scores.mean()

# Fit the model to get coefficients
elastic_net.fit(X_poly, y_elastic)

# Find Latitude^2 coefficient
# Features: [Timezone, Latitude, Timezone^2, Timezone*Latitude, Latitude^2]
latitude_squared_idx = list(feature_names).index('Latitude^2')
latitude_squared_coef = elastic_net.coef_[latitude_squared_idx]

print(f"\n    ELASTIC NET RESULTS:")
print(f"    - 5-fold CV mean absolute error: {mean_mae:.4f}")
print(f"    - Latitude^2 coefficient: {latitude_squared_coef:.4f}")

# ============================================================================
# PART 8: KRUSKAL-WALLIS H TEST
# ============================================================================

print("\n[11] Conducting Kruskal-Wallis H test...")
print("    (Altitude across DST categories: E, U, N)")

dst_e = airports_filtered[airports_filtered['DST'] == 'E']['Altitude'].values
dst_u = airports_filtered[airports_filtered['DST'] == 'U']['Altitude'].values
dst_n = airports_filtered[airports_filtered['DST'] == 'N']['Altitude'].values

h_stat, h_pvalue = kruskal(dst_e, dst_u, dst_n)

print(f"\n    KRUSKAL-WALLIS RESULTS:")
print(f"    - H statistic: {h_stat:.4f}")
print(f"    - P-value: {h_pvalue:.4f}")

# ============================================================================
# PART 9: BOOTSTRAP RESAMPLING FOR IQR CONFIDENCE INTERVAL
# ============================================================================

print("\n[12] Performing bootstrap resampling...")
print("    (200 iterations, seed 123, 95% CI for Altitude IQR)")

# Filter: Latitude > 0
bootstrap_data = airports_filtered[airports_filtered['Latitude'] > 0]['Altitude'].values
print(f"    - Airports with Latitude > 0: {len(bootstrap_data)}")

np.random.seed(123)
n_bootstrap = 200
iqr_bootstrap = []

for i in range(n_bootstrap):
    sample = np.random.choice(bootstrap_data, size=len(bootstrap_data), replace=True)
    q75, q25 = np.percentile(sample, [75, 25])
    iqr = q75 - q25
    iqr_bootstrap.append(iqr)

# 95% confidence interval
ci_lower = np.percentile(iqr_bootstrap, 2.5)
ci_upper = np.percentile(iqr_bootstrap, 97.5)

print(f"\n    BOOTSTRAP RESULTS:")
print(f"    - 95% CI lower bound: {ci_lower:.4f}")
print(f"    - 95% CI upper bound: {ci_upper:.4f}")

# ============================================================================
# PART 10: OLS REGRESSION WITH DST DUMMY VARIABLES
# ============================================================================

print("\n[13] Running OLS regression...")
print("    (Altitude ~ Longitude + DST dummies, E as reference)")

# Filter for DST in [E, U]
ols_data = airports_filtered[airports_filtered['DST'].isin(['E', 'U'])].copy()
print(f"    - Sample size (DST=E or U): {len(ols_data)}")

# Create dummy variables with E as reference
ols_data['DST_U'] = (ols_data['DST'] == 'U').astype(int)

# Prepare for OLS
X_ols = ols_data[['Longitude', 'DST_U']].values
X_ols = sm.add_constant(X_ols)
y_ols = ols_data['Altitude'].values

# Fit OLS
ols_model = sm.OLS(y_ols, X_ols).fit()

# Extract F-statistic and adjusted R-squared
f_stat = ols_model.fvalue
adj_r2 = ols_model.rsquared_adj

print(f"\n    OLS REGRESSION RESULTS:")
print(f"    - Overall model F-statistic: {f_stat:.4f}")
print(f"    - Adjusted R-squared: {adj_r2:.4f}")

# ============================================================================
# PART 11: t-SNE DIMENSIONALITY REDUCTION
# ============================================================================

print("\n[14] Applying t-SNE dimensionality reduction...")
print("    (perplexity=20, lr=100, max_iter=1000, seed=99)")
print("    Note: Using max_iter=1000 for convergence (sklearn minimum is 250)")

# Use filtered airports
tsne_features = airports_filtered[['Latitude', 'Longitude', 'Altitude', 'Timezone']].copy()
X_tsne = tsne_features.values

# Standardize
scaler_tsne = StandardScaler()
X_tsne_scaled = scaler_tsne.fit_transform(X_tsne)

# Apply t-SNE (using 1000 iterations for proper convergence)
tsne = TSNE(n_components=2, perplexity=20, learning_rate=100, max_iter=1000,
            random_state=99, verbose=0)
X_tsne_embedded = tsne.fit_transform(X_tsne_scaled)

kl_divergence = tsne.kl_divergence_

print(f"\n    t-SNE RESULTS:")
print(f"    - Final Kullback-Leibler divergence: {kl_divergence:.4f}")

# ============================================================================
# PART 12: MANN-WHITNEY U TEST
# ============================================================================

print("\n[15] Executing Mann-Whitney U test...")
print("    (Altitude: latitude-positive vs latitude-negative)")

altitude_pos_lat = airports_filtered[airports_filtered['Latitude'] > 0]['Altitude'].values
altitude_neg_lat = airports_filtered[airports_filtered['Latitude'] < 0]['Altitude'].values

print(f"    - Positive latitude airports: {len(altitude_pos_lat)}")
print(f"    - Negative latitude airports: {len(altitude_neg_lat)}")

# Mann-Whitney U test with continuity correction (default in scipy)
u_stat, u_pvalue = mannwhitneyu(altitude_pos_lat, altitude_neg_lat,
                                 alternative='two-sided')

print(f"\n    MANN-WHITNEY U TEST RESULTS:")
print(f"    - U statistic: {int(u_stat)}")
print(f"    - P-value: {u_pvalue:.4f}")

# ============================================================================
# PART 13: VIOLIN PLOT VISUALIZATION
# ============================================================================

print("\n[16] Generating violin plot with box plot overlay...")

# Prepare data for violin plot
plot_data = airports_filtered[airports_filtered['DST'].isin(['E', 'U', 'N'])].copy()

# Sort DST categories alphabetically
plot_data['DST'] = pd.Categorical(plot_data['DST'], categories=['E', 'N', 'U'], ordered=True)

# Create the plot
plt.figure(figsize=(12, 8))
ax = sns.violinplot(x='DST', y='Altitude', data=plot_data,
                     order=['E', 'N', 'U'], inner=None, palette='Set2')
sns.boxplot(x='DST', y='Altitude', data=plot_data,
            order=['E', 'N', 'U'], width=0.3,
            boxprops=dict(alpha=0.7), ax=ax)

plt.xlabel('DST Category', fontsize=12, fontweight='bold')
plt.ylabel('Altitude (feet)', fontsize=12, fontweight='bold')
plt.title('Altitude Distribution by DST Category\nViolin Plot with Box Plot Overlay',
          fontsize=14, fontweight='bold', pad=20)
plt.grid(axis='y', alpha=0.3, linestyle='--')

# Save the plot
plt.tight_layout()
plt.savefig('altitude_by_dst_violin_plot.png', dpi=300, bbox_inches='tight')
print("    - Plot saved as 'altitude_by_dst_violin_plot.png'")

plt.close()

# ============================================================================
# SUMMARY REPORT
# ============================================================================

print("\n" + "="*80)
print("SUMMARY OF STATISTICAL ANALYSES")
print("="*80)

print(f"""
1. KENDALL TAU-B CORRELATION (Countries with >10 airports):
   - Correlation coefficient: {tau:.4f}
   - P-value: {p_value:.4f}

2. GAUSSIAN MIXTURE MODEL CLUSTERING:
   - Average silhouette coefficient: {silhouette_avg:.4f}
   - Component label with max sample count: {max_label:.4f}

3. ELASTIC NET REGRESSION:
   - 5-fold CV mean absolute error: {mean_mae:.4f}
   - Latitude^2 coefficient: {latitude_squared_coef:.4f}

4. KRUSKAL-WALLIS H TEST:
   - H statistic: {h_stat:.4f}
   - P-value: {h_pvalue:.4f}

5. BOOTSTRAP RESAMPLING (95% CI for Altitude IQR):
   - Lower bound: {ci_lower:.4f}
   - Upper bound: {ci_upper:.4f}

6. OLS REGRESSION:
   - Overall model F-statistic: {f_stat:.4f}
   - Adjusted R-squared: {adj_r2:.4f}

7. t-SNE DIMENSIONALITY REDUCTION:
   - Final Kullback-Leibler divergence: {kl_divergence:.4f}

8. MANN-WHITNEY U TEST:
   - U statistic: {int(u_stat)}
   - P-value: {u_pvalue:.4f}

9. VISUALIZATION:
   - Violin plot with box plot overlay saved
""")

print("="*80)
print("ANALYSIS COMPLETE")
print("="*80)
