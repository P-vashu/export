#!/usr/bin/env python3
"""
Aviation Network Statistical Analysis
Complex statistical operations on airlines, airports, and airplanes datasets
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_val_score
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)

print("=" * 80)
print("AVIATION NETWORK STATISTICAL ANALYSIS")
print("=" * 80)
print()

# ============================================================================
# 1. DATA LOADING AND PREPROCESSING
# ============================================================================
print("1. DATA LOADING AND PREPROCESSING")
print("-" * 80)

# Load datasets
print("Loading datasets...")
airlines = pd.read_csv('airlines.csv')
airports = pd.read_csv('airports.csv')
airplanes = pd.read_csv('airplanes.csv')
print(f"Loaded {len(airlines)} airlines, {len(airports)} airports, {len(airplanes)} airplanes")
print()

# Filter airports: DST equals E, U, or N
print("Filtering airports (DST = E, U, or N)...")
airports_filtered = airports[airports['DST'].isin(['E', 'U', 'N'])].copy()
print(f"Filtered airports: {len(airports_filtered)} records")

# Convert Timezone to numeric, replace \N with -999
print("Converting Timezone to numeric...")
def convert_timezone(x):
    if x == '\\N' or pd.isna(x):
        return -999.0
    try:
        return float(x)
    except:
        return -999.0

airports_filtered['Timezone'] = airports_filtered['Timezone'].apply(convert_timezone)
print(f"Timezone converted (\\N replaced with -999)")
print()

# Clean airlines: remove where both ICAO and Callsign contain \N
print("Cleaning airlines...")
airlines_clean = airlines[~((airlines['ICAO'] == '\\N') & (airlines['Callsign'] == '\\N'))].copy()
print(f"After removing both ICAO and Callsign = \\N: {len(airlines_clean)} records")

# Retain airlines where Active equals 'Y' (case-sensitive)
airlines_clean = airlines_clean[airlines_clean['Active'] == 'Y'].copy()
print(f"After filtering Active = 'Y': {len(airlines_clean)} records")
print()

# Aggregate airports by Country
print("Aggregating airports by Country...")
airports_agg = airports_filtered.groupby('Country').agg(
    median_altitude=('Altitude', 'median'),
    std_latitude=('Latitude', 'std'),
    airport_count=('Altitude', 'count')
).reset_index()
print(f"Aggregated to {len(airports_agg)} countries")
print()

# Inner join aggregated airports with airlines on Country
print("Inner joining aggregated airports with airlines...")
joined_data = pd.merge(airports_agg, airlines_clean, on='Country', how='inner')
print(f"Joined data: {len(joined_data)} records")
print()

# Group airplanes by IATA code and count occurrences
print("Grouping airplanes by IATA code...")
airplanes_grouped = airplanes.groupby('IATA code').size().reset_index(name='count')
print(f"Grouped airplanes: {len(airplanes_grouped)} unique IATA codes")
print()

# ============================================================================
# 2. KENDALL TAU-B CORRELATION
# ============================================================================
print("=" * 80)
print("2. KENDALL TAU-B CORRELATION")
print("-" * 80)
print("Testing correlation between median Altitude and Latitude std")
print("(where airport count > 10)")
print()

# Filter aggregated airports where count > 10
agg_filtered = airports_agg[airports_agg['airport_count'] > 10].copy()
print(f"Countries with > 10 airports: {len(agg_filtered)}")

# Compute Kendall tau-b (asymptotic method)
kendall_result = stats.kendalltau(
    agg_filtered['median_altitude'],
    agg_filtered['std_latitude'],
    method='asymptotic'
)
print(f"Kendall tau-b coefficient: {kendall_result.correlation:.5f}")
print(f"P-value: {kendall_result.pvalue:.7f}")
print()

# ============================================================================
# 3. GAUSSIAN MIXTURE MODEL
# ============================================================================
print("=" * 80)
print("3. GAUSSIAN MIXTURE MODEL")
print("-" * 80)
print("GMM with 3 components, full covariance, seed 42")
print("Features: standardized Latitude, Longitude, sqrt(Altitude)")
print("(Altitude >= 0)")
print()

# Filter airports with non-negative Altitude
gmm_data = airports_filtered[airports_filtered['Altitude'] >= 0].copy()
print(f"Airports with Altitude >= 0: {len(gmm_data)}")

# Prepare features: Latitude, Longitude, sqrt(Altitude)
gmm_features = gmm_data[['Latitude', 'Longitude', 'Altitude']].copy()
gmm_features['sqrt_altitude'] = np.sqrt(gmm_features['Altitude'])
gmm_features = gmm_features[['Latitude', 'Longitude', 'sqrt_altitude']].values

# Standardize features
scaler_gmm = StandardScaler()
gmm_features_scaled = scaler_gmm.fit_transform(gmm_features)

# Fit GMM
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
gmm_labels = gmm.fit_predict(gmm_features_scaled)

# Compute silhouette score
silhouette_avg = silhouette_score(gmm_features_scaled, gmm_labels)
print(f"Average silhouette coefficient: {silhouette_avg:.5f}")

# Find component with maximum sample count
unique, counts = np.unique(gmm_labels, return_counts=True)
max_component = unique[np.argmax(counts)]
print(f"Component with maximum sample count: {max_component}")
print()

# ============================================================================
# 4. ELASTIC NET REGRESSION
# ============================================================================
print("=" * 80)
print("4. ELASTIC NET REGRESSION")
print("-" * 80)
print("Alpha=0.5, L1 ratio=0.7")
print("Modeling Altitude using degree 2 polynomial from Timezone and Latitude")
print("(no interactions)")
print()

# Prepare data for Elastic Net
elastic_data = airports_filtered[['Timezone', 'Latitude', 'Altitude']].dropna()
X_elastic = elastic_data[['Timezone', 'Latitude']].values
y_elastic = elastic_data['Altitude'].values

# Create polynomial features (degree 2, no interactions)
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
X_poly = poly.fit_transform(X_elastic)

# The features are: Timezone, Latitude, Timezone^2, Timezone*Latitude, Latitude^2
# We need to remove the interaction term (Timezone*Latitude)
# Feature order: [Timezone, Latitude, Timezone^2, Timezone*Latitude, Latitude^2]
# Remove column 3 (index 3) which is the interaction
X_poly_no_interaction = np.delete(X_poly, 3, axis=1)

# Fit Elastic Net with 5-fold CV
elastic_net = ElasticNet(alpha=0.5, l1_ratio=0.7, random_state=42, max_iter=10000)
cv_scores = cross_val_score(elastic_net, X_poly_no_interaction, y_elastic,
                            cv=5, scoring='neg_mean_absolute_error')
cv_mae = -cv_scores.mean()
print(f"5-fold cross-validation MAE: {cv_mae:.3f}")

# Fit on full data to get coefficients
elastic_net.fit(X_poly_no_interaction, y_elastic)
# Coefficients: [Timezone, Latitude, Timezone^2, Latitude^2]
latitude_squared_coef = elastic_net.coef_[3]  # Latitude^2 is the 4th coefficient
print(f"Latitude squared coefficient: {latitude_squared_coef:.6f}")
print()

# ============================================================================
# 5. KRUSKAL-WALLIS H TEST
# ============================================================================
print("=" * 80)
print("5. KRUSKAL-WALLIS H TEST")
print("-" * 80)
print("Comparing Altitude across DST categories (E, U, N)")
print()

# Prepare groups
group_E = airports_filtered[airports_filtered['DST'] == 'E']['Altitude'].dropna()
group_U = airports_filtered[airports_filtered['DST'] == 'U']['Altitude'].dropna()
group_N = airports_filtered[airports_filtered['DST'] == 'N']['Altitude'].dropna()

print(f"Group E: {len(group_E)} samples")
print(f"Group U: {len(group_U)} samples")
print(f"Group N: {len(group_N)} samples")

# Perform Kruskal-Wallis H test
kw_result = stats.kruskal(group_E, group_U, group_N)
print(f"Kruskal-Wallis H statistic: {kw_result.statistic:.4f}")
print(f"P-value: {kw_result.pvalue:.8f}")
print()

# ============================================================================
# 6. BOOTSTRAP CONFIDENCE INTERVAL
# ============================================================================
print("=" * 80)
print("6. BOOTSTRAP CONFIDENCE INTERVAL")
print("-" * 80)
print("5000 iterations, seed 123")
print("95% CI of Altitude IQR (Latitude > 0)")
print()

# Filter airports with positive Latitude
bootstrap_data = airports_filtered[airports_filtered['Latitude'] > 0]['Altitude'].dropna().values
print(f"Airports with Latitude > 0: {len(bootstrap_data)}")

# Bootstrap
np.random.seed(123)
n_bootstrap = 5000
bootstrap_iqrs = []

for i in range(n_bootstrap):
    sample = np.random.choice(bootstrap_data, size=len(bootstrap_data), replace=True)
    iqr = np.percentile(sample, 75) - np.percentile(sample, 25)
    bootstrap_iqrs.append(iqr)

# Compute 95% CI
ci_lower = np.percentile(bootstrap_iqrs, 2.5)
ci_upper = np.percentile(bootstrap_iqrs, 97.5)
print(f"95% CI bounds: [{ci_lower:.2f}, {ci_upper:.2f}]")
print()

# ============================================================================
# 7. OLS REGRESSION
# ============================================================================
print("=" * 80)
print("7. OLS REGRESSION")
print("-" * 80)
print("Predicting Altitude using Longitude and DST dummies")
print("(E as reference, DST = E or U only)")
print()

# Filter for DST = E or U
ols_data = airports_filtered[airports_filtered['DST'].isin(['E', 'U'])].copy()
print(f"Airports with DST = E or U: {len(ols_data)}")

# Create dummy variables with E as reference
ols_data['DST_U'] = (ols_data['DST'] == 'U').astype(int)

# Prepare data
X_ols = ols_data[['Longitude', 'DST_U']].values
y_ols = ols_data['Altitude'].values

# Add intercept
X_ols_with_intercept = np.column_stack([np.ones(len(X_ols)), X_ols])

# Fit OLS using matrix algebra
beta = np.linalg.lstsq(X_ols_with_intercept, y_ols, rcond=None)[0]
y_pred = X_ols_with_intercept @ beta
residuals = y_ols - y_pred

# Calculate R-squared
ss_res = np.sum(residuals**2)
ss_tot = np.sum((y_ols - np.mean(y_ols))**2)
r_squared = 1 - (ss_res / ss_tot)

# Calculate adjusted R-squared
n = len(y_ols)
p = X_ols.shape[1]  # number of predictors (excluding intercept)
adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)

# Calculate F-statistic
msr = (ss_tot - ss_res) / p
mse = ss_res / (n - p - 1)
f_statistic = msr / mse

print(f"F-statistic: {f_statistic:.4f}")
print(f"Adjusted R-squared: {adj_r_squared:.5f}")
print()

# ============================================================================
# 8. t-SNE DIMENSIONALITY REDUCTION
# ============================================================================
print("=" * 80)
print("8. t-SNE DIMENSIONALITY REDUCTION")
print("-" * 80)
print("Perplexity=30, learning_rate=200, n_iter=2500, seed=99")
print("Features: standardized Latitude, Longitude, Altitude, Timezone")
print()

# Prepare data
tsne_data = airports_filtered[['Latitude', 'Longitude', 'Altitude', 'Timezone']].dropna()
print(f"Airports for t-SNE: {len(tsne_data)}")

# Standardize features
scaler_tsne = StandardScaler()
tsne_features = scaler_tsne.fit_transform(tsne_data.values)

# Apply t-SNE
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, max_iter=2500,
            random_state=99, verbose=0)
tsne_result = tsne.fit_transform(tsne_features)

# Get final KL divergence
kl_divergence = tsne.kl_divergence_
print(f"Final KL divergence: {kl_divergence:.5f}")
print()

# ============================================================================
# 9. MANN-WHITNEY U TEST
# ============================================================================
print("=" * 80)
print("9. MANN-WHITNEY U TEST")
print("-" * 80)
print("Comparing Altitude between positive and negative Latitude")
print("(with continuity correction)")
print()

# Prepare groups
positive_lat = airports_filtered[airports_filtered['Latitude'] > 0]['Altitude'].dropna()
negative_lat = airports_filtered[airports_filtered['Latitude'] < 0]['Altitude'].dropna()

print(f"Positive Latitude: {len(positive_lat)} samples")
print(f"Negative Latitude: {len(negative_lat)} samples")

# Perform Mann-Whitney U test with continuity correction
mw_result = stats.mannwhitneyu(positive_lat, negative_lat, alternative='two-sided')
print(f"U statistic: {int(mw_result.statistic)}")
print(f"P-value: {mw_result.pvalue:.8f}")
print()

# ============================================================================
# 10. HIERARCHICAL CLUSTERING
# ============================================================================
print("=" * 80)
print("10. HIERARCHICAL CLUSTERING")
print("-" * 80)
print("Ward linkage, standardized Latitude and Longitude, cut to 6 clusters")
print()

# Prepare data
hclust_data = airports_filtered[['Latitude', 'Longitude']].dropna()
print(f"Airports for clustering: {len(hclust_data)}")

# Standardize features
scaler_hclust = StandardScaler()
hclust_features = scaler_hclust.fit_transform(hclust_data.values)

# Perform hierarchical clustering
linkage_matrix = linkage(hclust_features, method='ward')

# Cut to 6 clusters
clusters = fcluster(linkage_matrix, t=6, criterion='maxclust')

# Compute cophenetic correlation
from scipy.cluster.hierarchy import cophenet
coph_corr, coph_dist = cophenet(linkage_matrix, pdist(hclust_features))
print(f"Cophenetic correlation: {coph_corr:.5f}")
print()

# ============================================================================
# 11. VIOLIN PLOT
# ============================================================================
print("=" * 80)
print("11. VIOLIN PLOT GENERATION")
print("-" * 80)
print("Altitude by DST category (E, U, N) - ordered alphabetically")
print()

# Prepare data for plotting
plot_data = airports_filtered[['Altitude', 'DST']].copy()
plot_data = plot_data.sort_values('DST')

# Create figure
plt.figure(figsize=(10, 6))
sns.violinplot(data=plot_data, x='DST', y='Altitude', order=['E', 'N', 'U'],
               inner=None, color='lightblue')
sns.boxplot(data=plot_data, x='DST', y='Altitude', order=['E', 'N', 'U'],
            width=0.3, boxprops=dict(facecolor='white', edgecolor='black'),
            whiskerprops=dict(color='black'), capprops=dict(color='black'),
            medianprops=dict(color='red', linewidth=2))

plt.xlabel('DST Category', fontsize=12, fontweight='bold')
plt.ylabel('Altitude (feet)', fontsize=12, fontweight='bold')
plt.title('Altitude Distribution by DST Category', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()

# Save plot
plt.savefig('altitude_by_dst_violin_plot.png', dpi=300, bbox_inches='tight')
print("Violin plot saved as: altitude_by_dst_violin_plot.png")
print()

print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
