"""
Amazon Reviews Bayesian Logistic Regression Analysis
=====================================================
This script performs Bayesian logistic regression on combined Amazon review datasets
(ebooks, grocery, books) to model sentiment and visualize engagement patterns.

Author: Bayesian Analysis Pipeline
Date: 2025-11-15
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pymc as pm
import arviz as az
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 80)
print("AMAZON REVIEWS BAYESIAN LOGISTIC REGRESSION ANALYSIS")
print("=" * 80)

# ============================================================================
# STEP 1: Load datasets
# ============================================================================
print("\n[1/7] Loading datasets...")

ebook_df = pd.read_csv('amazon_ebook_Data.csv', index_col=0)
grocery_df = pd.read_csv('amazon_grocery_Data.csv', index_col=0)
books_df = pd.read_csv('amazon_books_Data.csv', index_col=0)

print(f"  - Ebook reviews: {len(ebook_df)} rows")
print(f"  - Grocery reviews: {len(grocery_df)} rows")
print(f"  - Books reviews: {len(books_df)} rows")

# ============================================================================
# STEP 2: Rename sentiment columns and handle typos
# ============================================================================
print("\n[2/7] Preprocessing data...")

# Rename sentiment columns to unified 'sentiment_label'
ebook_df = ebook_df.rename(columns={'Sentiment_ebook': 'sentiment_label'})
grocery_df = grocery_df.rename(columns={'Sentiment_grcry': 'sentiment_label'})
books_df = books_df.rename(columns={'Sentiment_books': 'sentiment_label'})

# Handle typo 'negaitve' → 'negative'
for df in [ebook_df, grocery_df, books_df]:
    if 'sentiment_label' in df.columns:
        df['sentiment_label'] = df['sentiment_label'].replace('negaitve', 'negative')

# Add dataset_source column
ebook_df['dataset_source'] = 'ebook'
grocery_df['dataset_source'] = 'grocery'
books_df['dataset_source'] = 'books'

print("  - Renamed sentiment columns to 'sentiment_label'")
print("  - Fixed typo 'negaitve' → 'negative'")
print("  - Added dataset_source column")

# ============================================================================
# STEP 3: Clean data - remove missing values
# ============================================================================
print("\n[3/7] Cleaning data...")

# Select relevant columns
columns_to_keep = ['star_rating', 'helpful_votes', 'total_votes', 'sentiment_label', 'dataset_source']

ebook_clean = ebook_df[columns_to_keep].copy()
grocery_clean = grocery_df[columns_to_keep].copy()
books_clean = books_df[columns_to_keep].copy()

# Remove missing values in required columns
ebook_clean = ebook_clean.dropna(subset=['star_rating', 'helpful_votes', 'total_votes', 'sentiment_label'])
grocery_clean = grocery_clean.dropna(subset=['star_rating', 'helpful_votes', 'total_votes', 'sentiment_label'])
books_clean = books_clean.dropna(subset=['star_rating', 'helpful_votes', 'total_votes', 'sentiment_label'])

print(f"  - Ebook after cleaning: {len(ebook_clean)} rows")
print(f"  - Grocery after cleaning: {len(grocery_clean)} rows")
print(f"  - Books after cleaning: {len(books_clean)} rows")

# ============================================================================
# STEP 4: Concatenate datasets and encode sentiment
# ============================================================================
print("\n[4/7] Concatenating datasets and encoding sentiment...")

# Concatenate vertically
combined_df = pd.concat([ebook_clean, grocery_clean, books_clean], axis=0, ignore_index=True)
print(f"  - Combined dataset: {len(combined_df)} rows")

# Encode sentiment_label: positive=1, negative=0
combined_df['sentiment_encoded'] = combined_df['sentiment_label'].map({'positive': 1, 'negative': 0})

# Remove any rows where sentiment encoding failed
combined_df = combined_df.dropna(subset=['sentiment_encoded'])
combined_df['sentiment_encoded'] = combined_df['sentiment_encoded'].astype(int)

print(f"  - Sentiment distribution: {combined_df['sentiment_encoded'].value_counts().to_dict()}")
print(f"  - Dataset source distribution: {combined_df['dataset_source'].value_counts().to_dict()}")

# ============================================================================
# STEP 5: Standardize features (z-score normalization)
# ============================================================================
print("\n[5/7] Standardizing features...")

# Standardize star_rating, helpful_votes, total_votes using z-score
combined_df['star_rating_std'] = stats.zscore(combined_df['star_rating'])
combined_df['helpful_votes_std'] = stats.zscore(combined_df['helpful_votes'])
combined_df['total_votes_std'] = stats.zscore(combined_df['total_votes'])

print("  - Standardized star_rating, helpful_votes, and total_votes")
print(f"  - Star rating std: mean={combined_df['star_rating_std'].mean():.6f}, std={combined_df['star_rating_std'].std():.6f}")
print(f"  - Helpful votes std: mean={combined_df['helpful_votes_std'].mean():.6f}, std={combined_df['helpful_votes_std'].std():.6f}")
print(f"  - Total votes std: mean={combined_df['total_votes_std'].mean():.6f}, std={combined_df['total_votes_std'].std():.6f}")

# Create dummy variables for dataset_source (ebook as reference category)
combined_df['is_grocery'] = (combined_df['dataset_source'] == 'grocery').astype(int)
combined_df['is_books'] = (combined_df['dataset_source'] == 'books').astype(int)

# Save processed data
combined_df.to_csv('amazon_combined_processed.csv', index=False)
print("  - Saved processed data to 'amazon_combined_processed.csv'")

# ============================================================================
# STEP 6: Fit Bayesian Logistic Regression using PyMC
# ============================================================================
print("\n[6/7] Fitting Bayesian Logistic Regression model...")

# Prepare data for modeling
X = combined_df[['star_rating_std', 'helpful_votes_std', 'total_votes_std', 'is_grocery', 'is_books']].values
y = combined_df['sentiment_encoded'].values

# Build Bayesian logistic regression model
with pm.Model() as logistic_model:
    # Priors
    # Cauchy(0, 10) for intercept
    intercept = pm.Cauchy('intercept', alpha=0, beta=10)

    # Cauchy(0, 2.5) for coefficients
    beta_star_rating = pm.Cauchy('beta_star_rating', alpha=0, beta=2.5)
    beta_helpful_votes = pm.Cauchy('beta_helpful_votes', alpha=0, beta=2.5)
    beta_total_votes = pm.Cauchy('beta_total_votes', alpha=0, beta=2.5)
    beta_grocery = pm.Cauchy('beta_grocery', alpha=0, beta=2.5)
    beta_books = pm.Cauchy('beta_books', alpha=0, beta=2.5)

    # Linear combination
    logit_p = (intercept +
               beta_star_rating * X[:, 0] +
               beta_helpful_votes * X[:, 1] +
               beta_total_votes * X[:, 2] +
               beta_grocery * X[:, 3] +
               beta_books * X[:, 4])

    # Likelihood
    y_obs = pm.Bernoulli('y_obs', logit_p=logit_p, observed=y)

    # Sample from posterior
    # 5000 iterations total: 1000 burn-in + 4000 post-burn-in
    print("  - Running MCMC sampling (5000 iterations, 1000 burn-in)...")
    print("  - Using 4 chains with target_accept=0.95 for convergence...")
    trace = pm.sample(
        draws=4000,           # post-burn-in draws
        tune=1000,            # burn-in samples
        chains=4,             # number of chains
        target_accept=0.95,   # high target_accept for better convergence
        random_seed=42,       # reproducibility
        return_inferencedata=True,
        progressbar=True
    )

print("\n  - MCMC sampling completed")

# ============================================================================
# STEP 7: Check convergence and extract metrics
# ============================================================================
print("\n[7/7] Checking convergence and extracting metrics...")

# Check R-hat (should be < 1.01)
summary = az.summary(trace, var_names=['intercept', 'beta_star_rating', 'beta_helpful_votes',
                                       'beta_total_votes', 'beta_grocery', 'beta_books'])

print("\n" + "=" * 80)
print("MODEL CONVERGENCE DIAGNOSTICS")
print("=" * 80)
print(summary[['mean', 'sd', 'hdi_3%', 'hdi_97%', 'r_hat', 'ess_bulk', 'ess_tail']])

# Check if all R-hat < 1.01
max_rhat = summary['r_hat'].max()
print(f"\nMaximum R-hat: {max_rhat:.6f}")
if max_rhat < 1.01:
    print("✓ Model converged successfully (all R-hat < 1.01)")
    convergence_status = "CONVERGED"
else:
    print("✗ Warning: Model may not have converged (R-hat >= 1.01)")
    convergence_status = "NOT CONVERGED"

# Extract posterior samples
posterior_star_rating = trace.posterior['beta_star_rating'].values.flatten()
posterior_total_votes = trace.posterior['beta_total_votes'].values.flatten()
posterior_helpful_votes = trace.posterior['beta_helpful_votes'].values.flatten()

# Calculate requested metrics
# 1. Posterior mean odds ratio for star_rating (3 decimals)
odds_ratio_star_rating = np.exp(posterior_star_rating.mean())

# 2. 95% credible interval for total_votes coefficient (4 decimals)
ci_total_votes_lower = np.percentile(posterior_total_votes, 2.5)
ci_total_votes_upper = np.percentile(posterior_total_votes, 97.5)

# 3. Effective sample size for helpful_votes (integer)
ess_helpful_votes = int(summary.loc['beta_helpful_votes', 'ess_bulk'])

print("\n" + "=" * 80)
print("REQUESTED METRICS")
print("=" * 80)
print(f"Posterior mean odds ratio for star_rating: {odds_ratio_star_rating:.3f}")
print(f"95% credible interval for total_votes:")
print(f"  - Lower bound: {ci_total_votes_lower:.4f}")
print(f"  - Upper bound: {ci_total_votes_upper:.4f}")
print(f"Effective sample size for helpful_votes: {ess_helpful_votes}")

# Save results to file
with open('bayesian_results_final.txt', 'w') as f:
    f.write("BAYESIAN LOGISTIC REGRESSION RESULTS (FINAL)\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"CONVERGENCE STATUS: {convergence_status}\n")
    f.write(f"Maximum R-hat: {max_rhat:.6f}\n\n")
    f.write("REQUESTED METRICS:\n")
    f.write(f"Posterior mean odds ratio for star_rating: {odds_ratio_star_rating:.3f}\n")
    f.write(f"95% credible interval lower bound for total_votes: {ci_total_votes_lower:.4f}\n")
    f.write(f"95% credible interval upper bound for total_votes: {ci_total_votes_upper:.4f}\n")
    f.write(f"Effective sample size for helpful_votes: {ess_helpful_votes}\n")
    f.write("\n" + "=" * 80 + "\n")
    f.write("MODEL SUMMARY:\n")
    f.write("=" * 80 + "\n")
    f.write(summary.to_string())

print("\n  - Results saved to 'bayesian_results_final.txt'")

# Save trace for later analysis
trace.to_netcdf('bayesian_trace_final.nc')
print("  - Trace saved to 'bayesian_trace_final.nc'")

# ============================================================================
# VISUALIZATION: Create hexbin plot
# ============================================================================
print("\n[VISUALIZATION] Creating hexbin plot...")

plt.figure(figsize=(10, 8))
hexbin = plt.hexbin(
    combined_df['helpful_votes'],
    combined_df['total_votes'],
    gridsize=25,
    cmap='YlOrRd',
    norm=plt.matplotlib.colors.LogNorm(),
    mincnt=1
)

plt.colorbar(hexbin, label='Count (log scale)')
plt.xlabel('Helpful Votes', fontsize=12)
plt.ylabel('Total Votes', fontsize=12)
plt.title('Amazon Reviews Engagement Pattern: Helpful Votes vs Total Votes',
          fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig('hexbin_engagement_pattern.png', dpi=300, bbox_inches='tight')
print("  - Hexbin plot saved to 'hexbin_engagement_pattern.png'")

plt.close()

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print("\nGenerated files:")
print("  1. amazon_combined_processed.csv - Processed and standardized data")
print("  2. bayesian_results_final.txt - Complete results summary")
print("  3. bayesian_trace_final.nc - MCMC trace for further analysis")
print("  4. hexbin_engagement_pattern.png - Engagement visualization")
print("\nKey Results:")
print(f"  - Odds ratio (star_rating): {odds_ratio_star_rating:.3f}")
print(f"  - 95% CI (total_votes): [{ci_total_votes_lower:.4f}, {ci_total_votes_upper:.4f}]")
print(f"  - ESS (helpful_votes): {ess_helpful_votes}")
print(f"  - Convergence: {convergence_status}")
