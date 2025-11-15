"""
Hierarchical Bayesian Logistic Regression Analysis on Amazon Review Data
Author: Data Science Team
Date: 2025-11-15
"""

import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("="*80)
print("HIERARCHICAL BAYESIAN LOGISTIC REGRESSION ANALYSIS")
print("="*80)
print()

# ============================================================================
# STEP 1: LOAD AND PREPROCESS DATA
# ============================================================================
print("STEP 1: Loading and preprocessing datasets...")
print("-"*80)

# Load the three datasets
print("Loading amazon_ebook_Data.csv...")
df_ebook = pd.read_csv('amazon_ebook_Data.csv')
print(f"  - Loaded {len(df_ebook)} records")

print("Loading amazon_grocery_Data.csv...")
df_grocery = pd.read_csv('amazon_grocery_Data.csv')
print(f"  - Loaded {len(df_grocery)} records")

print("Loading amazon_books_Data.csv...")
df_books = pd.read_csv('amazon_books_Data.csv')
print(f"  - Loaded {len(df_books)} records")
print()

# Rename sentiment columns to unified name
print("Standardizing sentiment column names...")
df_ebook = df_ebook.rename(columns={'Sentiment_ebook': 'sentiment_label'})
df_grocery = df_grocery.rename(columns={'Sentiment_grcry': 'sentiment_label'})
df_books = df_books.rename(columns={'Sentiment_books': 'sentiment_label'})
print("  - Renamed Sentiment_ebook -> sentiment_label")
print("  - Renamed Sentiment_grcry -> sentiment_label")
print("  - Renamed Sentiment_books -> sentiment_label")
print()

# Create category_code for each dataset
print("Creating category_code variable...")
df_ebook['category_code'] = 0
df_grocery['category_code'] = 1
df_books['category_code'] = 2
print("  - category_code 0: ebook reviews")
print("  - category_code 1: grocery reviews")
print("  - category_code 2: book reviews")
print()

# Concatenate datasets vertically
print("Concatenating datasets...")
df_combined = pd.concat([df_ebook, df_grocery, df_books], axis=0, ignore_index=True)
print(f"  - Combined dataset: {len(df_combined)} total records")
print()

# Retain only required columns
print("Retaining required columns for modeling...")
df_model = df_combined[['star_rating', 'helpful_votes', 'total_votes', 'category_code']].copy()
print(f"  - Columns retained: {list(df_model.columns)}")
print(f"  - Final dataset shape: {df_model.shape}")
print()

# Create binary star_rating (1 if star_rating==1, 0 otherwise)
print("Creating binary outcome variable...")
df_model['binary_rating'] = (df_model['star_rating'] == 1).astype(int)
print(f"  - Binary rating distribution:")
print(f"    * star_rating=1: {df_model['binary_rating'].sum()} records")
print(f"    * star_rating≠1: {(1-df_model['binary_rating']).sum()} records")
print()

print("Category distribution:")
print(df_model['category_code'].value_counts().sort_index())
print()

# ============================================================================
# STEP 2: HIERARCHICAL BAYESIAN LOGISTIC REGRESSION
# ============================================================================
print("="*80)
print("STEP 2: Hierarchical Bayesian Logistic Regression Model")
print("="*80)
print()

# Prepare data for modeling
y = df_model['binary_rating'].values
X = df_model['helpful_votes'].values
category = df_model['category_code'].values
n_categories = len(np.unique(category))

print(f"Model configuration:")
print(f"  - Dependent variable: binary star_rating (1 vs not-1)")
print(f"  - Independent variable: helpful_votes")
print(f"  - Hierarchical grouping: category_code ({n_categories} categories)")
print(f"  - Likelihood: Bernoulli with logit link")
print(f"  - Partial pooling: Group-level intercepts and slopes")
print()

print("Hyperprior specifications:")
print("  - Population intercept mean: Normal(μ=0, σ=10)")
print("  - Population slope mean: Normal(μ=0, σ=5)")
print("  - Between-group std (intercepts): HalfCauchy(β=5)")
print("  - Between-group std (slopes): HalfCauchy(β=5)")
print()

print("MCMC settings:")
print("  - Iterations: 5000")
print("  - Burn-in (tune): 1000")
print("  - Random seed: 42")
print()

# Build hierarchical model
print("Building PyMC model...")
with pm.Model() as hierarchical_model:
    # Hyperpriors for population-level parameters
    mu_intercept = pm.Normal('mu_intercept', mu=0, sigma=10)
    mu_slope = pm.Normal('mu_slope', mu=0, sigma=5)

    # Between-group standard deviations
    sigma_intercept = pm.HalfCauchy('sigma_intercept', beta=5)
    sigma_slope = pm.HalfCauchy('sigma_slope', beta=5)

    # Group-level intercepts and slopes (partial pooling)
    intercept_offset = pm.Normal('intercept_offset', mu=0, sigma=1, shape=n_categories)
    slope_offset = pm.Normal('slope_offset', mu=0, sigma=1, shape=n_categories)

    # Non-centered parameterization for better sampling
    intercept = pm.Deterministic('intercept', mu_intercept + sigma_intercept * intercept_offset)
    slope = pm.Deterministic('slope', mu_slope + sigma_slope * slope_offset)

    # Linear predictor
    logit_p = intercept[category] + slope[category] * X

    # Likelihood
    y_obs = pm.Bernoulli('y_obs', logit_p=logit_p, observed=y)

    print("Model structure:")
    print(hierarchical_model)
    print()

    # Sample from posterior
    print("Sampling from posterior distribution...")
    print("This may take several minutes...")
    trace = pm.sample(
        draws=5000,
        tune=1000,
        random_seed=42,
        return_inferencedata=True,
        progressbar=True,
        chains=4,
        cores=4,
        target_accept=0.99,
        init='adapt_diag'
    )

print()
print("Sampling completed!")
print()

# ============================================================================
# STEP 3: CONVERGENCE DIAGNOSTICS AND POSTERIOR STATISTICS
# ============================================================================
print("="*80)
print("STEP 3: Convergence Diagnostics and Posterior Statistics")
print("="*80)
print()

# Check convergence
print("Convergence diagnostics (R-hat values):")
print("-"*80)
summary = az.summary(trace, var_names=['mu_intercept', 'mu_slope',
                                        'sigma_intercept', 'sigma_slope',
                                        'intercept', 'slope'])
print(summary)
print()

# Get R-hat values
rhat_values = summary['r_hat'].values
max_rhat = rhat_values.max()
print(f"Maximum R-hat across all parameters: {max_rhat:.3f}")

if max_rhat < 1.01:
    print("✓ CONVERGENCE ACHIEVED (all R-hat < 1.01)")
else:
    print("⚠ WARNING: Some parameters have R-hat >= 1.01")
print()

# ============================================================================
# STEP 4: EXTRACT REQUIRED POSTERIOR STATISTICS
# ============================================================================
print("="*80)
print("STEP 4: Extracting Required Posterior Statistics")
print("="*80)
print()

# Extract posterior means
posterior = trace.posterior

# Group-level intercept for category_code 0
intercept_cat0_mean = posterior['intercept'].sel(intercept_dim_0=0).mean().values.item()
print(f"1. Group-level intercept for category_code 0: {intercept_cat0_mean:.4f}")

# Group-level slope for category_code 1
slope_cat1_mean = posterior['slope'].sel(slope_dim_0=1).mean().values.item()
print(f"2. Group-level slope for category_code 1: {slope_cat1_mean:.4f}")

# Between-group standard deviation for intercepts
sigma_intercept_mean = posterior['sigma_intercept'].mean().values.item()
print(f"3. Between-group std deviation (intercepts): {sigma_intercept_mean:.5f}")

# Population-level intercept mean
mu_intercept_mean = posterior['mu_intercept'].mean().values.item()
print(f"4. Population-level intercept mean: {mu_intercept_mean:.4f}")

# Maximum R-hat
print(f"5. Maximum R-hat value: {max_rhat:.3f}")

# Effective sample size for slope of category_code 2
slope_cat2_ess = summary.loc['slope[2]', 'ess_bulk']
print(f"6. Effective sample size for slope (category_code 2): {int(slope_cat2_ess)}")
print()

# ============================================================================
# STEP 5: CREATE HEXBIN PLOT
# ============================================================================
print("="*80)
print("STEP 5: Creating Hexbin Visualization")
print("="*80)
print()

print("Generating hexbin plot...")
print("  - X-axis: helpful_votes")
print("  - Y-axis: total_votes")
print("  - Grid size: 25")
print("  - Color scale: Logarithmic")

fig, ax = plt.subplots(figsize=(10, 8))
hb = ax.hexbin(
    df_model['helpful_votes'],
    df_model['total_votes'],
    gridsize=25,
    cmap='viridis',
    norm=LogNorm(),
    mincnt=1
)

ax.set_xlabel('Helpful Votes', fontsize=12)
ax.set_ylabel('Total Votes', fontsize=12)
ax.set_title('Hexbin Plot: Helpful Votes vs Total Votes\n(Logarithmic Color Scale)',
             fontsize=14, fontweight='bold')

# Add colorbar
cb = plt.colorbar(hb, ax=ax)
cb.set_label('Count (log scale)', fontsize=11)

plt.tight_layout()
plt.savefig('hexbin_helpful_vs_total_votes.png', dpi=300, bbox_inches='tight')
print("  ✓ Hexbin plot saved: hexbin_helpful_vs_total_votes.png")
print()

# ============================================================================
# STEP 6: FINAL SUMMARY REPORT
# ============================================================================
print("="*80)
print("FINAL SUMMARY REPORT")
print("="*80)
print()

print("REQUESTED POSTERIOR STATISTICS:")
print("-"*80)
print(f"1. Group-level intercept for category_code 0:      {intercept_cat0_mean:.4f}")
print(f"2. Group-level slope for category_code 1:          {slope_cat1_mean:.4f}")
print(f"3. Between-group std deviation (intercepts):       {sigma_intercept_mean:.5f}")
print(f"4. Population-level intercept mean:                {mu_intercept_mean:.4f}")
print(f"5. Maximum R-hat value:                            {max_rhat:.3f}")
print(f"6. Effective sample size (slope, category_code 2): {int(slope_cat2_ess)}")
print()

print("="*80)
print("ANALYSIS COMPLETE")
print("="*80)
