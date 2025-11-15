import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 80)
print("AMAZON REVIEWS BAYESIAN ANALYSIS (IMPROVED CONVERGENCE)")
print("=" * 80)

# Load processed data
print("\n[1/3] Loading preprocessed data...")
combined_df = pd.read_csv('amazon_combined_processed.csv')
print(f"  - Combined dataset: {len(combined_df)} rows")

# ============================================================================
# Fit Bayesian Logistic Regression with improved sampling parameters
# ============================================================================
print("\n[2/3] Fitting Bayesian Logistic Regression model (improved)...")

import pymc as pm
import arviz as az

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

    # Sample from posterior with improved parameters
    print("  - Running MCMC sampling (5000 iterations, 1000 burn-in)...")
    print("  - Using 4 chains with higher target_accept for better convergence...")
    trace = pm.sample(
        draws=4000,       # 5000 - 1000 burn-in = 4000 post-burn-in
        tune=1000,        # burn-in samples
        chains=4,         # Increased from 2 to 4 chains
        target_accept=0.95,  # Increased target_accept to reduce divergences
        random_seed=42,
        return_inferencedata=True,
        progressbar=True
    )

print("\n  - MCMC sampling completed")

# ========================================================================
# Check convergence and extract metrics
# ========================================================================
print("\n[3/3] Checking convergence and extracting metrics...")

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

# Calculate metrics
# 1. Posterior mean odds ratio for star_rating
odds_ratio_star_rating = np.exp(posterior_star_rating.mean())

# 2. 95% credible interval for total_votes coefficient
ci_total_votes_lower = np.percentile(posterior_total_votes, 2.5)
ci_total_votes_upper = np.percentile(posterior_total_votes, 97.5)

# 3. Effective sample size for helpful_votes
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

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
