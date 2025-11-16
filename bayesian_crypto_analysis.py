import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Set random seed for reproducibility
np.random.seed(42)

# Load the three cryptocurrency datasets
print("Loading datasets...")
df_aave = pd.read_csv('coin_Aave.csv')
df_polkadot = pd.read_csv('coin_Polkadot.csv')
df_uniswap = pd.read_csv('coin_Uniswap.csv')

# Merge datasets on Date column using inner join
print("Merging datasets...")
df_merged = df_aave[['Date', 'Close', 'Open', 'High', 'Low', 'Volume']].rename(
    columns={'Close': 'Aave_Close', 'Open': 'Aave_Open', 'High': 'Aave_High',
             'Low': 'Aave_Low', 'Volume': 'Aave_Volume'}
)

df_merged = df_merged.merge(
    df_polkadot[['Date', 'Close', 'Open', 'High', 'Low', 'Volume']].rename(
        columns={'Close': 'Polkadot_Close', 'Open': 'Polkadot_Open',
                 'High': 'Polkadot_High', 'Low': 'Polkadot_Low', 'Volume': 'Polkadot_Volume'}
    ),
    on='Date',
    how='inner'
)

df_merged = df_merged.merge(
    df_uniswap[['Date', 'Close', 'Open', 'High', 'Low', 'Volume']].rename(
        columns={'Close': 'Uniswap_Close', 'Open': 'Uniswap_Open',
                 'High': 'Uniswap_High', 'Low': 'Uniswap_Low', 'Volume': 'Uniswap_Volume'}
    ),
    on='Date',
    how='inner'
)

print(f"Merged dataset shape: {df_merged.shape}")

# Create binary target variable for Aave (1 if Close > Open, 0 otherwise)
df_merged['Aave_Direction'] = (df_merged['Aave_Close'] > df_merged['Aave_Open']).astype(int)

# Create predictor variables
# 1. Polkadot daily returns: (Close - Open) / Open
df_merged['Polkadot_Returns'] = (df_merged['Polkadot_Close'] - df_merged['Polkadot_Open']) / df_merged['Polkadot_Open']

# 2. Uniswap daily returns: (Close - Open) / Open
df_merged['Uniswap_Returns'] = (df_merged['Uniswap_Close'] - df_merged['Uniswap_Open']) / df_merged['Uniswap_Open']

# 3. Normalized Aave volume: Volume / max(Volume)
df_merged['Aave_Volume_Norm'] = df_merged['Aave_Volume'] / df_merged['Aave_Volume'].max()

# Standardize predictor variables using z-score normalization (ddof=1)
polkadot_mean = df_merged['Polkadot_Returns'].mean()
polkadot_std = df_merged['Polkadot_Returns'].std(ddof=1)
df_merged['Polkadot_Returns_Std'] = (df_merged['Polkadot_Returns'] - polkadot_mean) / polkadot_std

uniswap_mean = df_merged['Uniswap_Returns'].mean()
uniswap_std = df_merged['Uniswap_Returns'].std(ddof=1)
df_merged['Uniswap_Returns_Std'] = (df_merged['Uniswap_Returns'] - uniswap_mean) / uniswap_std

volume_mean = df_merged['Aave_Volume_Norm'].mean()
volume_std = df_merged['Aave_Volume_Norm'].std(ddof=1)
df_merged['Aave_Volume_Std'] = (df_merged['Aave_Volume_Norm'] - volume_mean) / volume_std

print("\nStandardization parameters:")
print(f"Polkadot Returns - Mean: {polkadot_mean}, Std: {polkadot_std}")
print(f"Uniswap Returns - Mean: {uniswap_mean}, Std: {uniswap_std}")
print(f"Aave Volume - Mean: {volume_mean}, Std: {volume_std}")

# Prepare data for Bayesian logistic regression
X = df_merged[['Polkadot_Returns_Std', 'Uniswap_Returns_Std', 'Aave_Volume_Std']].values
y = df_merged['Aave_Direction'].values
n_samples, n_features = X.shape

print(f"\nDataset for modeling: {n_samples} samples, {n_features} features")

# Bayesian Logistic Regression with Metropolis-Hastings MCMC

def sigmoid(z):
    """Sigmoid function for logistic regression"""
    return 1 / (1 + np.exp(-z))

def log_prior(beta, intercept):
    """Log prior probability: N(0, 2.5) for coefficients, N(0, 10) for intercept"""
    log_p_beta = -0.5 * np.sum(beta**2 / 2.5**2)
    log_p_intercept = -0.5 * (intercept**2 / 10**2)
    return log_p_beta + log_p_intercept

def log_likelihood(beta, intercept, X, y):
    """Log likelihood for logistic regression"""
    linear_pred = intercept + np.dot(X, beta)
    p = sigmoid(linear_pred)
    # Avoid log(0) by clipping
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

def log_posterior(beta, intercept, X, y):
    """Log posterior probability (unnormalized)"""
    return log_prior(beta, intercept) + log_likelihood(beta, intercept, X, y)

# Metropolis-Hastings MCMC
print("\nRunning Metropolis-Hastings MCMC...")
n_iterations = 15000
burn_in = 5000
proposal_std = 0.15

# Initialize parameters
beta_current = np.zeros(n_features)
intercept_current = 0.0

# Storage for posterior samples
beta_samples = np.zeros((n_iterations - burn_in, n_features))
intercept_samples = np.zeros(n_iterations - burn_in)

# MCMC sampling
accepted = 0
for i in range(n_iterations):
    # Propose new parameters
    beta_proposal = beta_current + np.random.normal(0, proposal_std, n_features)
    intercept_proposal = intercept_current + np.random.normal(0, proposal_std)

    # Compute acceptance probability
    log_p_current = log_posterior(beta_current, intercept_current, X, y)
    log_p_proposal = log_posterior(beta_proposal, intercept_proposal, X, y)

    log_acceptance_ratio = log_p_proposal - log_p_current

    # Accept or reject
    if np.log(np.random.uniform()) < log_acceptance_ratio:
        beta_current = beta_proposal
        intercept_current = intercept_proposal
        accepted += 1

    # Store samples after burn-in
    if i >= burn_in:
        beta_samples[i - burn_in] = beta_current
        intercept_samples[i - burn_in] = intercept_current

    if (i + 1) % 3000 == 0:
        print(f"Iteration {i + 1}/{n_iterations}")

acceptance_rate = accepted / n_iterations
print(f"\nAcceptance rate: {acceptance_rate:.4f}")

# Compute posterior statistics
print("\n" + "="*60)
print("POSTERIOR STATISTICS")
print("="*60)

# Mean posterior coefficients (6 decimal places)
mean_beta_polkadot = beta_samples[:, 0].mean()
mean_beta_uniswap = beta_samples[:, 1].mean()
mean_beta_volume = beta_samples[:, 2].mean()
mean_intercept = intercept_samples.mean()

print(f"\nMean Posterior Coefficients:")
print(f"Polkadot Returns: {mean_beta_polkadot:.6f}")
print(f"Uniswap Returns: {mean_beta_uniswap:.6f}")
print(f"Aave Volume: {mean_beta_volume:.6f}")
print(f"Intercept: {mean_intercept:.6f}")

# 95% equal-tailed credible intervals (2.5th and 97.5th percentiles)
ci_polkadot = np.percentile(beta_samples[:, 0], [2.5, 97.5])
ci_uniswap = np.percentile(beta_samples[:, 1], [2.5, 97.5])
ci_volume = np.percentile(beta_samples[:, 2], [2.5, 97.5])

print(f"\n95% Credible Intervals:")
print(f"Polkadot Returns: [{ci_polkadot[0]:.4f}, {ci_polkadot[1]:.4f}]")
print(f"Uniswap Returns: [{ci_uniswap[0]:.4f}, {ci_uniswap[1]:.4f}]")
print(f"Aave Volume: [{ci_volume[0]:.4f}, {ci_volume[1]:.4f}]")

# Generate predictions using mean posterior coefficients
linear_pred = mean_intercept + np.dot(X, np.array([mean_beta_polkadot, mean_beta_uniswap, mean_beta_volume]))
predicted_probs = sigmoid(linear_pred)

# Convert to binary predictions (threshold = 0.5)
predicted_classes = (predicted_probs > 0.5).astype(int)

# Compute classification accuracy
accuracy = (predicted_classes == y).mean()
print(f"\nClassification Accuracy: {accuracy:.4f}")

# Posterior predictive probability for new observation
# Polkadot returns = 0.05, Uniswap returns = -0.02, Aave volume = 0.3
new_polkadot = 0.05
new_uniswap = -0.02
new_volume = 0.3

# Apply same standardization
new_polkadot_std = (new_polkadot - polkadot_mean) / polkadot_std
new_uniswap_std = (new_uniswap - uniswap_mean) / uniswap_std
new_volume_std = (new_volume - volume_mean) / volume_std

new_obs = np.array([new_polkadot_std, new_uniswap_std, new_volume_std])

# Compute posterior predictive probability using all posterior samples
posterior_pred_probs = []
for i in range(len(beta_samples)):
    linear_pred_new = intercept_samples[i] + np.dot(new_obs, beta_samples[i])
    prob = sigmoid(linear_pred_new)
    posterior_pred_probs.append(prob)

# Mean of posterior predictive distribution
posterior_pred_prob = np.mean(posterior_pred_probs)
print(f"\nPosterior Predictive Probability for New Observation:")
print(f"  Polkadot returns: {new_polkadot}")
print(f"  Uniswap returns: {new_uniswap}")
print(f"  Aave volume: {new_volume}")
print(f"  Probability: {posterior_pred_prob:.5f}")

# Generate scatter plot
print("\nGenerating scatter plot...")
plt.figure(figsize=(10, 6))
plt.scatter(df_merged['Polkadot_Returns_Std'], df_merged['Uniswap_Returns_Std'],
            alpha=0.6, c='blue', edgecolors='black', linewidth=0.5)
plt.xlabel('Polkadot Standardized Returns', fontsize=12)
plt.ylabel('Uniswap Standardized Returns', fontsize=12)
plt.title('Cryptocurrency Standardized Returns: Polkadot vs Uniswap', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('crypto_returns_scatter.png', dpi=300, bbox_inches='tight')
print("Scatter plot saved as 'crypto_returns_scatter.png'")

print("\n" + "="*60)
print("SUMMARY OF RESULTS")
print("="*60)
print(f"\n1. Mean Posterior Coefficients (6 decimals):")
print(f"   Polkadot Returns: {mean_beta_polkadot:.6f}")
print(f"   Uniswap Returns: {mean_beta_uniswap:.6f}")
print(f"   Aave Volume: {mean_beta_volume:.6f}")

print(f"\n2. 95% Credible Intervals (4 decimals for bounds):")
print(f"   Polkadot: [{ci_polkadot[0]:.4f}, {ci_polkadot[1]:.4f}]")
print(f"   Uniswap: [{ci_uniswap[0]:.4f}, {ci_uniswap[1]:.4f}]")

print(f"\n3. Classification Accuracy (4 decimals): {accuracy:.4f}")

print(f"\n4. Posterior Predictive Probability (5 decimals): {posterior_pred_prob:.5f}")

print("\nAnalysis complete!")
