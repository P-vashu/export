import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Load and merge datasets
df_aave = pd.read_csv('coin_Aave.csv')
df_polkadot = pd.read_csv('coin_Polkadot.csv')
df_uniswap = pd.read_csv('coin_Uniswap.csv')

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

# Create target and predictors
df_merged['Aave_Direction'] = (df_merged['Aave_Close'] > df_merged['Aave_Open']).astype(int)
df_merged['Polkadot_Returns'] = (df_merged['Polkadot_Close'] - df_merged['Polkadot_Open']) / df_merged['Polkadot_Open']
df_merged['Uniswap_Returns'] = (df_merged['Uniswap_Close'] - df_merged['Uniswap_Open']) / df_merged['Uniswap_Open']
df_merged['Aave_Volume_Norm'] = df_merged['Aave_Volume'] / df_merged['Aave_Volume'].max()

# Standardize
polkadot_mean = df_merged['Polkadot_Returns'].mean()
polkadot_std = df_merged['Polkadot_Returns'].std(ddof=1)
df_merged['Polkadot_Returns_Std'] = (df_merged['Polkadot_Returns'] - polkadot_mean) / polkadot_std

uniswap_mean = df_merged['Uniswap_Returns'].mean()
uniswap_std = df_merged['Uniswap_Returns'].std(ddof=1)
df_merged['Uniswap_Returns_Std'] = (df_merged['Uniswap_Returns'] - uniswap_mean) / uniswap_std

volume_mean = df_merged['Aave_Volume_Norm'].mean()
volume_std = df_merged['Aave_Volume_Norm'].std(ddof=1)
df_merged['Aave_Volume_Std'] = (df_merged['Aave_Volume_Norm'] - volume_mean) / volume_std

X = df_merged[['Polkadot_Returns_Std', 'Uniswap_Returns_Std', 'Aave_Volume_Std']].values
y = df_merged['Aave_Direction'].values
n_samples, n_features = X.shape

# Bayesian Logistic Regression functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def log_prior(beta, intercept):
    log_p_beta = -0.5 * np.sum(beta**2 / 2.5**2)
    log_p_intercept = -0.5 * (intercept**2 / 10**2)
    return log_p_beta + log_p_intercept

def log_likelihood(beta, intercept, X, y):
    linear_pred = intercept + np.dot(X, beta)
    p = sigmoid(linear_pred)
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

def log_posterior(beta, intercept, X, y):
    return log_prior(beta, intercept) + log_likelihood(beta, intercept, X, y)

# MCMC
n_iterations = 15000
burn_in = 5000
proposal_std = 0.15

beta_current = np.zeros(n_features)
intercept_current = 0.0

beta_samples = np.zeros((n_iterations - burn_in, n_features))
intercept_samples = np.zeros(n_iterations - burn_in)

for i in range(n_iterations):
    beta_proposal = beta_current + np.random.normal(0, proposal_std, n_features)
    intercept_proposal = intercept_current + np.random.normal(0, proposal_std)

    log_p_current = log_posterior(beta_current, intercept_current, X, y)
    log_p_proposal = log_posterior(beta_proposal, intercept_proposal, X, y)
    log_acceptance_ratio = log_p_proposal - log_p_current

    if np.log(np.random.uniform()) < log_acceptance_ratio:
        beta_current = beta_proposal
        intercept_current = intercept_proposal

    if i >= burn_in:
        beta_samples[i - burn_in] = beta_current
        intercept_samples[i - burn_in] = intercept_current

# Compute statistics
mean_beta_polkadot = beta_samples[:, 0].mean()
mean_beta_uniswap = beta_samples[:, 1].mean()
mean_beta_volume = beta_samples[:, 2].mean()
mean_intercept = intercept_samples.mean()

ci_polkadot = np.percentile(beta_samples[:, 0], [2.5, 97.5])
ci_uniswap = np.percentile(beta_samples[:, 1], [2.5, 97.5])

# Accuracy
linear_pred = mean_intercept + np.dot(X, np.array([mean_beta_polkadot, mean_beta_uniswap, mean_beta_volume]))
predicted_probs = sigmoid(linear_pred)
predicted_classes = (predicted_probs > 0.5).astype(int)
accuracy = (predicted_classes == y).mean()

# Posterior predictive
new_polkadot = 0.05
new_uniswap = -0.02
new_volume = 0.3

new_polkadot_std = (new_polkadot - polkadot_mean) / polkadot_std
new_uniswap_std = (new_uniswap - uniswap_mean) / uniswap_std
new_volume_std = (new_volume - volume_mean) / volume_std

new_obs = np.array([new_polkadot_std, new_uniswap_std, new_volume_std])

posterior_pred_probs = []
for i in range(len(beta_samples)):
    linear_pred_new = intercept_samples[i] + np.dot(new_obs, beta_samples[i])
    prob = sigmoid(linear_pred_new)
    posterior_pred_probs.append(prob)

posterior_pred_prob = np.mean(posterior_pred_probs)

# Print required outputs
print("Mean Posterior Coefficient for Polkadot Returns:", f"{mean_beta_polkadot:.6f}")
print("Mean Posterior Coefficient for Uniswap Returns:", f"{mean_beta_uniswap:.6f}")
print("Mean Posterior Coefficient for Aave Volume:", f"{mean_beta_volume:.6f}")
print()
print("95% Credible Interval for Polkadot Coefficient:", f"[{ci_polkadot[0]:.4f}, {ci_polkadot[1]:.4f}]")
print("95% Credible Interval for Uniswap Coefficient:", f"[{ci_uniswap[0]:.4f}, {ci_uniswap[1]:.4f}]")
print()
print("Classification Accuracy:", f"{accuracy:.4f}")
print()
print("Posterior Predictive Probability:", f"{posterior_pred_prob:.5f}")

# Generate scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(df_merged['Polkadot_Returns_Std'], df_merged['Uniswap_Returns_Std'],
            alpha=0.6, c='blue', edgecolors='black', linewidth=0.5)
plt.xlabel('Polkadot Standardized Returns', fontsize=12)
plt.ylabel('Uniswap Standardized Returns', fontsize=12)
plt.title('Cryptocurrency Standardized Returns: Polkadot vs Uniswap', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('scatter_plot.png', dpi=300, bbox_inches='tight')
plt.show()
