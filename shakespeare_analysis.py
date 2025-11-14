import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.linear_model import LogisticRegression
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# STEP 1: Load and Preprocess Data
# ============================================================================

print("=" * 80)
print("STEP 1: Loading and Preprocessing Data")
print("=" * 80)

def tokenize_dialogue(text):
    """
    Tokenize dialogue by replacing specific punctuation with spaces,
    converting to lowercase, and splitting on whitespace.
    """
    if pd.isna(text):
        return []

    # Define punctuation to replace with spaces
    punctuation = ['.', ',', ';', ':', '!', '?', "'", '"', '(', ')', '[', ']',
                   '{', '}', '/', '-', '—', '…', ''', ''', '"', '"']

    text_clean = str(text)
    for punct in punctuation:
        text_clean = text_clean.replace(punct, ' ')

    # Convert to lowercase and split on whitespace
    tokens = text_clean.lower().split()

    return tokens

def preprocess_dataset(filepath, play_name):
    """
    Load and preprocess a single Shakespeare play dataset.
    """
    # Load data
    df = pd.read_csv(filepath)

    # Exclude stage directions
    df = df[df['character'] != '[stage direction]'].copy()

    # Tokenize dialogue and count words
    df['tokens'] = df['dialogue'].apply(tokenize_dialogue)
    df['word_count'] = df['tokens'].apply(len)

    # Aggregate by character
    char_stats = df.groupby('character').agg({
        'word_count': ['sum', 'count', 'std'],
        'dialogue': 'count'
    }).reset_index()

    char_stats.columns = ['character', 'total_word_count', 'word_count_count',
                          'word_count_std', 'dialogue_count']

    # Filter characters with total_word_count > 300
    char_stats = char_stats[char_stats['total_word_count'] > 300].copy()

    # Add play source
    char_stats['play_source'] = play_name

    # Merge back to get full dataset for retained characters
    retained_chars = char_stats['character'].unique()
    df_filtered = df[df['character'].isin(retained_chars)].copy()
    df_filtered['play_source'] = play_name

    return df_filtered, char_stats

# Load all three datasets
hamlet_df, hamlet_chars = preprocess_dataset('hamlet.csv', 'hamlet')
romeo_df, romeo_chars = preprocess_dataset('romeo_juliet.csv', 'romeo_juliet')
macbeth_df, macbeth_chars = preprocess_dataset('macbeth.csv', 'macbeth')

print(f"Hamlet: {len(hamlet_chars)} characters with >300 words")
print(f"Romeo & Juliet: {len(romeo_chars)} characters with >300 words")
print(f"Macbeth: {len(macbeth_chars)} characters with >300 words")
print()

# ============================================================================
# STEP 2: Causal Inference - Propensity Score Matching
# ============================================================================

print("=" * 80)
print("STEP 2: Causal Inference Analysis")
print("=" * 80)

# Prepare data for causal inference (hamlet vs romeo_juliet only)
hamlet_causal = hamlet_chars[['character', 'total_word_count', 'dialogue_count']].copy()
hamlet_causal['treatment'] = 1

romeo_causal = romeo_chars[['character', 'total_word_count', 'dialogue_count']].copy()
romeo_causal['treatment'] = 0

causal_df = pd.concat([hamlet_causal, romeo_causal], ignore_index=True)

# Calculate outcome variable: average words per dialogue line
causal_df['avg_words_per_line'] = causal_df['total_word_count'] / causal_df['dialogue_count']

# Covariate: total dialogue instances
X = causal_df[['dialogue_count']].values
y = causal_df['treatment'].values

# Fit logistic regression for propensity scores
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X, y)

# Get propensity scores
causal_df['propensity_score'] = lr_model.predict_proba(X)[:, 1]
causal_df['logit_ps'] = np.log(causal_df['propensity_score'] / (1 - causal_df['propensity_score']))

# Calculate caliper
caliper = 0.15 * np.std(causal_df['logit_ps'])

# Perform one-to-one nearest neighbor matching without replacement
treated = causal_df[causal_df['treatment'] == 1].copy().reset_index(drop=True)
control = causal_df[causal_df['treatment'] == 0].copy().reset_index(drop=True)

matched_pairs = []
used_controls = set()

for i, treated_unit in treated.iterrows():
    # Calculate distances to all unused control units
    available_controls = control[~control.index.isin(used_controls)]

    if len(available_controls) == 0:
        break

    distances = np.abs(available_controls['logit_ps'].values - treated_unit['logit_ps'])
    min_distance = np.min(distances)

    # Check if within caliper
    if min_distance <= caliper:
        # Find the first control unit with minimum distance (row order)
        min_idx = np.where(distances == min_distance)[0][0]
        control_idx = available_controls.iloc[min_idx].name

        matched_pairs.append({
            'treated_idx': i,
            'control_idx': control_idx,
            'treated_outcome': treated_unit['avg_words_per_line'],
            'control_outcome': control.loc[control_idx, 'avg_words_per_line']
        })

        used_controls.add(control_idx)

# Calculate ATT
matched_df = pd.DataFrame(matched_pairs)
att = (matched_df['treated_outcome'] - matched_df['control_outcome']).mean()
num_matched_pairs = len(matched_pairs)

print(f"Average Treatment Effect on the Treated (ATT): {att:.3f}")
print(f"Number of Successfully Matched Pairs: {num_matched_pairs}")
print()

# ============================================================================
# STEP 3: Time Series Forecasting - ARIMAX for Hamlet Character
# ============================================================================

print("=" * 80)
print("STEP 3: Time Series Forecasting for Hamlet Character")
print("=" * 80)

# Select only Hamlet character from hamlet.csv
hamlet_ts = hamlet_df[hamlet_df['character'] == 'Hamlet'].copy()

# Convert line_number to numeric and sort
hamlet_ts['line_number'] = pd.to_numeric(hamlet_ts['line_number'], errors='coerce')
hamlet_ts = hamlet_ts.dropna(subset=['line_number'])
hamlet_ts = hamlet_ts.sort_values('line_number').reset_index(drop=True)

# Create scene_numeric variable
def extract_scene_numeric(scene):
    """Extract numeric value from Roman numeral scene notation."""
    scene = str(scene).strip()

    if 'Prologue' in scene or 'prologue' in scene:
        return 0

    # Map Roman numerals to integers
    roman_map = {
        'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5,
        'VI': 6, 'VII': 7, 'VIII': 8, 'IX': 9, 'X': 10
    }

    # Extract Roman numeral pattern
    match = re.search(r'\b([IVX]+)\b', scene)
    if match:
        roman = match.group(1)
        return roman_map.get(roman, 0)

    return 0

hamlet_ts['scene_numeric'] = hamlet_ts['scene'].apply(extract_scene_numeric)

# Forward-fill missing values in word_count
hamlet_ts['word_count'] = hamlet_ts['word_count'].fillna(method='ffill')

# Create train/validation split (80/20)
total_obs = len(hamlet_ts)
train_size = int(np.floor(0.8 * total_obs))

train_data = hamlet_ts.iloc[:train_size].copy()
val_data = hamlet_ts.iloc[train_size:].copy()

print(f"Total observations: {total_obs}")
print(f"Training set size: {train_size}")
print(f"Validation set size: {len(val_data)}")

# Fit ARIMAX(1,1,1) model
model = SARIMAX(
    train_data['word_count'],
    exog=train_data[['scene_numeric']],
    order=(1, 1, 1),
    enforce_stationarity=False,
    enforce_invertibility=False
)

fitted_model = model.fit(disp=False)

# Forecast on validation set
val_exog = val_data[['scene_numeric']]
forecast = fitted_model.forecast(steps=len(val_data), exog=val_exog)

# Calculate MAPE (excluding zero values)
val_data_nonzero = val_data[val_data['word_count'] != 0].copy()
forecast_nonzero = forecast[val_data['word_count'] != 0]

mape = np.mean(np.abs((val_data_nonzero['word_count'].values - forecast_nonzero.values) /
                      val_data_nonzero['word_count'].values)) * 100

print(f"Mean Absolute Percentage Error (MAPE) on validation set: {mape:.2f}%")
print()

# ============================================================================
# STEP 4: Hierarchical Mixed-Effects Model
# ============================================================================

print("=" * 80)
print("STEP 4: Hierarchical Mixed-Effects Model")
print("=" * 80)

# Combine all three datasets
combined_df = pd.concat([hamlet_df, romeo_df, macbeth_df], ignore_index=True)

# Create act_scene grouping variable
combined_df['act_scene'] = combined_df['act'].astype(str) + '_' + combined_df['scene'].astype(str)

# Ensure play_source is categorical with hamlet as reference
combined_df['play_source'] = pd.Categorical(
    combined_df['play_source'],
    categories=['hamlet', 'macbeth', 'romeo_juliet']
)

# Fit mixed-effects model using REML
model_formula = 'word_count ~ C(play_source, Treatment(reference="hamlet"))'
mixed_model = smf.mixedlm(
    model_formula,
    data=combined_df,
    groups=combined_df['act_scene'],
    re_formula='1'
)

mixed_results = mixed_model.fit(reml=True)

# Extract fixed effects coefficients
fixed_effects = mixed_results.params
coef_macbeth = fixed_effects['C(play_source, Treatment(reference="hamlet"))[T.macbeth]']
coef_romeo = fixed_effects['C(play_source, Treatment(reference="hamlet"))[T.romeo_juliet]']

# Calculate ICC
# ICC = variance_random / (variance_random + variance_residual)
variance_random = float(mixed_results.cov_re.iloc[0, 0])
variance_residual = mixed_results.scale
icc = variance_random / (variance_random + variance_residual)

print(f"Fixed effect coefficient for macbeth: {coef_macbeth:.4f}")
print(f"Fixed effect coefficient for romeo_juliet: {coef_romeo:.4f}")
print(f"Intraclass Correlation Coefficient (ICC): {icc:.4f}")
print()

# ============================================================================
# STEP 5: Visualization - Scatter Plot
# ============================================================================

print("=" * 80)
print("STEP 5: Generating Scatter Plot")
print("=" * 80)

# Combine character statistics
all_chars = pd.concat([hamlet_chars, romeo_chars, macbeth_chars], ignore_index=True)

# Create scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(all_chars['total_word_count'], all_chars['word_count_std'], alpha=0.6)
plt.xlabel('Total Word Count')
plt.ylabel('Standard Deviation of Word Count')
plt.title('Character Analysis: Total Word Count vs. Standard Deviation')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('shakespeare_scatter_plot.png', dpi=300)
print("Scatter plot saved as 'shakespeare_scatter_plot.png'")
print()

# ============================================================================
# Summary Report
# ============================================================================

print("=" * 80)
print("SUMMARY REPORT")
print("=" * 80)
print("\nCausal Inference Results:")
print(f"  Average Treatment Effect on the Treated: {att:.3f}")
print(f"  Number of Matched Pairs: {num_matched_pairs}")

print("\nTime Series Forecasting Results:")
print(f"  MAPE on Validation Set: {mape:.2f}")

print("\nHierarchical Model Results:")
print(f"  Fixed Effect (Macbeth): {coef_macbeth:.4f}")
print(f"  Fixed Effect (Romeo & Juliet): {coef_romeo:.4f}")
print(f"  Intraclass Correlation Coefficient: {icc:.4f}")

print("\n" + "=" * 80)
print("Analysis Complete!")
print("=" * 80)
