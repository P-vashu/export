# Hierarchical Bayesian Logistic Regression Analysis Results

## Executive Summary

This analysis performed hierarchical Bayesian logistic regression on three Amazon review datasets (ebooks, grocery, and books) to examine the influence of `helpful_votes` on the probability of `star_rating` being 1 across distinct product categories.

---

## Data Preparation

### Datasets Loaded
- **amazon_ebook_Data.csv**: 100 ebook reviews (category_code = 0)
- **amazon_grocery_Data.csv**: 100 grocery product reviews (category_code = 1)
- **amazon_books_Data.csv**: 100 book reviews (category_code = 2)

### Preprocessing Steps
1. Standardized sentiment columns to unified `sentiment_label` across all datasets
2. Created categorical variable `category_code` (0, 1, 2)
3. Concatenated datasets vertically (total: 300 records)
4. Retained columns: `star_rating`, `helpful_votes`, `total_votes`, `category_code`
5. Created binary outcome: `binary_rating` (1 if star_rating=1, 0 otherwise)

### Binary Outcome Distribution
- **star_rating = 1**: 249 records (83%)
- **star_rating ≠ 1**: 51 records (17%)

---

## Model Specification

### Hierarchical Bayesian Logistic Regression
- **Likelihood**: Bernoulli with logit link function
- **Dependent Variable**: Binary star_rating (1 vs not-1)
- **Independent Variable**: helpful_votes
- **Hierarchical Grouping**: category_code (partial pooling)

### Prior Distributions
- **Population-level intercept mean**: Normal(μ=0, σ=10)
- **Population-level slope mean**: Normal(μ=0, σ=5)
- **Between-group std (intercepts)**: HalfCauchy(β=5)
- **Between-group std (slopes)**: HalfCauchy(β=5)

### MCMC Configuration
- **Iterations**: 5000 draws
- **Burn-in**: 1000 samples
- **Chains**: 4
- **Random Seed**: 42
- **Target Accept**: 0.99

---

## Convergence Diagnostics

✓ **CONVERGENCE ACHIEVED**
- Maximum R-hat: **1.000** (< 1.01 threshold)
- All parameters converged successfully
- Effective sample sizes: 2,904 to 21,682 (all adequate)

---

## Required Posterior Statistics

| Statistic | Value | Precision |
|-----------|-------|-----------|
| **1. Group-level intercept for category_code 0** | **1.7480** | 4 decimals |
| **2. Group-level slope for category_code 1** | **-0.0398** | 4 decimals |
| **3. Between-group std deviation (intercepts)** | **0.61378** | 5 decimals |
| **4. Population-level intercept mean** | **1.7703** | 4 decimals |
| **5. Maximum R-hat value** | **1.000** | 3 decimals |
| **6. Effective sample size (slope, category_code 2)** | **8216** | integer |

---

## Interpretation

### Group-Level Intercepts (Log-Odds)
- **Ebooks (category 0)**: 1.7480
- **Grocery (category 1)**: 1.6990
- **Books (category 2)**: 1.8840

All categories show positive baseline log-odds of receiving a 1-star rating, with books having the highest baseline tendency.

### Group-Level Slopes (Effect of helpful_votes)
- **Ebooks (category 0)**: -0.1000
- **Grocery (category 1)**: -0.0398
- **Books (category 2)**: -0.4040

Negative slopes across all categories indicate that higher `helpful_votes` is associated with **lower probability** of a 1-star rating. The effect is strongest for books (category 2) and weakest for grocery items (category 1).

### Between-Group Variation
- **Intercept variation**: 0.614 (moderate heterogeneity in baseline tendencies)
- **Slope variation**: 0.650 (moderate heterogeneity in helpful_votes effects)

The hierarchical structure effectively captures variation across product categories through partial pooling.

---

## Visualization

**Hexbin Plot**: `hexbin_helpful_vs_total_votes.png`
- X-axis: helpful_votes
- Y-axis: total_votes
- Grid size: 25
- Color scale: Logarithmic

The hexbin plot reveals the joint distribution of helpful and total votes across all 300 reviews, with logarithmic scaling to highlight density patterns.

---

## Files Generated

1. `hierarchical_bayesian_analysis.py` - Complete analysis script
2. `hexbin_helpful_vs_total_votes.png` - Hexbin visualization
3. `RESULTS_SUMMARY.md` - This summary document

---

## Conclusion

The hierarchical Bayesian logistic regression successfully modeled the relationship between `helpful_votes` and 1-star ratings across three Amazon product categories. The analysis achieved excellent convergence (R-hat < 1.01) and revealed that:

1. Higher helpful votes are associated with **lower** probability of 1-star ratings across all categories
2. The effect varies by category, with books showing the strongest negative association
3. Partial pooling effectively captured both category-specific patterns and shared population-level trends

**Analysis Date**: 2025-11-15
**Random Seed**: 42
**Convergence Status**: ✓ Achieved
