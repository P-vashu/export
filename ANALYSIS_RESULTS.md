# Cryptocurrency Volatility Analysis Results

## Executive Summary
This report presents advanced statistical analysis of cryptocurrency market dynamics using Bitcoin, BinanceCoin, and Aave datasets from October 5, 2020, to July 6, 2021.

---

## 1. Data Preprocessing

### Dataset Information
- **Analysis Period:** October 5, 2020 - July 6, 2021
- **Bitcoin Records:** 273 (after removing first row with undefined Daily_Return)
- **BinanceCoin Records:** 273 (after removing first row with undefined Daily_Return)
- **Aave Records:** 273 (after removing first row with undefined Daily_Return)

### Daily Return Calculation
Daily returns were calculated as:
```
Daily_Return = (Close_t - Close_{t-1}) / Close_{t-1} × 100
```

---

## 2. Seasonal Decomposition (Bitcoin)

**Method:** Multiplicative model with period 30, classical decomposition with extrapolate_trend='freq'

### Results:
- **Trend Component Mean:** `36378.1594`
- **Residual Component Standard Deviation:** `0.0641`

**Interpretation:** The Bitcoin price exhibited an upward trend with a mean of approximately $36,378 during the analysis period. The low residual standard deviation (0.0641) indicates that the multiplicative seasonal decomposition model effectively captured the temporal patterns in Bitcoin prices, with minimal unexplained variation.

---

## 3. Principal Component Analysis (PCA)

**Method:** Z-score normalization applied to High, Low, Open, Close, and Volume columns of concatenated dataset (n=819)

### Results:
- **First Principal Component (PC1) Explained Variance Ratio:** `0.9535`
- **Cumulative Explained Variance (PC1-PC3):** `0.9998`

**Interpretation:** The first principal component captures 95.35% of the total variance across all normalized features, indicating strong correlation among price variables (High, Low, Open, Close) and Volume. The first three components explain virtually all variance (99.98%), suggesting high dimensional redundancy in the cryptocurrency price data.

---

## 4. Granger Causality Test

**Method:** SSR F-test with lag 5, testing if BinanceCoin Daily_Return Granger-causes Bitcoin Daily_Return

**Data:** Inner join on Date between Bitcoin and BinanceCoin (n=273)

### Results:
- **F-statistic:** `1.366`
- **P-value:** `0.237350`

**Interpretation:** At conventional significance levels (α = 0.05), we fail to reject the null hypothesis that BinanceCoin daily returns do not Granger-cause Bitcoin daily returns. The p-value of 0.237 suggests that past values of BinanceCoin returns do not provide statistically significant predictive power for Bitcoin returns beyond what is contained in Bitcoin's own past returns.

---

## 5. Generalized Pareto Distribution (Bitcoin Volume)

**Method:** Maximum Likelihood Estimation with location parameter fixed at 0, analyzing exceedances above 95th percentile

**Threshold (95th percentile):** 85,524,751,547.05

**Number of Exceedances:** 14

### Results:
- **Shape Parameter (ξ):** `0.6827`
- **Scale Parameter (σ):** `12232871190.0979`

**Interpretation:** The positive shape parameter (ξ = 0.6827) indicates a heavy-tailed distribution, characteristic of extreme value distributions with no finite upper bound. This suggests that Bitcoin trading volume exhibits extreme volatility with potential for very large outliers. The heavy tail implies substantial tail risk and the possibility of extreme volume spikes beyond the 95th percentile.

---

## 6. Quantile Regression (BinanceCoin)

**Method:** Quantile regression with Volume (independent variable) and Close (dependent variable) at quantiles 0.25, 0.50, and 0.75

### Results:

| Quantile | Slope Coefficient | Scientific Notation |
|----------|-------------------|---------------------|
| 0.25     | 0.00000           | 6.3688 × 10⁻⁸       |
| 0.50     | 0.00000           | 6.9001 × 10⁻⁸       |
| 0.75     | 0.00000           | 7.5745 × 10⁻⁸       |

**Slope Coefficient at 0.50 Quantile:** `0.00000` (precisely: 6.9001 × 10⁻⁸)

**Interpretation:** The extremely small slope coefficients (on the order of 10⁻⁸) indicate an exceptionally weak linear relationship between trading volume and closing price across all quantiles. This suggests that trading volume alone is not a strong predictor of BinanceCoin's closing price. The slight increase in slope from lower to higher quantiles (0.25 → 0.75) suggests marginally stronger relationships at higher price levels.

**Note:** Large condition number (5.12 × 10⁹) indicates potential numerical issues due to vastly different scales between Volume (billions) and Close (hundreds).

---

## 7. Spectral Analysis (Aave)

**Method:** Periodogram analysis using FFT on mean-centered Close prices, analyzing positive frequencies up to Nyquist frequency (0.5 cycles/day)

**Data Points:** 273

### Results:
- **Frequency with Highest Power Spectral Density:** `0.003663` cycles/day
- **Periodicity:** `273.00` days

**Interpretation:** The dominant frequency component corresponds to the fundamental frequency of the data series (1/273 ≈ 0.003663), indicating that the strongest periodic pattern in Aave's closing prices spans the entire observation period. This suggests a long-term trend or cycle rather than short-term periodic fluctuations. The absence of stronger higher-frequency components indicates limited evidence of regular short-term cyclical patterns in Aave prices during this period.

---

## 8. Cross-Asset Return Correlation

**Visualization:** Hexbin plot of Bitcoin Daily_Return vs. BinanceCoin Daily_Return

**File:** `hexbin_bitcoin_binancecoin_returns.png`

**Data Points:** 273 (inner join on Date)

**Description:** The hexbin plot visualizes the joint distribution of daily returns between Bitcoin and BinanceCoin, revealing the strength and pattern of return co-movement between these two cryptocurrencies during the analysis period.

---

## Key Findings Summary

1. **Temporal Patterns:** Bitcoin prices show clear trend components with minimal seasonal noise during the study period.

2. **Dimensional Structure:** High correlation among price features (95.35% variance explained by PC1) indicates that cryptocurrency prices move together as a coherent market system.

3. **Causal Relationships:** No statistically significant Granger causality from BinanceCoin to Bitcoin at lag 5, suggesting independent price discovery processes.

4. **Extreme Value Behavior:** Bitcoin trading volume exhibits heavy-tailed distribution with substantial extreme value risk.

5. **Volume-Price Dynamics:** Weak relationship between trading volume and closing prices for BinanceCoin, suggesting that volume alone is insufficient for price prediction.

6. **Spectral Characteristics:** Aave prices dominated by long-period trends rather than short-term cyclical patterns.

---

## Methodology Notes

- All analyses performed in Python using statsmodels, scikit-learn, scipy, pandas, numpy, and matplotlib
- Date filtering strictly applied: October 5, 2020 - July 6, 2021
- Daily returns calculated with exact formula specification
- Z-score normalization: (X - μ) / σ
- All statistical tests performed at standard significance levels
- Numerical precision maintained throughout calculations

---

**Analysis Date:** November 10, 2025
**Analyst:** Quantitative Analysis Team
