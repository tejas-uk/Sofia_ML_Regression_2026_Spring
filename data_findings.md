# Data Findings — Sofia ML Regression 2026

Complete EDA results from `eda.ipynb`. All numbers are computed on the raw CSVs.

---

## Dataset Overview

| Property | Train | Test |
|---|---|---|
| Rows | 2,500 | 2,500 |
| Features | 15 (`x0`–`x14`) | 15 (`x0`–`x14`) |
| Target column | `target` (continuous) | — |
| Id column | `Id` | `Id` |

Features carry no domain meaning. All 15 are continuous numerical variables.

---

## 1. Target Distribution

The target is **not normally distributed** — it is one of the most extreme cases of heavy-tailed data a regression model can encounter.

| Statistic | Value |
|---|---|
| Mean | −22.04 |
| Median | −0.77 |
| Std deviation | 1,978.18 |
| Minimum | −41,008.02 |
| Maximum | +69,628.20 |
| 1st percentile | −2,654.94 |
| 25th percentile | −43.17 |
| 75th percentile | +42.00 |
| 99th percentile | +1,591.06 |
| **Skewness** | **+13.59** |
| **Kurtosis (excess)** | **707.81** |

### What these numbers mean

- The **mean (−22) and median (−0.77) are nearly 2,000 standard deviations apart** when expressed relative to the bulk of the data. This signals that a tiny number of extreme values are pulling the mean far from where 98% of the data actually sits.
- A **skewness of 13.59** is extreme. Values above 1.0 are typically considered "highly skewed." This means the right tail (large positive targets) is far longer than the left.
- A **kurtosis of 707.81** (excess kurtosis; normal = 0) indicates tails that are vastly heavier than any normal distribution. The Q-Q plot confirms sharp deviation at both ends.
- The **interquartile range is only ±43** (p25 to p75), but the full range spans 110,636 units. The core 50% of data is compressed into a very narrow band.

### Outlier counts

| Threshold | # Rows | % of Train |
|---|---|---|
| Outside 1st–99th percentile | 50 | 2.00% |
| \|target\| > 1,000 | 84 | 3.36% |
| \|target\| > 5,000 | 24 | 0.96% |

**Modelling implication:** Standard OLS linear regression minimises squared error, which makes it highly sensitive to these extreme values — a handful of rows with |target| > 10,000 will dominate the loss. Consider Huber regression, log-transforming the target (if all positive), or training two models (one for bulk, one for extremes).

---

## 2. Missing Values

Every single feature has missing values in both train and test. The rates are low and uniform.

| Feature | Train missing | Train % | Test missing | Test % |
|---|---|---|---|---|
| x0 | 32 | 1.3% | 41 | 1.6% |
| x1 | 30 | 1.2% | 47 | 1.9% |
| x2 | 22 | 0.9% | 40 | 1.6% |
| x3 | 24 | 1.0% | 37 | 1.5% |
| x4 | 34 | 1.4% | 30 | 1.2% |
| x5 | 28 | 1.1% | 37 | 1.5% |
| x6 | 30 | 1.2% | 36 | 1.4% |
| x7 | 30 | 1.2% | 38 | 1.5% |
| x8 | 34 | 1.4% | 27 | 1.1% |
| x9 | 30 | 1.2% | 41 | 1.6% |
| x10 | 31 | 1.2% | 28 | 1.1% |
| x11 | 32 | 1.3% | 30 | 1.2% |
| x12 | 38 | 1.5% | 29 | 1.2% |
| x13 | 28 | 1.1% | 34 | 1.4% |
| x14 | 25 | 1.0% | 37 | 1.5% |

### Row-level missingness

| # Features missing per row | # Rows | % of Train |
|---|---|---|
| 0 | 2,268 | 90.7% |
| ≥ 1 | 232 | 9.3% |
| ≥ 2 | 144 | 5.8% |
| ≥ 3 | 72 | 2.9% |

### Co-occurrence of missingness

The maximum number of rows where any two features are simultaneously missing is **7 rows** (x12 & x4, and x2 & x7). Given that each feature is missing ~30 rows, an expected co-occurrence under independence would be roughly 30 × 30 / 2500 ≈ 0.36 rows. The observed co-occurrence of 7 is higher than chance but still very small.

**Conclusion:** Missingness is approximately **MCAR (Missing Completely At Random)**. The pattern is not strongly structured. Median imputation is a safe and sufficient strategy — there is no evidence that missing-ness itself is predictive.

---

## 3. Feature–Target Correlations (Linear)

Pearson r between each raw feature and the target, ranked by absolute value:

| Feature | r | p-value | Significant? |
|---|---|---|---|
| **x9** | **+0.232** | **1.26 × 10⁻³¹** | **Yes** |
| x8 | −0.048 | 0.018 | Yes (barely) |
| x4 | +0.041 | 0.043 | Yes (barely) |
| x6 | −0.040 | 0.044 | Yes (barely) |
| x11 | +0.027 | 0.172 | No |
| x5 | −0.026 | 0.192 | No |
| x10 | −0.023 | 0.246 | No |
| x2 | −0.023 | 0.258 | No |
| x7 | +0.020 | 0.311 | No |
| x12 | +0.015 | 0.461 | No |
| x1 | +0.005 | 0.806 | No |
| x0 | +0.003 | 0.868 | No |
| x14 | −0.003 | 0.876 | No |
| x13 | +0.003 | 0.886 | No |
| x3 | −0.002 | 0.921 | No |

### Key observations

- **x9 completely dominates** linear signal. Its r = 0.232 is roughly 5× larger than the next-closest feature (x8 at −0.048) and is statistically unambiguous (p = 10⁻³¹).
- **x8, x4, and x6 are marginally significant** (p < 0.05) but with r < 0.05, they explain less than 0.25% of target variance individually.
- **11 out of 15 features show no statistically significant linear relationship** with the target.
- Even x9 only explains **5.4% of target variance** (r² = 0.054). The problem is genuinely hard — a linear model using all raw features will have low R².

---

## 4. Feature–Feature Correlations

| Pair | Pearson r |
|---|---|
| x1 × x12 | 0.063 |
| x11 × x14 | 0.047 |
| x4 × x6 | 0.038 |
| x7 × x13 | 0.037 |
| x1 × x3 | 0.036 |
| x7 × x8 | 0.035 |
| x6 × x11 | 0.035 |
| x1 × x4 | 0.033 |
| x0 × x11 | 0.032 |
| x2 × x7 | 0.032 |

**The 15 features are essentially orthogonal to each other.** The maximum pairwise correlation is 0.063, well below any meaningful threshold (typically r > 0.3 is considered "moderate"). This has two consequences:

1. Dimension reduction (PCA, etc.) will not compress the feature space effectively — each feature carries independent information.
2. There is no multicollinearity, so coefficient estimates from linear models will be stable and interpretable.

---

## 5. Train vs Test Distribution (KS Test)

Kolmogorov-Smirnov test comparing each feature's empirical distribution in train vs test:

| Feature | KS Statistic | p-value |
|---|---|---|
| x0 | 0.0225 | 0.548 |
| x1 | 0.0321 | 0.154 |
| x2 | 0.0162 | 0.896 |
| x3 | 0.0241 | 0.458 |
| x4 | 0.0206 | 0.663 |
| x5 | 0.0363 | 0.075 |
| x6 | 0.0155 | 0.921 |
| x7 | 0.0224 | 0.553 |
| x8 | 0.0234 | 0.496 |
| x9 | 0.0352 | 0.092 |
| x10 | 0.0152 | 0.933 |
| x11 | 0.0200 | 0.694 |
| x12 | 0.0346 | 0.102 |
| x13 | 0.0286 | 0.257 |
| x14 | 0.0250 | 0.414 |

**No feature shows a statistically significant distribution shift** (all p > 0.05). All KS statistics are below 0.037. Train and test appear to be drawn from the same underlying distribution. A model trained on the full training set should generalise to the test set without covariate shift problems.

---

## 6. Pairwise Interaction Effects

Since raw feature–target correlations are weak, we computed the Pearson r between every `xi × xj` product and the target to detect multiplicative interaction effects:

| Interaction | r with target |
|---|---|
| **x9 × x11** | **+0.160** |
| **x9 × x12** | **+0.105** |
| x2 × x9 | −0.105 |
| x0 × x9 | −0.092 |
| x8 × x9 | −0.090 |
| x5 × x9 | +0.089 |
| x1 × x9 | +0.085 |
| x7 × x11 | +0.066 |
| x1 × x11 | +0.054 |
| x1 × x7 | +0.051 |
| x9 × x10 | −0.049 |
| x3 × x9 | −0.048 |
| x11 × x12 | +0.047 |
| x1 × x2 | −0.047 |
| x3 × x7 | −0.042 |

### Key observations

- **x9 appears in 9 of the top 15 interactions.** It is not merely a linear predictor — it amplifies or dampens the effect of nearly every other feature.
- The strongest interaction is **x9 × x11 (r = 0.160)**, which is actually stronger than the raw correlation of x9 alone with the target after controlling for the massive outlier inflation. This is the most important feature engineering candidate.
- **x11 appears in three of the top 10 interactions** (with x9, x7, x1) despite having no significant raw correlation with the target (r = 0.027). It is a "latent amplifier" — its signal only appears when combined with other features.
- Non-x9 interactions peak at ~0.066 (x7 × x11), indicating that the non-x9 signal is genuinely weak.

**Modelling implication:** Adding polynomial/interaction features — at minimum `x9 × x11`, `x9 × x12`, and `x9 × (all others)` — should improve R² substantially over raw features. Tree-based models (Random Forest, Gradient Boosting) will capture these automatically.

---

## 7. PCA Structure

PCA was run on median-imputed, standardised features.

| Component | Variance explained | Cumulative |
|---|---|---|
| PC1 | 7.46% | 7.5% |
| PC2 | 7.43% | 14.9% |
| PC3 | 7.13% | 22.0% |
| PC4 | 7.07% | 29.1% |
| PC5 | 6.94% | 36.0% |
| PC6 | 6.89% | 42.9% |
| PC7 | 6.70% | 49.6% |
| PC8 | 6.65% | 56.3% |
| PC9 | 6.51% | 62.8% |
| PC10 | 6.49% | 69.3% |
| PC11 | 6.37% | 75.6% |
| PC12 | 6.21% | 81.8% |
| PC13 | 6.18% | 88.0% |
| PC14 | 6.05% | 94.1% |
| PC15 | 5.93% | 100.0% |

The explained variance is **almost perfectly flat across all 15 components** (ranging only from 5.93% to 7.46%). A random set of orthogonal features would each explain 100/15 = 6.67%. The actual values are indistinguishable from this baseline. This confirms the feature–feature correlation finding: **the features are genuinely independent and carry equal amounts of information**. There is no dominant axis to project onto.

Examining PC1 vs PC2 coloured by log |target| and by the extreme-target flag (|y| > 1,000) showed **no visible clustering** of extreme rows. Outlier rows are scattered uniformly across the PCA 2D space — they are not a geometrically separable subgroup.

---

## 8. Outlier Anatomy — Feature Profiles of Extreme Rows

For the 84 rows (3.36%) where |target| > 1,000, we compared their standardised feature means to the rest of the dataset. Values are in units of standard deviations:

| Feature | Δ (extreme − normal) | Direction |
|---|---|---|
| **x9** | **−0.220** | Extreme rows have *lower* x9 |
| **x12** | **+0.169** | Extreme rows have *higher* x12 |
| x3 | −0.110 | Extreme rows have lower x3 |
| x0 | −0.110 | Extreme rows have lower x0 |
| x1 | +0.090 | Extreme rows have higher x1 |
| x2 | −0.087 | Extreme rows have lower x2 |
| x11 | +0.086 | Extreme rows have higher x11 |
| x7 | −0.066 | Extreme rows have lower x7 |
| x4 | +0.063 | Extreme rows have higher x4 |
| x8 | −0.062 | Extreme rows have lower x8 |
| x13 | −0.045 | Moderate |
| x5 | −0.020 | Negligible |
| x14 | +0.012 | Negligible |
| x10 | +0.012 | Negligible |
| x6 | −0.006 | Negligible |

Despite x9 being the strongest linear predictor, extreme-target rows actually have **lower x9 values** on average (−0.22σ), not higher. This initially seems contradictory but makes sense given the heavy positive skew — the biggest positive targets are driven by interaction effects (x9 × x11, etc.) and non-linearities, not purely by large x9 values. x12 showing the opposite direction (+0.17σ) reinforces this — the extreme regime is governed by a different combination of features than the bulk.

No single feature shows a large enough deviation (> 0.5σ) to cleanly separate extreme rows from normal ones. The outlier-generating mechanism is distributed across multiple features.

---

## 9. x9 Deep Dive

x9 is the only feature worth examining in isolation. Its relationship with the target is **monotone but non-linear**.

### Conditional median by x9 quartile

| x9 Quartile | x9 range | Median target |
|---|---|---|
| Q1 (bottom 25%) | ≤ −4.92 | −17.34 |
| Q2 | −4.92 to −0.25 | −4.16 |
| Q3 | −0.25 to +4.56 | +7.78 |
| Q4 (top 25%) | > +4.56 | +9.72 |

The median target rises from −17.3 (low x9) to +9.7 (high x9), confirming the positive linear relationship. However, the step from Q3 to Q4 is small (+7.78 → +9.72) compared to the earlier jumps, indicating **diminishing returns at high x9 values** — the relationship is sub-linear in the upper range.

A LOWESS smoother on the x9–target scatter (with target clipped to the 1–99th percentile) confirms this: the curve is roughly linear through the middle of x9's range but flattens in the upper tail.

**Modelling implication:** A raw x9 term will partially capture this, but a log or square root transformation of x9 (or including x9² as a polynomial feature) may better represent the diminishing-returns shape.

---

## 10. Summary of Actionable Findings

| # | Finding | Action |
|---|---|---|
| 1 | Target is extremely heavy-tailed (skew=13.6, kurt=708) | Use Huber loss, or evaluate on bulk vs. extreme subsets separately |
| 2 | 9.3% of rows have ≥1 missing feature; pattern is random | Median imputation is sufficient; do not invest in complex imputation |
| 3 | x9 is the only strong individual predictor (r=0.232) | Always include x9; consider x9 transformations (log, sqrt, polynomial) |
| 4 | 11/15 features are individually non-significant | Do not expect a strong linear model; need interactions or non-linear methods |
| 5 | Features are nearly orthogonal (max pairwise r=0.063) | PCA/dimensionality reduction will not help; keep all features |
| 6 | x9 × x11 is the strongest interaction (r=0.160) | Explicitly engineer x9 × x11 and other x9 products as features |
| 7 | x11 is a hidden amplifier (appears in 3 top interactions) | Include x11 interactions even though its raw correlation is weak |
| 8 | No distribution shift between train and test (all KS p>0.05) | Models trained on full train should generalise; no domain adaptation needed |
| 9 | Extreme rows not separable in PCA space | A single unified model is preferable to a two-stage outlier/bulk split |
| 10 | x9–target relationship has diminishing returns above Q3 | Test polynomial (x9²) or spline features for x9 |
