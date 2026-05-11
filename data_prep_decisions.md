# Data Preparation — Decisions & Rationale

This document explains every transformation applied in `data_prep.ipynb`, what was done, and the specific EDA finding that justified each choice. Numbers cited here come from `data_findings.md`.

---

## Pipeline Overview

Raw CSVs → **Imputation** → **Target Preparation** → **Feature Engineering** → **Scaling** → `processed/`

The pipeline is fit **exclusively on training data** and then applied to test. This is a hard rule: any statistic derived from test data (medians, means, standard deviations) would constitute data leakage and produce optimistically biased cross-validation scores.

---

## Step 1 · Imputation

### What was done
Every missing value in every feature was replaced with the **median of that feature computed from the training set**. The same median values were then used to fill missing values in the test set.

Implemented with `sklearn.SimpleImputer(strategy="median")` fitted on train.

### Why median, not mean or model-based imputation

**Finding:** Every feature has 0.9–1.5% missing values in train and 1.1–1.9% in test. The co-occurrence heatmap showed that the maximum number of rows where any two features are simultaneously missing is **7 rows** — effectively indistinguishable from the ~0.36 expected under pure randomness (30 × 30 / 2500). This pattern is consistent with **MCAR (Missing Completely At Random)**.

Under MCAR, any imputation strategy produces unbiased estimates. The choice between them comes down to robustness:

- **Mean imputation** is sensitive to the extreme target outliers. Although we're imputing features (not the target), the feature distributions themselves contain values spread across wide ranges (x5 goes from −52.7 to +74.7). The median is resistant to those tails.
- **Model-based imputation** (e.g., KNN, iterative imputation) is computationally expensive and only pays off when features are correlated with each other. The EDA found that the maximum pairwise feature correlation is **0.063** — the features are essentially independent. A model predicting x1 from all other features would have near-zero predictive power, so it would converge to the mean/median anyway.

**Conclusion:** Median imputation is statistically justified and computationally appropriate for this dataset. Complex imputation would add cost with no measurable benefit.

### Why fit only on train
If the imputer were fit on the full dataset (train + test combined), the test set's feature distributions would contaminate the imputation medians. This is a subtle but real form of leakage: during cross-validation, models would implicitly "know" something about the test distribution. Fitting only on train eliminates this.

---

## Step 2 · Target Preparation

### What was done
Three target-related outputs were created:

1. **`y_train.csv`** — the original, unmodified target
2. **`y_train_winsorized.csv`** — target clipped to the 1st–99th percentile range [−2,654.94, +1,591.06]
3. **`is_extreme.csv`** — binary flag: 1 if |target| > 1,000, else 0

### Why two target versions

**Finding:** The target has skewness = +13.59 and excess kurtosis = 708.81. For reference, a normal distribution has both equal to 0. The interquartile range spans only ±43, but the full range spans 110,636 units. 84 rows (3.36%) have |target| > 1,000 and 24 rows (0.96%) have |target| > 5,000.

The consequence for training is severe. Ordinary Least Squares minimises **squared** residuals. A single row with target = 69,628 contributes a squared error of ~4.8 billion to the loss function. The model will dedicate a disproportionate share of its capacity trying to fit that one row, at the cost of accuracy on the 2,416 rows with |target| < 1,000. This is not a hypothetical concern — it is the primary reason plain linear regression fails on heavy-tailed data.

**Two strategies address this:**

- **Winsorized target** clips the extremes before training. The model is trained as if the largest target values were 1,591 and the most negative were −2,655. This makes OLS-family models (Ridge, Lasso, ElasticNet) tractable. The cost is that the model will systematically under-predict on true extreme rows — but those rows are only 2% of the data, and accurately predicting the other 98% is more valuable for R².
- **Raw target** preserves the full signal. Tree-based models (Random Forest, Gradient Boosting) are not sensitive to this problem because they split on feature thresholds — a row with target = 69,628 influences the split criterion but not in a squared way. Huber regression explicitly down-weights residuals beyond a threshold. These models should be trained on the raw target.

### Why 1st–99th percentile as the winsorization boundary

Clipping at 1st–99th is a standard choice that removes the 50 most extreme rows (25 on each tail) while preserving 98% of the training data unchanged. Narrower bounds (5th–95th) would clip too aggressively and distort the target distribution. Wider bounds (0.1st–99.9th) would still leave in rows with |target| > 10,000 that remain problematic for OLS.

### Why the `is_extreme` flag

The outlier anatomy analysis showed that extreme-target rows (|target| > 1,000) have measurable feature differences from normal rows — notably lower x9 (−0.22 standard deviations) and higher x12 (+0.17 standard deviations). While these deviations are not large enough to cleanly separate the groups, the flag enables two modelling strategies:

1. **Sample weights** — pass `sample_weight=1 + is_extreme * k` to penalise or de-penalise extreme rows during training.
2. **Diagnostic filtering** — compute R² separately on extreme and non-extreme rows to understand where a model is failing.

---

## Step 3 · Feature Engineering

### What was done

13 new features were added to the base 15, for a total of 28 features:

| New Feature | Formula | r with raw target |
|---|---|---|
| x9_sq | x9² | +0.049 |
| x9_x_x11 | x9 × x11 | +0.160 |
| x9_x_x12 | x9 × x12 | +0.105 |
| x9_x_x2 | x9 × x2 | −0.105 |
| x9_x_x0 | x9 × x0 | −0.092 |
| x9_x_x8 | x9 × x8 | −0.090 |
| x9_x_x5 | x9 × x5 | +0.089 |
| x9_x_x1 | x9 × x1 | +0.085 |
| x9_x_x10 | x9 × x10 | −0.049 |
| x9_x_x3 | x9 × x3 | −0.048 |
| x11_x_x7 | x11 × x7 | +0.066 |
| x11_x_x1 | x11 × x1 | +0.054 |
| x11_x_x12 | x11 × x12 | +0.047 |

### Why x9²

**Finding:** The conditional median analysis showed the median target rising from −17.3 (x9 in Q1) to +7.8 (x9 in Q3) but only reaching +9.7 in Q4 — a much smaller increment despite a much larger x9 range. A LOWESS smoother on the x9–target scatter confirmed this diminishing-returns shape.

A raw x9 term captures linear growth but not curvature. Including x9² allows a linear model to fit a parabolic relationship. After scaling, x9² and x9 are independent enough to be included together without causing instability.

Note: a log transform was considered but rejected because **x9 has negative values** (range: −23.9 to +29.0). Square root of the absolute value was tested and showed near-zero correlation (|x9| r = 0.001 with target), so it was excluded.

### Why x9 interactions (and which ones to include)

**Finding:** x9 is involved in 9 of the top 15 pairwise interaction correlations. The interaction matrix (xi × xj correlated with target) revealed that **the dominant signal in this dataset is not x9 itself but x9 acting as a multiplier on other features**.

The strongest interactions are:

- **x9 × x11** (r = 0.160): stronger than x9 alone. x11 had essentially no raw correlation with target (r = 0.027, p = 0.172), but when multiplied by x9 it becomes the single strongest engineered feature. This suggests x11 acts as a "direction" signal — the sign and magnitude of x11 determines whether high x9 leads to a very positive or very negative target.
- **x9 × x12** (r = 0.105) and **x9 × x2** (r = −0.105): similar magnitude, opposite sign.

**Threshold for inclusion:** Features with |r| > 0.04 with the target were included. This threshold was chosen because:
- Features below 0.04 (x9 × x14 at 0.019, x9 × x13 at 0.010, x9 × x4 at 0.007, x9 × x6 at 0.006, x9 × x7 at 0.001) are within the noise range of a dataset with 2,500 rows. At n=2,500, the 95% confidence interval for r=0 spans roughly ±0.04.
- Including all 14 possible x9 interactions would add noise features that could slightly degrade regularised models.

**Features excluded from x9 interactions:** x14, x13, x4, x6, x7 — all with |r| < 0.02 against target when multiplied by x9.

### Why x11 interactions (and which ones to include)

**Finding:** x11 appeared in 3 of the top 10 non-self interactions despite having no raw correlation with target (r = 0.027). This pattern — weak alone, strong in combination — identifies x11 as a **latent amplifier**: a feature whose value gates the effect of other features rather than contributing additively.

The three x11 interactions retained:
- **x11 × x7** (r = 0.066): the strongest non-x9 interaction in the entire matrix
- **x11 × x1** (r = 0.054): second strongest
- **x11 × x12** (r = 0.047): third strongest

All three clear the same |r| > 0.04 threshold used for x9. The remaining x11 interactions (with x0, x2, x5, x8, x10) ranged from 0.034 to 0.041 — marginal and excluded to keep feature count manageable.

### What was not engineered

- **Higher-order terms beyond x9²**: x9³ and beyond showed diminishing correlation gains and would risk overfitting with only 2,500 training rows.
- **All pairwise interactions (PolynomialFeatures degree=2)**: This would produce 15×14/2 = 105 new features. With n=2,500 samples and almost all raw pairwise correlations below 0.04, most of these would be noise. Regularisation (Lasso) could prune them, but the compute cost and risk of overfitting in cross-validation outweigh the benefit compared to selecting the interactions already known to be informative.
- **PCA features**: PCA explained variance is perfectly flat (5.93% to 7.46% per component). There is no dominant axis to extract. PCA would add complexity without capturing any structure.

---

## Step 4 · Scaling

### What was done
`sklearn.StandardScaler` was fitted on the training set (after imputation and feature engineering) and applied to both train and test. Each feature was transformed to zero mean and unit standard deviation.

### Why scaling is necessary

**Finding:** The 15 base features have widely varying standard deviations — x3 has std = 2.0 while x5 has std = 15.2. The engineered interaction features are products of two features, so their scale is roughly the product of the two parents' standard deviations. For example, x9_x_x11 involves x9 (std ≈ 7.0) × x11 (std ≈ 14.5), giving a product with std ≈ 100.

Unscaled features create two problems:
1. **Regularised linear models** (Ridge, Lasso, ElasticNet) penalise the magnitude of coefficients. A feature with a large scale will have a small coefficient even if it is informative, while a feature with a small scale will have a large coefficient even if it is noise. The penalty then treats them inequitably — it effectively under-regularises large-scale features and over-regularises small-scale ones.
2. **Gradient-based optimisers** converge faster and more reliably when all features are on similar scales.

Tree-based models (Random Forest, Gradient Boosting) are scale-invariant because splits depend only on feature rank order, not absolute values. However, applying StandardScaler to tree model inputs does not harm them, and using one consistent pipeline for all models avoids mistakes when switching.

### Why StandardScaler over MinMaxScaler or RobustScaler

- **MinMaxScaler** maps to [0,1] range but is sensitive to outliers — one extreme value in a feature collapses the rest of the distribution into a narrow band. The feature distributions here span wide ranges with some outliers.
- **RobustScaler** uses median and IQR, which is robust to outliers but produces features that are not unit-variance. StandardScaler's unit-variance property is important for Ridge/Lasso penalty calibration.
- **StandardScaler** is the standard choice when the data is approximately normal, which all 15 base features approximately satisfy (the outliers are in the *target*, not the features).

---

## What Was Not Done (and Why)

### Removing outlier rows from training
The PCA scatter plot coloured by extreme-target flag showed **no spatial separation** between extreme and normal rows in the 2D PCA projection. Extreme rows are not a geometrically distinct cluster — they appear to be produced by the same feature-generating mechanism as normal rows, just at higher amplitudes. Removing them would discard 84 training samples (3.36%) while likely not improving generalisation on the private leaderboard (which presumably contains the same distribution of extreme targets).

### Log-transforming the target
A log transform would require all target values to be positive. The target has a minimum of −41,008. A signed log (e.g., `sign(y) * log(1 + |y|)`) was considered but rejected: it changes the metric being optimised. R² on log-transformed targets measures a different quantity than R² on raw targets, and the competition evaluates on raw targets.

### Feature selection before modelling
Correlation with target is a weak proxy for feature importance in the presence of interactions and non-linearities. Removing features at this stage based on correlation thresholds could discard features that are important in combination. It is better to let regularisation (Lasso alpha, tree max_depth, etc.) handle selection during model training, where cross-validation gives honest feedback.

---

## Output File Reference

| File | Contents | Shape |
|---|---|---|
| `processed/X_train.csv` | Scaled engineered features, training set | 2500 × 28 |
| `processed/X_test.csv` | Same pipeline applied to test | 2500 × 28 |
| `processed/y_train.csv` | Raw target | 2500 × 1 |
| `processed/y_train_winsorized.csv` | Target clipped to [−2654.94, +1591.06] | 2500 × 1 |
| `processed/is_extreme.csv` | 1 if \|target\| > 1000, else 0 | 2500 × 1 |
| `processed/train_ids.csv` | Training row IDs | 2500 × 1 |
| `processed/test_ids.csv` | Test row IDs (for submission) | 2500 × 1 |
| `processed/feature_names.txt` | Ordered list of 28 feature names | — |
| `processed/imputer.joblib` | Fitted `SimpleImputer` (train medians) | — |
| `processed/scaler.joblib` | Fitted `StandardScaler` | — |

The `.joblib` files allow the exact same transforms to be applied to any new data without re-fitting.
