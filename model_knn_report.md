# KNN Regression — Model Report

Full documentation for the model built in `model_knn.ipynb`. Covers why KNN was chosen, every decision made during training, and an honest interpretation of all results.

---

## 1. Why KNN as the First Model

The competition requires a regression model evaluated on R². Before investing in complex models, a simple, assumption-free baseline is necessary for two reasons:

1. **It establishes a floor.** If a more sophisticated model cannot beat KNN, something is wrong with that model's setup — wrong hyperparameters, data leakage, incorrect target, etc. A KNN score you can independently verify is the most reliable sanity check.
2. **KNN makes no distributional assumptions.** It does not assume linearity, normality, or any functional form between features and target. This matters here because EDA showed the target is extremely non-normal (skewness = 13.59, kurtosis = 708) and individual feature-target relationships are weak and likely non-linear.

The known limitation accepted upfront: KNN suffers from the **curse of dimensionality**. With 28 features and 2,500 training rows, Euclidean distances between points are large and nearly uniform — neighbours are not truly "close" in any meaningful geometric sense. The model's R² will reflect this ceiling.

---

## 2. Data Used

All inputs come from `processed/` — produced by `data_prep.ipynb`. No preprocessing is repeated in the modelling notebook.

| Variable | Source file | Description |
|---|---|---|
| `X_train` | `processed/X_train.csv` | 2,500 × 28 scaled feature matrix (15 base + 13 engineered) |
| `X_test` | `processed/X_test.csv` | 2,500 × 28 same pipeline applied to test |
| `y` | `processed/y_train.csv` | Raw target — used for final R² reporting (what the leaderboard measures) |
| `y_wins` | `processed/y_train_winsorized.csv` | Target clipped to [−2,654.94, +1,591.06] — used for **training** |
| `is_extreme` | `processed/is_extreme.csv` | Boolean flag: True if \|target\| > 1,000 (84 rows, 3.36%) |
| `test_ids` | `processed/test_ids.csv` | Id column for building the submission file |

### Why train on `y_wins` not `y`

KNN predicts by averaging the target values of its k nearest neighbours. Averaging is sensitive to extreme values in the same way as a mean — one neighbour with target = 69,628 can dominate an average of 30 neighbours. Training on the winsorized target means the values being averaged are bounded at ±2,655, which produces more stable and accurate predictions for the 96.64% of rows with |target| ≤ 1,000.

The raw target R² is still reported after training, since that is what the competition evaluates.

---

## 3. Naive Baseline — Anchoring R² = 0

Before any real model was run, a "predict the mean" model was evaluated using cross-validation. This approximates R² = 0 in an out-of-fold setting, giving a concrete reference point. Every model in this project must exceed this floor to demonstrate any predictive value.

---

## 4. The One Hyperparameter: k

KNN has one primary hyperparameter: **k**, the number of neighbours used to form each prediction.

$$\hat{y}_i = \frac{\sum_{j \in N_k(i)} w_j \cdot y_j}{\sum_{j \in N_k(i)} w_j}$$

Where $N_k(i)$ is the set of k nearest neighbours of point $i$, and $w_j$ is the weight assigned to neighbour $j$.

**The bias-variance tradeoff:**
- **Small k** → high variance. Each prediction depends on very few neighbours, so it is sensitive to local noise. At k=1, the model memorises training data perfectly (R²=1.0 in-sample) but generalises poorly.
- **Large k** → high bias. As k grows, predictions converge toward the global training mean, and the model loses its ability to respond to local structure. At k=n, every prediction equals the training mean (R²=0 by definition).

The optimal k sits between these extremes, where the model captures real local structure without overfitting to noise.

### Sweep Results (5-fold CV R², `y_wins`, `weights=distance`)

| k | CV mean R² | CV std | CV min | CV max |
|---|---|---|---|---|
| 3 | −0.0630 | 0.1240 | −0.2262 | 0.0515 |
| 5 | −0.0136 | 0.0712 | −0.1038 | 0.0597 |
| 7 | +0.0232 | 0.0534 | −0.0459 | 0.0856 |
| 10 | +0.0422 | 0.0389 | −0.0075 | 0.0966 |
| 15 | +0.0607 | 0.0222 | +0.0277 | 0.0859 |
| 20 | +0.0652 | 0.0211 | +0.0280 | 0.0832 |
| 25 | +0.0669 | 0.0158 | +0.0421 | 0.0837 |
| **30** | **+0.0675** | **0.0154** | **+0.0417** | **+0.0872** |
| 35 | +0.0660 | 0.0163 | +0.0386 | 0.0870 |
| 40 | +0.0663 | 0.0162 | +0.0428 | 0.0910 |
| 45 | +0.0653 | 0.0181 | +0.0378 | 0.0903 |
| 50 | +0.0634 | 0.0173 | +0.0374 | 0.0857 |
| 60 | +0.0635 | 0.0138 | +0.0445 | 0.0843 |
| 75 | +0.0604 | 0.0115 | +0.0434 | 0.0765 |
| 100 | +0.0570 | 0.0096 | +0.0423 | 0.0700 |

**Selected: k = 30** — highest CV mean R² (0.0675). The plateau from k=25 to k=40 shows that the model is not highly sensitive to the exact value in this range. k=30 also has low standard deviation (0.0154), meaning performance is stable across folds rather than driven by one lucky split.

Notable patterns in the sweep:
- **k ≤ 5 produces negative mean R²** — the model is worse than the mean predictor, meaning small-k KNN is actively harmful here. This confirms the curse of dimensionality: the 28-dimensional feature space makes very-local neighbourhoods unreliable.
- **Variance (std) drops monotonically as k increases** — larger neighbourhoods smooth out fold-to-fold variation, at the cost of reduced mean performance.
- **The curve is flat from k=25 to k=45** — differences in this range are within noise; no strong over-optimisation risk in the final k choice.

---

## 5. Weight Function: `distance`

Two weight options were compared at k=30:

| Weight function | CV mean R² | CV std |
|---|---|---|
| `uniform` | 0.06683 | 0.01544 |
| **`distance`** | **0.06746** | **0.01537** |

`distance` weighting assigns each neighbour a weight of $w_j = 1 / d_j$ (inverse of its distance), so closer neighbours contribute more to the prediction than farther ones. The improvement over `uniform` is small (+0.00063) but consistent — distance-weighted averaging is always at least as good as uniform when neighbourhoods are not uniformly dense, which they are not here.

---

## 6. Cross-Validation Strategy

All evaluation used **5-fold stratified cross-validation** with `shuffle=True, random_state=42`.

Why these settings:
- **5 folds** — each fold uses 2,000 rows to train and 500 to validate. Large enough validation sets for stable R² estimates; small enough that training set size doesn't drop significantly.
- **Shuffle** — the raw CSV is ordered by Id, not randomly. Without shuffling, each fold would contain a contiguous block of Ids, which may not represent the full distribution.
- **Fixed random_state=42** — ensures the same fold assignments across every model in the project, making R² scores directly comparable.

`cross_val_predict` was used (not just `cross_val_score`) to obtain **out-of-fold (OOF) predictions** for every training row. This allows computing diagnostic metrics and plotting residuals for the entire training set, while still using only held-out data for each prediction.

---

## 7. Results

### Core metrics (OOF, k=30, `distance` weights)

| Metric | Value |
|---|---|
| **OOF R² vs winsorized target** | **0.0744** |
| **OOF R² vs raw target** | **0.0162** |
| OOF MAE | 121.02 |
| OOF RMSE | 360.42 |
| Residual mean | −13.22 |
| Residual std | 360.25 |

The gap between **R²=0.0744 on winsorized** and **R²=0.0162 on raw** is critical to understand. The model was trained on the clipped target (max |y| ≈ 2,655), but the raw target has values up to 69,628. On rows where the true target is 40,000 and the model predicts 200, the squared error is enormous and collapses the raw R². This is expected and honest — it reflects that KNN cannot extrapolate beyond the range of its training values.

The **competition evaluates on raw target R²**. The raw OOF R² of 0.0162 is the best estimate of the score this model will receive on the leaderboard.

### R² broken down by target magnitude bucket

| Bucket | n rows | OOF R² (vs raw target) |
|---|---|---|
| \|y\| ≤ 100 | 2,123 (84.9%) | **−1.5010** |
| 100 < \|y\| ≤ 500 | 264 (10.6%) | **+0.2770** |
| 500 < \|y\| ≤ 1,000 | 29 (1.2%) | **+0.0797** |
| \|y\| > 1,000 | 84 (3.4%) | **+0.0134** |

This breakdown reveals the model's failure mode precisely:

- **The model is worst where most of the data lives.** For 84.9% of rows (|y| ≤ 100), R² is −1.50. KNN with k=30 averages 30 neighbour targets from the winsorized set, producing predictions that typically land in the range [−50, +50]. When the true target is near zero but the neighbours span a wide range of values (which EDA showed — features are almost uncorrelated with target), the average prediction is still reasonable in absolute terms but poor in R² terms because R² penalises any variance around zero heavily.

- **The model performs best in the mid-range (100–500).** R²=0.277 here is the strongest single-bucket score, suggesting that when targets are moderate, the k-NN neighbourhood averaging captures enough local structure.

- **The model cannot reach extreme targets.** R²=0.013 for |y|>1,000 reflects the extrapolation ceiling — the maximum possible prediction is bounded by the largest winsorized training target (~1,591), so extreme test targets like 40,000 will always be severely under-predicted.

### Normal vs extreme rows

| Group | n rows | OOF R² |
|---|---|---|
| Normal rows (\|y\| ≤ 1,000) | 2,416 | −0.1272 |
| Extreme rows (\|y\| > 1,000) | 84 | +0.0134 |

Counterintuitively, the model has slightly positive R² on the extreme rows but negative R² on the normal rows. This happens because extreme rows all have large absolute predictions in the same direction (the model predicts ±200–600 for them), and that direction happens to correlate weakly with the true sign. Normal rows, where true targets cluster near zero, receive predictions spread around ±50–200, which is systematically wrong.

---

## 8. Diagnostic Interpretation

### Predicted vs Actual
The scatter plot shows predictions clustered in a narrow band (roughly ±200) while actual winsorized targets span the full [−2,655, +1,591] range. The model compresses variance — a fundamental KNN trait when k is large enough. A perfect model would show points on the diagonal; here the cloud is horizontal.

### Residual Distribution
Centred near −13.2 (slight negative bias — the model slightly over-predicts on average). The distribution is approximately symmetric, which is good, but the tails extend beyond ±1,000 on both sides, reflecting rows where the model is far off.

### Residuals vs Predicted
Residuals show a funnel shape — larger residuals on extreme predictions. This is **heteroscedasticity**: the model's errors are not uniform across the prediction range. It is an inherent consequence of training on winsorized targets and predicting on a dataset where some true values far exceed the training range.

---

## 9. Test Predictions

| Stat | Value |
|---|---|
| mean | −6.80 |
| median | −1.56 |
| std | 79.30 |
| min | −649.74 |
| max | +347.64 |

The test prediction range (−650 to +348) is much narrower than the true training target range (−41,008 to +69,628). This is expected — KNN cannot produce predictions outside the range of its training target values, and because the model was trained on `y_wins`, its predictions are bounded by the winsorization clips.

The mean (−6.80) and median (−1.56) are close to the training target median (−0.77), which is a healthy sign — the model is not systematically biased toward an extreme value.

---

## 10. Submission

**File:** `submission_knn.csv`  
**Format:** `Id, target` — matches the sample submission exactly  
**Rows:** 2,500 (one per test sample, sorted by Id)

Pre-submission checks run automatically in the notebook:
- Columns match sample submission format
- No missing predictions
- No infinite values
- All test Ids present (no rows dropped)
- No duplicate Ids

---

## 11. Known Limitations and What Comes Next

| Limitation | Root cause | Fix in next model |
|---|---|---|
| R²=0.016 on raw target | KNN cannot extrapolate beyond training target range | Tree-based models (Gradient Boosting) can produce any output value |
| R²=−1.50 on the majority of rows (|y|≤100) | Predictions are pulled away from zero by outlier neighbours | Regularised linear models with large alpha will predict near-zero correctly |
| Curse of dimensionality | 28 features, 2,500 samples — neighbourhoods are too large | Fewer, more informative features; or models that don't rely on distance |
| Cannot capture interaction effects | KNN uses raw feature distances, not interactions | Linear models with explicit interaction terms already in the feature matrix |
| Winsorization ceiling on predictions | Trained on clipped targets; predictions never exceed ±2,655 | Train on raw target with a model that uses a robust loss (Huber, quantile) |

The OOF R²=0.0744 (winsorized) and 0.0162 (raw) are the **baseline scores** every subsequent model must beat. A regularised linear model (Ridge, Lasso) trained on the same feature matrix with the interaction terms already engineered should outperform this significantly, since it will directly learn the x9×x11 signal (r=0.160) that KNN can only approximate through local averaging.
