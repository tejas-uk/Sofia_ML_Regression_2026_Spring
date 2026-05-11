# Evaluation Metric: R² (Coefficient of Determination)

## What Is R²?

R² measures how much of the variance in the target variable your model explains compared to a naive baseline (predicting the mean every time).

$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$$

Where:
- $SS_{res} = \sum (y_i - \hat{y}_i)^2$ — sum of squared residuals (your model's errors)
- $SS_{tot} = \sum (y_i - \bar{y})^2$ — total variance in the target (baseline errors)

## Interpreting the Score

| R² Value | Meaning |
|---|---|
| **1.0** | Perfect prediction — model matches every target exactly |
| **0.0** | Model is no better than predicting the mean for every sample |
| **< 0** | Model is actively worse than the mean baseline |

There is no lower bound — a sufficiently bad model can produce arbitrarily negative R².

## Why R² Is the Right Metric Here

This competition uses anonymous features with no domain meaning, so absolute error scales are uninterpretable. R² is scale-invariant: it doesn't matter whether targets range from 0–1 or −10,000–10,000, the metric always reflects relative predictive power against the same baseline.

It also directly rewards variance explanation. A model that captures the shape of the data (high-variance regions) benefits more than one that just minimizes small, uniform errors — which aligns with what regression generalization actually means.

## Computing R² in scikit-learn

```python
# On a held-out split
from sklearn.metrics import r2_score
r2 = r2_score(y_true, y_pred)

# Equivalent — any sklearn regressor's .score() method returns R²
r2 = model.score(X_test, y_test)

# Cross-validated R² (more reliable than a single split)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5, scoring="r2")
print(scores.mean(), scores.std())
```

## Overfitting Warning

The competition ranks first on a **Public** test set, then re-ranks on a **Private** test set. R² on training data or the public leaderboard can be misleading — a model that overfits will rank high publicly but collapse privately.

To avoid this:
- Use cross-validation (not a single train/val split) to estimate true generalization
- Prefer simpler models when cross-validated R² is similar to a complex model's
- Regularized models (Ridge, Lasso, ElasticNet) tend to generalize better than unregularized linear regression on noisy data

## Baseline Target

The competition awards full baseline points for any valid submission. A cross-validated R² consistently above **0.0** means you are beating the mean predictor. Competitive submissions typically push this as close to **1.0** as possible without overfitting.
