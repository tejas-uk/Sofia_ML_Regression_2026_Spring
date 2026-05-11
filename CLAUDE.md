# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Kaggle-style regression competition homework for Sofia University's Machine Learning class (Spring 2026). The goal is to train a regression model on anonymous multivariate features and predict a continuous `target` variable.

- **Evaluation metric**: R² (coefficient of determination) — maximize this; perfect score is 1.0
- **Submission limit**: 5 per day
- **Submission format**: CSV with `Id,target` columns (see `spring2026_sampleSubmission.csv`)

## Data

| File | Description |
|---|---|
| `spring2026_kaggle_linear_regression_challenge_train.csv` | Training set — columns: `x0`–`x14` (features), `target`, `Id` |
| `spring2026_kaggle_linear_regression_challenge_test.csv` | Test set — same feature columns, `Id` only (no `target`) |
| `spring2026_sampleSubmission.csv` | Submission format example |

Key data notes:
- 15 anonymous numeric features (`x0` through `x14`)
- **Missing values exist** in both train and test sets — handle before fitting
- Features carry no inherent domain meaning

## Recommended Workflow

```python
import pandas as pd
from sklearn.model_selection import cross_val_score

train = pd.read_csv("spring2026_kaggle_linear_regression_challenge_train.csv")
test  = pd.read_csv("spring2026_kaggle_linear_regression_challenge_test.csv")

X = train.drop(columns=["target", "Id"])
y = train["target"]
X_test = test.drop(columns=["Id"])

# Evaluate with cross-validation (R²)
scores = cross_val_score(model, X, y, cv=5, scoring="r2")
print(scores.mean())

# Generate submission
preds = model.predict(X_test)
submission = pd.DataFrame({"Id": test["Id"], "target": preds})
submission.to_csv("submission.csv", index=False)
```

## Submission Format

```csv
Id,target
0,0.5
6,-0.09
7,1000.0
```

The `Id` values in the submission must match those in the test set exactly.
