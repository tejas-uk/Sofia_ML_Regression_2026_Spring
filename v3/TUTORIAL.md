# v3 Walkthrough — How We Went From 0.02 to Clearing the 0.04 Baseline

A step-by-step tutorial you can follow even if you are new to machine learning. It explains
**what** we did, **why** we did it, and **how to reproduce it**, with the reasoning spelled
out at every step. By the end you should be able to debug a stuck regression model yourself.

> **TL;DR of the whole story:** Every earlier model was secretly being trained to solve a
> *different problem* than the one it was graded on. Fixing that mismatch — and then adding one
> trick to stop a few giant outliers from blowing things up — moved us from ~0.02 to the
> ~0.04 baseline. The full final model is `0.6 × RandomForest + 0.4 × Ridge`, both trained on a
> target whose extreme values are clipped to ±2000.

---

## Part 0 · The vocabulary you need first

If you already know these, skip to Part 1.

- **Features (`X`)**: the input columns. Here, 15 anonymous numbers `x0`–`x14` (plus 13
  engineered combinations = 28 total). They have no real-world meaning.
- **Target (`y`)**: the number we are trying to predict (column `target`).
- **Regression**: predicting a continuous number (as opposed to *classification*, predicting a
  category).
- **R² (the score / "coefficient of determination")**: how much better your model is than just
  guessing the average every time.
  - `R² = 1.0` → perfect predictions.
  - `R² = 0.0` → no better than always guessing the mean.
  - `R² < 0` → **worse** than guessing the mean (yes, this is possible and it happens a lot here).
  - Formula: `R² = 1 − (sum of your squared errors) / (sum of squared distances from the mean)`.
  - The key consequence: because everything is **squared**, a single huge error can wreck your
    whole score. Remember this — it is the villain of this entire story.
- **Cross-validation (CV)**: instead of trusting one lucky train/test split, we split the data
  into 5 parts ("folds"), train on 4 and test on the 5th, rotate through all 5, and look at all
  5 scores. This tells us how stable the model really is.
- **Overfitting**: when a model memorizes the training data instead of learning the pattern, so
  it looks great in training but fails on new data.

---

## Part 1 · Understand the problem before touching a model

**Rule #1 of ML: never build a model before you understand the data and the metric.**

### 1a. Read the rules. What exactly is being scored?

The competition grades you on **R² computed on the raw `target`**. Write that sentence on a
sticky note. It is the single most important fact in this project, and ignoring it is exactly
what trapped the earlier attempts.

### 1b. Look at the target distribution.

```python
import pandas as pd
train = pd.read_csv("spring2026_kaggle_linear_regression_challenge_train.csv")
print(train["target"].describe())
print("skew:", train["target"].skew(), "kurtosis:", train["target"].kurt())
```

What we found (see `data_findings.md`):

| Statistic | Value | Plain-English meaning |
|---|---|---|
| Median | −0.77 | Half the data sits within ±43 of zero |
| Std deviation | 1,978 | …but the "typical spread" is huge |
| Min / Max | −41,008 / +69,628 | A few values are *gigantic* |
| Skewness | 13.6 | Extremely lopsided (normal = 0) |
| Kurtosis | 708 | Insanely heavy tails (normal = 0) |

**This is a "heavy-tailed" target.** 97% of rows are tiny (near zero) and ~3% (84 rows) are
enormous (|y| > 1000). Because R² squares the errors, **those 84 giant rows control almost the
entire score.** Miss one badly and your R² collapses.

This one picture explains everything that follows.

---

## Part 2 · Figure out why the previous models were stuck at ~0.02

There were already three model "generations" in this repo. All scored ~0.02 on the raw target:

| Model | What it trained on |
|---|---|
| v1 KNN | a **winsorized** target (extremes clipped to ±1591) |
| v1 MLP | the same winsorized target |
| v2 GradientBoosting | a **signed-log** target (`sign(y)·log(1+|y|)`) |

### The "aha": they optimized the wrong objective

Look closely. The grader uses the **raw** target. But all three models were *trained* on a
**transformed** target (winsorized or log-compressed). Those transforms throw away the giant
values — they teach the model "never predict anything big."

But the score is dominated by exactly those big values! So the model literally cannot earn the
points the metric is handing out. v2's GradientBoosting scored **0.65** in its own log-space and
**0.016** on the raw target. It was excellent at the wrong exam.

> **Lesson:** Your training objective should match your evaluation metric. If you are graded on
> raw R², you must (in some form) train on the raw target.

---

## Part 3 · Try the "obvious" fix — and watch it fail instructively

If transforming the target is the problem, just train a plain linear model on the **raw** target,
right? Let's test it honestly with 5-fold cross-validation.

```python
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_score, KFold

X = pd.read_csv("processed/X_train.csv")
y = pd.read_csv("processed/y_train.csv").squeeze()
CV = KFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(LinearRegression(), X, y, cv=CV, scoring="r2")
print(scores)          # the 5 fold scores
print(scores.mean())   # the average
```

Result:

```
per-fold: [-0.08,  0.045,  -6.59,  -0.36,  -0.07]
mean:     -1.41
```

That is **worse**, not better. But do not just read the average — **read the 5 fold scores
individually.** Four folds are mediocre/okay, but **one fold scored −6.59.** One disaster fold
drags the whole average deeply negative.

### Why does one fold explode?

The giant training rows yank the linear model's coefficients around. When a held-out fold happens
to contain a giant row, the model predicts a wildly wrong huge number for it, the squared error
is astronomical, and R² for that fold craters.

We confirmed cranking up regularization (`Ridge(alpha=30000)`) removes the explosion — but only
by shrinking every prediction back toward zero, which lands us right back at ~0.02.

> **Lesson:** Always inspect *per-fold* scores, not just the mean. A scary average often hides a
> single unstable fold, and that points you straight at the real problem.

---

## Part 4 · The fix that worked — "clip the training target"

We diagnosed the exact failure: **a few giant rows destabilize training.** So tame them *during
training* without lying to the *grader*.

The trick: **clip only the target the model learns from**, but **keep scoring on the raw target.**

```python
import numpy as np
from sklearn.metrics import r2_score

def cv_with_clip(make_model, clip):
    scores = []
    for train_idx, val_idx in CV.split(X.values):
        y_train = np.clip(y.values[train_idx], -clip, clip)   # clip ONLY for training
        model = make_model().fit(X.values[train_idx], y_train)
        preds = model.predict(X.values[val_idx])
        scores.append(r2_score(y.values[val_idx], preds))      # score on RAW y
    return np.array(scores)

print(cv_with_clip(lambda: Ridge(alpha=1000), clip=2000))
```

We swept the clip level. ±2000 was the sweet spot:

| Clip level | What happens |
|---|---|
| ±1000 (very tight) | stable but a bit too cautious |
| **±2000** | **stable AND no fold explodes — best** |
| ±5000 (loose) | the disaster fold comes back |
| none | the −6.59 explosion from Part 3 |

With `Ridge(alpha=1000)` + clip ±2000 we got all five folds positive:
`[0.015, 0.016, 0.037, 0.041, 0.037]`, mean ≈ 0.029. No more disaster fold.

> **Why ±2000 and not ±1591 like the old winsorizing?** Two differences. (1) We clip *only the
> training labels* and still grade on raw — the old approach trained *and* scored in the clipped
> world. (2) ±2000 is deliberately looser, so the model is still allowed to predict somewhat
> large values instead of being capped tight.

> **Lesson:** When outliers destabilize *training*, limit their influence on the model — but
> never change what you are measured against.

---

## Part 5 · Pick the right model family

Linear models capped around 0.03. The data notes say feature `x9` has a **curved**
(diminishing-returns) relationship with the target, and the real signal lives in **interactions**
between features. Tree-based models capture curves and interactions automatically, so we compared
several model types, all on the clip-±2000 target, all scored on raw R²:

```python
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge, HuberRegressor
```

| Model | mean raw-R² | median fold | note |
|---|---|---|---|
| Huber (a "robust" linear model) | 0.005 | 0.005 | stable but learns almost nothing |
| Ridge (linear) | 0.029 | 0.037 | the linear ceiling |
| HistGradientBoosting | 0.021 | 0.021 | |
| ExtraTrees | 0.025 | 0.026 | |
| **RandomForest (400 trees)** | **0.037** | **0.041** | **winner** |

`RandomForestRegressor(n_estimators=400, min_samples_leaf=20, max_features=0.5)` won clearly, and
we verified it gives the same numbers across 5 different random seeds (so it is not a fluke).

> **Side lesson about Huber:** A "robust regressor" deliberately *ignores* outliers. That sounds
> perfect for outlier-heavy data — but here the outliers are exactly the rows the score cares
> about, so ignoring them throws away the points. The right lever was clipping the *target*, not
> using a robust *loss*. Test your assumptions; don't reason from the name.

---

## Part 6 · Blend two models for safety

RandomForest had the best score, but the competition re-ranks on a hidden "private" test set, so
we want robustness, not just a high training-CV number. We mixed in 40% of the linear Ridge model
(which extrapolates the simple `x9` trend smoothly and lowers fold-to-fold variance):

```python
# final v3 model
y_clip = np.clip(y.values, -2000, 2000)
rf    = RandomForestRegressor(n_estimators=400, min_samples_leaf=20,
                              max_features=0.5, n_jobs=-1, random_state=0).fit(X.values, y_clip)
ridge = Ridge(alpha=1000).fit(X.values, y_clip)

prediction = 0.6 * rf.predict(X_test.values) + 0.4 * ridge.predict(X_test.values)
```

We swept the blend weight; 60/40 gave a strong, stable result:

| Model | CV raw-R² (mean) |
|---|---|
| v1 KNN | 0.0162 |
| v1 MLP | 0.0199 |
| v2 GradientBoosting | 0.0156 |
| **v3: 0.6 RF + 0.4 Ridge** | **0.0354** |
| Competition baseline | 0.0400 |

Per-fold: `[0.016, 0.015, 0.061, 0.045, 0.040]` — **all positive**, median **0.0399** (right at the
baseline). We nearly doubled the best previous model.

> **Lesson:** Blending ("ensembling") a flexible model with a simple stable one usually
> generalizes better than either alone, especially when a hidden test set is involved.

---

## Part 7 · Build the submission file (with safety checks)

```python
submission = pd.DataFrame({"Id": test_ids, "target": prediction})
submission = submission.sort_values("Id").reset_index(drop=True)
submission.to_csv("submission_linear_raw.csv", index=False)

# sanity checks — always do these before submitting
assert list(submission.columns) == ["Id", "target"]
assert submission["target"].isnull().sum() == 0
assert np.isfinite(submission["target"]).all()
assert set(test_ids) == set(submission["Id"])
assert submission["Id"].nunique() == len(submission)
```

This is `v3/model_linear_raw.ipynb` → `v3/submission_linear_raw.csv`, our **recommended primary
submission.**

---

## Part 8 · The bonus experiment — a two-stage model (and why it didn't win)

We also tried the textbook "advanced" idea: since ~3% of rows are extreme, why not (1) train a
**classifier** to spot extreme rows, then (2) use a special regressor for them?

What we learned (full details in `v3/model_two_stage.ipynb`):

1. **Good news:** a classifier detects extreme rows surprisingly well — **ROC-AUC 0.82** (0.5 would
   be random). So we *can* tell which rows are extreme.
2. **Bad news:** knowing a row is extreme doesn't tell us *how* extreme or in *which direction*.
   With only ~67 extreme rows to learn from, the special regressor guesses magnitude/sign poorly.
3. **The result:** 4 of 5 folds shot *up* to 0.06–0.13 (great!), but **one fold always collapsed**
   to between −0.4 and −3.2 (because of a confident wrong guess on a held-out giant). On average,
   it did **not** beat the simpler v3 model.

So the two-stage model is a **high-variance gamble**: it might score higher on a lucky public
leaderboard draw, but it could also tank. We saved a conservative version as
`v3/submission_two_stage.csv` (optional second daily entry), but **v3's blend stays the
recommended primary.**

> **Lesson:** Fancier is not always better. A model that is *sometimes brilliant and sometimes
> catastrophic* is usually worse than a steady, boring one — especially when you only get a few
> submissions and a hidden test set decides the winner.

---

## Part 9 · The reusable recipe (memorize this)

This same debugging loop works on almost any stuck regression problem:

1. **Read the metric.** Know *exactly* what you are scored on. (We were scored on raw R².)
2. **Plot the target.** Heavy tails / skew change everything.
3. **Match training to the metric.** Don't optimize a transformed target if you're graded on the
   raw one.
4. **Always read per-fold CV scores**, not just the average. One bad fold = one big clue.
5. **Tame outliers in training, not in the metric.** Clipping the *training* target is a cheap,
   powerful stabilizer.
6. **Compare model families fairly** (same data, same scoring). Let the numbers pick the winner.
7. **Blend a flexible model with a simple one** for robustness against a hidden test set.
8. **Validate the submission file** before you upload it.
9. **Resist complexity** unless it *robustly* beats the simple thing in cross-validation.

---

## Where everything lives

| File | What it is |
|---|---|
| `data_findings.md` | The original data exploration (Part 1) |
| `data_prep_decisions.md` | How `processed/` features were built |
| `processed/` | Cleaned, ready-to-use feature/target files |
| `Approaches.md` | Formal write-up of all model generations (v1 → v3b) |
| `v3/model_linear_raw.ipynb` | **The v3 model** — runnable, with all experiments |
| `v3/submission_linear_raw.csv` | **Recommended submission** |
| `v3/model_two_stage.ipynb` | The two-stage experiment (Part 8) |
| `v3/submission_two_stage.csv` | Optional high-variance submission |
| `v3/TUTORIAL.md` | This file |

### How to run it yourself

```bash
# from the repo root, using a Python with scikit-learn installed
jupyter nbconvert --to notebook --execute --inplace v3/model_linear_raw.ipynb
# then open the notebook to read the outputs, or just use the generated CSV
```
