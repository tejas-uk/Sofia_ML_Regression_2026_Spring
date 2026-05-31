# The Full Journey — From a Stuck 0.02 to a 0.046 Leaderboard Score

This document tells the **complete story** of how this project's model went from being stuck at
R² ≈ 0.02 to scoring **0.04623** on the competition leaderboard — in chronological order, with the
reasoning behind every decision explained in plain language.

If you want the short "recipe" version, read `v3/TUTORIAL.md`. This file is the *story* — it shows
how the thinking actually unfolded, including the dead ends, because the dead ends are where most
of the learning is.

---

## Table of contents

1. [The setup: what we were trying to do](#1-the-setup)
2. [The single most important fact about the data](#2-the-most-important-fact)
3. [Phase 1: Why the existing models were stuck](#3-phase-1-why-stuck)
4. [Phase 2: The "obvious fix" that made things worse](#4-phase-2-obvious-fix-fails)
5. [Phase 3: The fix that worked — clip-stabilization](#5-phase-3-clip-stabilization)
6. [Phase 4: Choosing the right model](#6-phase-4-choosing-model)
7. [Phase 5: Blending for safety](#7-phase-5-blending)
8. [Phase 6: The two-stage experiment](#8-phase-6-two-stage)
9. [Phase 7: The leaderboard revelation](#9-phase-7-leaderboard)
10. [Phase 8: The submission ladder strategy](#10-phase-8-ladder)
11. [Phase 9: The plot twist #2 — bigger predictions scored WORSE](#11-phase-9-pivot)
12. [Where things stand & what's next](#12-where-we-stand)
13. [Glossary](#13-glossary)

---

<a name="1-the-setup"></a>
## 1. The setup: what we were trying to do

A Kaggle-style competition. We're given:
- A **training set**: 2,500 rows, each with 15 anonymous number columns (`x0`–`x14`) and a known
  answer (`target`).
- A **test set**: 2,500 rows with the same columns but **no answer** — we must predict it.

We submit our predicted `target` values and get scored. The score is **R²** (explained below).
You can submit up to **5 times per day**. The features have no real-world meaning — they're just
numbers — so we can't use domain knowledge. It's pure pattern-finding.

**Goal:** beat the competition's "baseline" score of **0.04**.

---

<a name="2-the-most-important-fact"></a>
## 2. The single most important fact about the data

Before building anything, we looked at the `target` column. Here's what it looks like:

| Statistic | Value | Translation |
|---|---|---|
| Median | −0.77 | A typical answer is basically zero |
| Min / Max | −41,008 / +69,628 | …but a few answers are *enormous* |
| Skewness | 13.6 | Wildly lopsided (a bell curve is 0) |
| Kurtosis | 708 | Insanely heavy "tails" (a bell curve is 0) |

In words: **about 97% of the answers are tiny (near zero), and about 3% (84 rows) are gigantic.**

### Why this one fact controls everything

The score, R², is built on **squared errors**. Squaring means big mistakes are punished
*enormously*. If the true answer is 50,000 and you guess 100, your error is 49,900 — squared, that's
~2.5 *billion*. A single row like that can swamp the errors from thousands of normal rows.

So the score is **dominated by those ~84 giant rows**. Predict them okay-ish and your score moves;
predict the other 2,400 perfectly but botch the giants and your score barely budges. Remember this —
it is the hero *and* the villain of the whole story.

> **R² in one line:** 1.0 = perfect, 0.0 = no better than always guessing the average, below 0 =
> *worse* than guessing the average (which happens constantly here).

---

<a name="3-phase-1-why-stuck"></a>
## 3. Phase 1: Why the existing models were stuck at ~0.02

Three models already existed in the repo, all scoring about 0.02:

| Model | Trained to predict… |
|---|---|
| v1 KNN | a **winsorized** target (giant values clipped to ±1,591) |
| v1 MLP (neural net) | the same winsorized target |
| v2 Gradient Boosting | a **signed-log** target (giants math-compressed to be small) |

Notice the pattern: **every model was trained on a *modified* version of the answer** where the
giant values had been shrunk or removed.

### The "aha" moment

The grader scores you on the **original, raw** answers — including the giants. But every model was
*trained* to predict a version where the giants don't exist. We literally taught each model "never
predict anything big"… while being graded almost entirely on how well we predict the big ones.

It's like training for a sprint and being graded on a marathon. v2's gradient booster scored a
beautiful **0.65** in its own compressed world and **0.016** on what actually counted. It aced the
wrong test.

> **Lesson #1:** Your training goal must match the thing you're graded on. If you're scored on raw
> answers, you have to train (in some form) on raw answers.

---

<a name="4-phase-2-obvious-fix-fails"></a>
## 4. Phase 2: The "obvious fix" that made things worse (and taught us the most)

If transforming the answer is the problem, just train a plain model directly on the **raw** answer,
right? We tested it with **cross-validation** (split the data into 5 chunks, train on 4, test on the
5th, rotate — see glossary). Plain linear regression on raw answers gave:

```
The 5 chunk scores: [-0.08,  +0.045,  -6.59,  -0.36,  -0.07]
Average:            -1.41
```

Worse! But here's the critical habit: **don't just read the average — read all five scores.** Four
chunks are mediocre, but **one chunk scored −6.59**, single-handedly dragging the average into the
basement.

### Why one chunk explodes

The giant training rows yank the model's internal settings ("coefficients") around violently. When
the held-out test chunk happens to contain a giant row, the model spits out a wildly wrong huge
number, the squared error is astronomical, and that chunk's score craters.

We confirmed that turning regularization way up (a knob that forces the model to stay cautious)
removes the explosion — but only by making every prediction timid again, landing us right back at
~0.02.

> **Lesson #2:** Always look at the per-chunk cross-validation scores, not just the average. A scary
> average usually hides one unstable chunk — and that chunk is a giant arrow pointing at your real
> problem.

---

<a name="5-phase-3-clip-stabilization"></a>
## 5. Phase 3: The fix that worked — "clip the training answer"

We now understood the exact disease: **a handful of giant rows destabilize training.** The cure:
calm them down *while the model is learning*, but **keep grading on the real, raw answers.**

The trick — **clip only the answers the model learns from**, to ±2,000:

```python
y_train = np.clip(y_train, -2000, 2000)   # tame giants FOR TRAINING ONLY
model.fit(X_train, y_train)
preds = model.predict(X_val)
score = r2_score(y_val_RAW, preds)         # but GRADE on the real raw answers
```

We swept the clip level and ±2,000 was the sweet spot:

| Clip at | Result |
|---|---|
| ±1,000 (tight) | stable but a little too timid |
| **±2,000** | **stable AND no chunk explodes — best** |
| ±5,000 (loose) | the exploding chunk comes back |
| no clip | the −6.59 disaster from Phase 2 |

With this, all five chunks went positive: `[0.015, 0.016, 0.037, 0.041, 0.037]`. The disaster chunk
was gone.

### Why this differs from the old "winsorizing"

The old v1 approach clipped the answers **and graded itself in that clipped world**, so it never
learned to predict anything large. We clip **only what the model studies** and **grade on reality**,
and we use a looser limit (±2,000 vs ±1,591) so the model is still allowed to make somewhat-big
predictions.

> **Lesson #3:** When outliers wreck *training*, limit their influence on the model — but never
> change the yardstick you're measured against.

---

<a name="6-phase-4-choosing-model"></a>
## 6. Phase 4: Choosing the right model

Linear models capped out around 0.03. The data notes said feature `x9` has a **curved** relationship
with the answer, and the real signal hides in **combinations** of features (e.g. `x9 × x11`).
Tree-based models capture curves and combinations automatically, so we compared several model types
(all using the clip-±2,000 trick, all graded on raw answers):

| Model | Average raw score | note |
|---|---|---|
| Huber (a "robust" linear model) | 0.005 | stable but learns almost nothing |
| Ridge (plain linear) | 0.029 | the linear ceiling |
| Gradient Boosting | 0.021 | |
| Extra Trees | 0.025 | |
| **Random Forest (400 trees)** | **0.037** | **winner** |

Random Forest won clearly, and we re-ran it with 5 different random seeds to confirm it wasn't luck.

> **A counterintuitive lesson:** "Robust" models (like Huber) are *designed to ignore outliers*.
> That sounds perfect for outlier-heavy data — but here the outliers are exactly the rows the score
> rewards, so ignoring them throws away the points. The right move was clipping the *answer*, not
> using an outlier-ignoring *model*. Don't trust a technique because its name sounds right — test it.

---

<a name="7-phase-5-blending"></a>
## 7. Phase 5: Blending two models for safety

Random Forest had the best score, but competitions re-rank everyone on a **hidden "private" test set**
at the end, so we want *robustness*, not just a high practice score. We mixed in 40% of the simple
linear model:

```
final prediction = 0.6 × RandomForest + 0.4 × Ridge   (both on the clip-±2000 answer)
```

This is the **v3 model**. Final scoreboard:

| Model | Average raw cross-val score |
|---|---|
| v1 KNN | 0.0162 |
| v1 MLP | 0.0199 |
| v2 Gradient Boosting | 0.0156 |
| **v3: 0.6 RF + 0.4 Ridge** | **0.0354** |
| Competition baseline | 0.0400 |

Per-chunk: `[0.016, 0.015, 0.061, 0.045, 0.040]` — all positive, with a median of 0.0399 (right at the
baseline). We had roughly doubled the best previous model.
Files: `v3/model_linear_raw.ipynb` → `v3/submission_linear_raw.csv`.

> **Lesson #4:** Blending a flexible model with a simple, steady one usually survives the hidden
> test set better than either alone.

---

<a name="8-phase-6-two-stage"></a>
## 8. Phase 6: The two-stage experiment (a brilliant gamble)

Then we tried the "advanced" idea: since ~3% of rows are giants, why not (1) build a **detector** that
flags likely-giant rows, then (2) use a **special model** just for them? We combine softly:
`prediction = (1 − chance_it's_giant) × normal_model + chance_it's_giant × giant_model`.

What we found:

1. **Good news:** the giant-detector works well — it identifies giant rows with **82% accuracy**
   (AUC 0.82; 50% would be random). So we *can* tell which rows are giants. (This actually corrected
   an earlier project belief that giants were undetectable — they're just not detectable by *simple
   straight-line* methods.)
2. **Bad news:** knowing a row is a giant doesn't tell us *how* giant, or in which *direction*. With
   only ~67 giant rows to learn from, the special model guesses magnitude and sign poorly.
3. **The result — high reward, high risk:** 4 of the 5 cross-val chunks shot *up* to 0.06–0.13
   (excellent!), but **one chunk always collapsed** to between −0.4 and −3.2 (a confident wrong guess
   on a held-out giant). On *average* it didn't beat the simpler v3 model.

So the two-stage model is a **gamble**: usually great, occasionally catastrophic. We judged it too
risky to recommend, kept v3 as the safe pick, and saved a *conservative* two-stage version as an
optional second entry.

> **Lesson #5:** A model that's *usually brilliant but sometimes catastrophic* often looks *worse*
> on average than a steady, boring one. We almost dismissed the two-stage model entirely because of
> this. (Phase 7 is why we shouldn't have.)

---

<a name="9-phase-7-leaderboard"></a>
## 9. Phase 7: The leaderboard revelation (the plot twist)

We had been judging models **only by cross-validation**, with zero feedback from the real
competition. So we submitted. The conservative two-stage entry — the one with an unimpressive
cross-val *average* of 0.0336 — scored **0.04623 on the real leaderboard**, our best result.

This was the most important learning in the whole project. Here's why.

Recall the two-stage's five cross-val chunk scores: `[0.030, 0.026, −0.036, 0.074, 0.074]`. The
*average* (0.0336) is dragged down by that one weak −0.036 chunk. But the **typical (median)** chunk
is ~0.03–0.07. The real leaderboard scored 0.046 — i.e. **the public test set behaves like one of
the GOOD chunks. The disaster chunk we feared simply didn't happen on the real data.**

### What this changes

We had been selecting models by their cross-val **average**. But the leaderboard tracks the
**median / good-chunk** behavior, not the average. **That flips our entire selection rule:** the
aggressive two-stage configs we *threw away* for having bad averages actually have the **best
medians** — so they're now our top contenders.

> **Lesson #6:** Cross-validation is a proxy, not the truth. Get real feedback as early as possible,
> then figure out *which cross-val statistic actually predicts your real score.* Here it was the
> median, not the mean — and that realization was worth more than any modeling trick.

---

<a name="10-phase-8-ladder"></a>
## 10. Phase 8: The submission ladder strategy

Armed with "the leaderboard rewards high-median configs," we built a **ladder** of increasingly
aggressive two-stage models to climb the leaderboard with the daily submissions:

| Submission file | Cross-val median (leaderboard proxy) | Worst chunk (private-test risk) | Prediction range |
|---|---|---|---|
| `submission_two_stage.csv` *(scored 0.04623)* | 0.030 | −0.04 | ±800 |
| `submission_ts_soft_cap8000.csv` | 0.053 | −0.40 | ±1,600 |
| `submission_ts_signc_s04.csv` | 0.074 | −0.84 | ±2,300 |
| `submission_ts_signc_s06.csv` | 0.082 | −1.82 | ±3,300 |

The two columns capture the core trade-off:
- **Cross-val median** = how high it should score on the *public* leaderboard (higher = better).
- **Worst chunk** = how badly it could blow up on the hidden *private* leaderboard (more negative =
  riskier).

### The plan

1. Submit the ladder **in order**, one rung at a time, watching the public score. Submitting the
   gentlest step first also *verifies* the "median predicts leaderboard" theory with a second data
   point before betting big.
2. Keep the safe 0.046 entry (`submission_two_stage.csv`) as the **hedge for the private leaderboard**,
   in case the private test set turns out to resemble our disaster chunk.

> **Lesson #7:** With limited submissions and a hidden final test, treat the leaderboard like a
> careful experiment: change one thing at a time, confirm your theory, and always keep a safe fallback.

---

<a name="11-phase-9-pivot"></a>
## 11. Phase 9: The plot twist #2 — bigger predictions scored WORSE

The ladder plan from Phase 8 assumed "higher CV-median → higher leaderboard." We submitted more of it,
and the real scores **demolished that assumption**:

| Submission | How big its predictions get | **Real leaderboard** |
|---|---|---|
| conservative two-stage (±800) | small | **0.04623** (still the best) |
| `cap8000` (±1,600) | medium | 0.02 |
| `sign-coupled s0.6` (±3,300) | large | −0.157 |

The pattern is now crystal clear and *backwards* from what cross-validation said: **the bigger our
predictions, the worse we score.** Every attempt to bet harder on the giant rows made things worse.

### Why? The "pooled CV" trap

We ran one decisive experiment (in `v3/model_v4_conservative.ipynb`): take our predictions, multiply
them all by a scale `s`, and measure the score two different ways.

- **"Pooled" way** (mush all the cross-val test pieces together, score once): wants `s ≈ 2.6` —
  i.e. "make your predictions almost 3× bigger!"
- **"Per-fold" way** (score each of the 5 pieces separately, then look at the typical one): wants
  `s < 1` — i.e. "make your predictions *smaller*."

These two completely disagree. And the **real leaderboard sided with the per-fold way.**

The reason is the 84 giants again. The "pooled" score is dominated by them, and it loves big
predictions because a big (correctly-aimed) guess shrinks a giant's enormous squared error. But on the
**real test set, the giants are unpredictable** — we can detect *that* a row is a giant (82% accuracy)
but not *how big* or *which direction*. So betting big on them is betting on noise, and the leaderboard
punishes it. The "pooled" cross-validation score was lying to us the whole time.

> **Lesson #8 (the most subtle one):** On data ruled by a few outliers, the obvious averaged
> cross-validation score can actively mislead you. Always look at the *per-piece* (per-fold)
> distribution — and trust the real leaderboard over any offline estimate.

### What we also ruled out

We checked whether **richer features** would help — adding missing-value flags, a smooth curve for
`x9`, and *all* 105 pairwise feature-combinations instead of our hand-picked 13. **None helped; the
fancy combinations even hurt** (they fed the model noise). The 28 features we already had are about as
good as this data allows. Feature engineering is a closed book here.

### The resulting model (v4)

Given all this, the right move is *not* a bigger model — it's a **more conservative** one. The v4 model
is simply our conservative two-stage with its **output shrunk to 70%**. That makes every cross-val fold
positive and keeps the predictions small (range ±561), which is exactly what the leaderboard rewards.
Files: `v3/model_v4_conservative.ipynb` → `v3/submission_v4_conservative.csv`.

---

<a name="12-where-we-stand"></a>
## 12. Where things stand & what's next

**Current best public score: 0.04623** (conservative two-stage, ±800 predictions), above the 0.04 baseline.

**What we now know for sure:**
- Bigger predictions → worse leaderboard. **Stay conservative.**
- Richer features don't help. **The 28 features are enough.**
- The giant rows are unpredictable noise on the real test set. **Don't chase them.**
- We are at/near the genuine signal ceiling (~0.046). Remaining gains are small.

**Immediate next step:** submit the conservative bracket to pinpoint the prediction-range "peak" —
`submission_ts_cap2000_s03.csv` (±639) and `submission_v4_conservative.csv` (±561). Whichever scores
highest tells us the ideal prediction scale.

**Longer-shot ideas if we want to push past the ceiling:**
- *Oracle experiments* — calculate the best score theoretically possible to know if any headroom exists.
- *Refocus entirely on the middle band* — the `100 < |answer| < 500` rows score a strong +0.30; a model
  that confidently predicts that band and forces everything else near zero might edge higher.
- *Accept the ceiling* — 0.046 already clears the baseline; the safest play is to lock in the best
  conservative submission for the private leaderboard.

---

<a name="13-glossary"></a>
## 13. Glossary

- **Feature (`X`)**: an input column. We have 15 raw ones plus 13 engineered combinations = 28.
- **Target (`y`)**: the number we predict.
- **R²**: the score. 1 = perfect, 0 = no better than guessing the average, <0 = worse than that.
- **Heavy-tailed / skewed**: a distribution where most values are small but a few are extreme.
- **Cross-validation (CV)**: split data into k chunks (here 5), train on k−1 and test on the last,
  rotate through all chunks. Gives k scores so you can see stability, not just one lucky result.
- **Mean vs. median of the chunk scores**: mean = average (sensitive to one bad chunk); median =
  the middle value (ignores extremes). We learned the *median* predicts the leaderboard here.
- **Pooled vs. per-fold scoring**: *pooled* = combine every chunk's test predictions and compute one
  score (dominated by outliers — it misled us). *Per-fold* = score each chunk separately then
  summarize (this tracked the real leaderboard). On outlier-heavy data, prefer per-fold.
- **Clipping**: forcing values to stay within a range (e.g. anything above 2,000 becomes 2,000).
- **Winsorizing**: clipping based on percentiles; the old approach's mistake was *also grading*
  in the clipped world.
- **Regularization**: a knob that forces a model to stay simple/cautious to avoid overfitting.
- **Random Forest / Gradient Boosting / Extra Trees**: models built from many decision trees; they
  capture curves and feature-combinations automatically.
- **Ridge / Huber**: linear (straight-line) models; Huber deliberately downplays outliers.
- **Ensemble / blend / stack**: combining several models' predictions into one.
- **Two-stage model**: first classify a row (giant or not), then route it to a specialist model.
- **AUC**: a classifier-quality score; 0.5 = random guessing, 1.0 = perfect. Ours was 0.82.
- **Public vs. private leaderboard**: the live score you see is computed on a *public* slice of the
  test set; final rankings use a *hidden private* slice. A model can look great publicly and flop
  privately — which is why we keep a safe fallback.
