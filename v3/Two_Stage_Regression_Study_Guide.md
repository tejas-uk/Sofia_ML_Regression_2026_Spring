# Sign-Coupled Two-Stage Regression — Presentation Study Guide

A slide-by-slide study companion for `Two_Stage_Regression_Presentation.pptx`.
This explains **what** was done to build the winning submission `submission_ts_signc_s06.csv` (final **private** leaderboard R² **0.07434**) and, more importantly, **why** — including the k-fold training/evaluation method and the public-vs-private twist that decided the competition.

Source material: `v3/model_two_stage.ipynb`, `v3/model_v4_conservative.ipynb`, `data_prep_decisions.md`, `data_findings.md`, `v3/PROJECT_JOURNEY.md`, the `processed/` pipeline, and a live re-run of the cross-validation.

---

## The 30-second version (memorize this)

The target is wildly heavy-tailed — ~3.4% of rows are "giants" that dominate a squared-error R² score. The winning model is a **sign-coupled two-stage** model: (1) a classifier detects which rows are likely giants, and (2) a specialist regressor predicts only the **magnitude** of a giant while the **sign** is borrowed from the stable backbone model. It was tuned aggressively (shrink 0.6, cap 10000, predictions up to ±3,300). On cross-validation its **mean looked terrible (−0.30)** because one fold collapses, but its **median was 0.082** — and the median is what tracks the real board. On the hidden **private** test set it scored **0.07434**, almost exactly that median, and won.

---

## Slide 1 — Title

**Talking point:** This is the story of how `submission_ts_signc_s06.csv` won — the project's best result on the final private leaderboard.

Headline numbers:

- **0.07434** — the final *private* leaderboard R², the winning score (~2× the 0.040 baseline).
- **sign-coupled** — the key mechanism: the specialist predicts magnitude, the sign comes from the backbone.
- **5-fold** — the evaluation method, and the lesson that you read the *median* of the folds, not the mean.
- **±3,300** — how large the predictions get (this was the *aggressive* configuration).

---

## Slide 2 — The task

**What the competition is.** Given 2,500 training rows with 15 anonymous numeric features (`x0`–`x14`) and a known `target`, predict the target for 2,500 test rows.

**The four framing facts:**

- **Data** — 15 features with no real-world meaning, ~1–1.5% missing per feature. Pure pattern-finding.
- **Metric — R² on the *raw* target.** R² = 1.0 perfect, 0 = guessing the mean, **negative = worse than the mean**. Because errors are *squared*, a few extreme rows control the score.
- **Constraint** — 5 submissions/day. The live score is on a *public* slice; the final rank uses a hidden *private* slice. This is why we kept both a safe hedge and a high-upside bet. (The high-upside bet is what won.)
- **Goal** — beat the 0.040 baseline. Final private result: **0.07434**.

**If asked "why is the metric the whole story?":** Squared error means one giant row's error can dwarf thousands of normal rows. Whoever handles the giants least-badly wins.

---

## Slide 3 — EDA 1: The one fact that controls everything

**The target is extremely heavy-tailed.** ~96.6% of targets sit in |y| < 200, but the full range is **−41,008 to +69,628**. **84 rows (3.4%)** have |target| > 1,000. Median is −0.77.

**Why it controls everything.** R² squares errors, so the ~84 giants dominate. Predict them okay-ish → the score moves a lot; predict the other 96% perfectly but botch the giants → the score barely moves. The giants are the *hero and the villain*: where the points are, and what destabilizes training.

**If asked "couldn't you just drop the outliers?":** Under this metric the outliers *are* the signal you're scored on. Dropping them optimizes the wrong thing.

---

## Slide 4 — EDA 2: Off-the-chart skew and kurtosis

**Skewness = 13.6, excess kurtosis = 708** (a normal curve is 0 on both). The interquartile range is only ±43 but the full range spans 110,000+. This quantifies why ordinary least-squares (which minimizes squared error) will obsess over the giants. It's structure, not noise.

**If asked "what's kurtosis?":** A measure of tail heaviness — how often extreme values occur. 708 vs. 0 means extremes are vastly more common here than any bell curve.

---

## Slide 5 — EDA 3: Raw features barely correlate with the target

Strongest single feature correlates only ~**0.06** with the target; most are inside the noise band (|r| < 0.04, the ~95% confidence range for "zero" at n=2,500). Max feature-to-feature correlation is 0.063 — features are essentially independent. So plain linear models cap near 0.03, and the signal must come from **curvature** and **interactions**.

---

## Slide 6 — EDA 4: The signal hides in x9

1. **x9 is curved** — median target rises then flattens (diminishing returns); a squared term captures it.
2. **x9 is a multiplier** — x9 × other features beat every raw feature. **x9 × x11 (r = 0.16)** is the standout.
3. **x11 is a latent amplifier** — near-zero alone (r ≈ 0.027), strongest signal once multiplied by x9. It acts as a "direction gate" deciding whether high x9 pushes the target strongly positive or negative.

**If asked "why is an interaction stronger than its parts?":** Because x9's effect on the target *depends on* x11. Neither predicts alone; the product captures the conditional relationship.

---

## Slide 7 — Feature engineering: 15 raw → 28 features

- **x9²** — captures the curve. (Log rejected: x9 goes negative; sqrt-of-abs had ~0 correlation.)
- **9 x9-interactions** — x9 × {x11, x12, x2, x0, x8, x5, x1, x10, x3}.
- **3 x11-interactions** — x11 × {x7, x1, x12}.
- **Inclusion rule |r| > 0.04** — the noise floor at n=2,500.

**Deliberately not done:** all 105 pairwise interactions (tested — added noise, *hurt* the worst fold), x9³+ (overfitting risk), PCA (explained variance is flat, no axis to extract). Later work confirmed richer features never beat these 28 — feature engineering is a closed book here.

**If asked "why not let regularization prune everything?":** It was tried; the ~90 extra noise features lowered CV scores. With limited data, few good features beat many noisy ones.

---

## Slide 8 — Data preparation: a leak-free pipeline

Order: Raw CSV → median imputation → feature engineering (15→28) → standard scaling → **clip target ±2000 (training only)** → `processed/`. Everything is fit on **train only** (using any test statistic would be leakage).

- **Median imputation** — missingness is MCAR and features are independent, so median is robust and model-based imputation is pointless.
- **Standard scaling** — features have wildly different scales (std 2 to ~100 for interactions); Ridge's coefficient penalty needs unit variance to be fair.
- **The core trick: clip the *training* target to ±2,000, grade on raw.** Calms the giants while the model learns without moving the yardstick. ±2,000 was the swept sweet spot (no clip → a CV fold craters to −6.59; ±5,000 → instability returns; ±1,000 → too timid).

**Distinction from the old "winsorizing" mistake:** the old models clipped *and graded in the clipped world*, so they never learned to predict large values and then failed on the raw-scored board. Clip only what's learned; grade on reality.

---

## Slide 9 — Model architecture: sign-coupled two-stage

**The skeleton (shared by all two-stage variants):** a detector probability `p` blends a stable "normal" prediction with an "extreme" prediction:
`pred = (1 − p)·normal + p·extreme`.

**The winning variant's twist — split the giant's prediction into two jobs:**

- The **extreme regressor predicts magnitude only** — a Ridge on the 84 extreme rows, fit on `log(1+|y|)` (the absolute value), so it only ever learns *how big*.
- The **sign is borrowed from the stable backbone** — `sign(normal_prediction)`. Direction was the source of the worst errors, so we don't let the fragile specialist decide it.

**The exact code:**

```python
def two_stage_signcoupled(Xtr, ytr, Xpr, shrink=0.6, cap=10000):
    etr = (np.abs(ytr) > 1000).astype(int)            # flag extreme training rows
    pn  = normal_pred(Xtr, ytr, Xpr)                  # backbone: 0.6·RF + 0.4·Ridge on ±2000-clipped y
    Xe, ye = Xtr[etr == 1], ytr[etr == 1]             # only the ~84 extremes
    mag = Ridge(alpha=10.0).fit(Xe, np.log1p(np.abs(ye))).predict(Xpr)  # log-MAGNITUDE only
    pe  = np.sign(pn) * np.clip(np.expm1(mag), 0, cap)   # magnitude from specialist, SIGN from backbone, cap ±10000
    p   = mk_clf().fit(Xtr, etr).predict_proba(Xpr)[:, 1] * shrink     # detector prob × 0.6
    return (1 - p) * pn + p * pe                       # soft blend
```

**What `s06` means:** `shrink = 0.6`, `cap = 10000`. Higher shrink + cap = bolder, larger predictions (range ±3,300; 147 rows predicted as giants). The `s04` sibling is the same with `shrink = 0.4`. The conservative `submission_two_stage.csv` used `shrink = 0.3` and predicted *sign+magnitude together* — safer, but it won only the public board, not the final.

**Why *soft* routing, not *hard*:** hard routing would send each row entirely to one model; one misroute (a giant prediction on a normal row) is catastrophic under squared error. Soft routing weights by probability so a wrong flag only nudges.

**Why borrow the sign:** the specialist could confidently predict the *wrong direction*, doubling the error. Coupling the sign to the steady backbone removes the largest failure mode (it doesn't remove it entirely — when the backbone's sign is *also* wrong, amplifying magnitude still hurts — which is why one fold still collapses).

**If asked "why log-magnitude?":** Extremes span 41k–70k; logging compresses that range so a linear model can fit it, then `expm1` maps back and `cap` limits it.

---

## Slide 10 — How 5-fold cross-validation actually runs

This is the **evaluation** method (it does not produce the submission). Setup, fixed once:

```python
CV = KFold(n_splits=5, shuffle=True, random_state=42)
```

That shuffles the 2,500 rows (seed 42 → reproducible) into **5 folds of ~500**. Then:

```python
for tr, va in CV.split(X):
    score = r2_score(y[va], predict_fn(X[tr], y[tr], X[va]))
```

Each of the 5 rounds:

1. **Split** into ~2,000 training rows (`tr`) and ~500 held-out rows (`va`).
2. **Refit the *entire* model on `tr` only** — backbone RF+Ridge, the magnitude Ridge (on tr's extremes), and the detector classifier are all trained fresh. Nothing sees `va`. This is what prevents leakage.
3. **Predict `va`** and score R² against its *raw* targets.

You get **5 scores**. For `signc_s06` (re-run live): per-fold `[+0.090, +0.082, −1.817, +0.040, +0.117]` → **median 0.082, mean −0.298, min −1.82**. Four folds are strong; fold 2 collapses because its held-out giants get a confident wrong size/direction with only ~67 training extremes to learn from.

**The lesson on the slide: read the median, not the mean.** The one collapsed fold poisons the mean (−0.30), which would tell you to throw the model away. The median (0.082) reflects the typical fold — and that's what tracked the real board.

### Out-of-fold (OOF)

A refinement used in `model_v4_conservative.ipynb`: stitch each fold's held-out predictions into one array, so **every row has a prediction from a model that never saw it**:

```python
oof = np.zeros(len(y))
for tr, va in folds:
    oof[va] = two_stage(X[tr], y[tr], X[va])
```

That enables the diagnostic on slide 12.

**If asked "why does one fold always collapse?":** A sign/magnitude error on a held-out giant. With ~67 training extremes and near-zero usable signal for magnitude, this tail risk is irreducible.

---

## Slide 11 — Validation: the detector works, the magnitude is the risk

1. **Extremes ARE detectable. Classifier ROC-AUC ≈ 0.82** (live per-fold: `[0.76, 0.86, 0.84, 0.84, 0.82]`). A non-linear RF separates them even though linear/PCA methods couldn't.
2. **Magnitude/sign are NOT reliably predictable** from ~67 extremes — hence the fold-2 collapse.

The chart shows the AUC bar and the per-fold profile with both the **median (0.082)** and **mean (−0.298)** lines drawn, so the gap between them is visible — that gap *is* the whole risk story.

**If asked "isn't a −1.82 fold disqualifying?":** On the mean, yes. But it's a single unlucky draw of unseen giants. The median says the typical outcome is strongly positive, and the private set turned out to be a typical (good) draw.

---

## Slide 12 — The CV trap: why pooled cross-validation lied

Take the OOF predictions, multiply them by a global scale `s`, and score two ways:

- **Pooled OOF R²** — concatenate all folds, compute one R². Dominated by the ~84 giants. It keeps *rewarding bigger predictions* (it wanted to scale up well past ×1).
- **Per-fold median R²** — score each fold separately, take the median. It peaks near ×1 then **craters** as predictions grow.

They disagree, and **the private leaderboard sided with per-fold.** That's the evidence that the pooled/averaged score is actively misleading on outlier-heavy data — a big correct guess shrinks a giant's enormous squared error in the pooled view, but on real unseen data the giants are unpredictable, so betting big is betting on noise.

**If asked "so why submit an aggressive model at all?":** Because the *per-fold median* (the trustworthy stat) was highest for the aggressive sign-coupled config, and we kept the conservative version as the safe hedge. The aggressive bet paid off on the private slice.

---

## Slide 13 — Final output: submission_ts_signc_s06.csv → private R² 0.07434

- **What was submitted:** `two_stage_signcoupled(shrink=0.6, cap=10000)`, refit on all 2,500 training rows and predicted on the test set once. Predictions range **±3,300**, with 147 rows predicted as giants.
- **The result:** public scoring *punished* the bold predictions (bigger predictions scored worse publicly), but the hidden **private** slice behaved like a typical good fold, so the model scored **0.07434** — almost exactly its CV median of 0.082. It was the best entry and the winner.
- **The comparison bar:** baseline 0.040 → v3 blend 0.035 (CV) → conservative two-stage 0.046 (public) → signc_s06 CV-median 0.082 → **signc_s06 private 0.07434**.

**The one-line story:** the median predicted the private score; the mean would have made us discard the winner.

**If asked "did you get lucky?":** Partly — the private slice could have resembled the bad fold. That's why a safe hedge was kept. But the *expected* (median) outcome favored this model, so it wasn't a blind gamble; it was a bet on the statistic that actually tracks reality.

---

## Slide 14 — Key takeaways

1. **Match training to the metric.** Train (in clipped form) on the raw target you're graded on — don't ace a transformed proxy.
2. **Sign-couple the specialist.** Let the extreme model size the giants; borrow their *sign* from the stable backbone to kill the worst (wrong-direction) errors.
3. **Read the per-fold median.** Mean −0.30 said reject; median 0.082 said winner. One collapsed fold poisons the mean on outlier-heavy data.
4. **Public ≠ private.** The bold predictions lost the public board but won the hidden private one (0.07434). Keep the high-upside bet alongside a safe hedge.

**Closing line:** Final private R² **0.07434** — best entry, roughly double the 0.040 baseline.

---

## Glossary (quick reference for Q&A)

- **R²** — 1 = perfect, 0 = guessing the mean, < 0 = worse than the mean.
- **Heavy-tailed / skew / kurtosis** — most values small, a few extreme; skew = lopsidedness, kurtosis = tail heaviness (both 0 for a normal curve).
- **k-fold cross-validation** — split into k=5 chunks, train on 4, score the held-out 5th, rotate; gives 5 scores so you see stability.
- **Mean vs. median of folds** — mean is dragged down by one bad fold; median reflects the typical fold. Here the median predicted the private board.
- **Out-of-fold (OOF)** — predictions for each row made by a model that never trained on it (stitched from the held-out folds).
- **Pooled vs. per-fold scoring** — pooled = concatenate all OOF predictions, one R² (outlier-dominated, misled us); per-fold = score each fold then summarize (tracked reality).
- **Clipping** — forcing values within a range (anything above 2,000 → 2,000).
- **Winsorizing** — percentile-based clipping; the old mistake also *graded* in the clipped world.
- **Soft vs. hard routing** — soft weights both models by probability; hard sends each row to one. Soft is safer because a misroute costs little.
- **Sign-coupling** — the extreme model supplies only magnitude; the sign is taken from the stable backbone.
- **shrink / cap** — `shrink` scales the detector probability down (less aggressive routing); `cap` limits the extreme prediction's magnitude. s06 = shrink 0.6, cap 10000.
- **ROC-AUC** — classifier quality; 0.5 random, 1.0 perfect. Ours ≈ 0.82.
- **signed-log / log-magnitude** — `sign(y)·log(1+|y|)` keeps sign; `log(1+|y|)` keeps only size (used by the sign-coupled model).
- **Public vs. private leaderboard** — live score on a public slice; final rank on a hidden private slice. A model can flip between them — which is exactly what happened here.

---

## Likely questions and crisp answers

**Q: In one sentence, what is the winning model?**
A sign-coupled two-stage model: a classifier flags likely-giant rows, a specialist predicts only their magnitude while the sign is borrowed from a stable backbone, blended by probability and tuned aggressively (shrink 0.6, cap 10000).

**Q: How is it different from the conservative `submission_two_stage.csv`?**
That one predicted sign *and* magnitude for giants and stayed small (shrink 0.3, ±800) — it won the *public* board (0.046). The sign-coupled `s06` predicts magnitude only, borrows the sign, bets bigger (±3,300), and won the *private* board (0.07434).

**Q: Why trust a model whose CV mean is −0.30?**
Because the mean is poisoned by one unlucky fold of unseen giants. The median (0.082) is the statistic that tracks the real board, and the private result (0.074) confirmed it.

**Q: How does the k-fold loop avoid leakage?**
The entire pipeline — backbone, magnitude Ridge, classifier — is refit on the training folds only inside each round; the held-out fold is never seen until scoring.

**Q: What's the single most transferable lesson?**
On data ruled by a few outliers, the averaged/pooled cross-validation score lies. Read the per-fold median, and remember the public score can disagree with the private one — so keep a high-upside bet *and* a hedge.
