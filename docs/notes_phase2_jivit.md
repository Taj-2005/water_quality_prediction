# docs/notes_phase2_jivit.md — What I Built and Why
**Owner:** Jivit Rana  
**Phase 2 task:** Hybrid approach and hybrid ablation  
**Branch:** `feat/jivit-hybrid`  
**Main files:** `src/dl/hybrid.py`, `notebooks/15_hybrid.ipynb`, `reports/hybrid_results.csv`, `reports/hybrid_ablation.csv`

---

## What I Built

### The residual hybrid (`src/dl/hybrid.py`)

A **residual XGBoost + MLP hybrid** that corrects XGBoost's predictions with a
neural network trained on its errors.

```
ŷ_hybrid = ŷ_xgb  +  ŷ_mlp_residual
```

**Key components:**

1. **Out-of-fold XGBoost predictions** — the most important design choice.  
   XGBoost predictions used to compute training residuals are produced by
   K-fold cross-validation (default K = 5).  For every training row, the
   prediction comes from a XGBoost fold that *never saw that row*, so the
   residual the MLP learns from is genuine, not artificially small.  
   This rule is documented at the top of `hybrid.py` with the label
   **CRITICAL ANTI-LEAKAGE RULE** and is enforced structurally: the
   function `_oof_xgb_predictions` only receives `X_train` and `y_train`.

2. **MLP residual corrector** — trained on `(X_train, train_residuals)`.  
   Until Himanshu's `src/dl/model.py` and Parth's `src/dl/train.py` are
   merged, a lightweight scikit-learn `MLPRegressor` with layers (256, 128, 64)
   fills in.  The import logic in `_try_import_mlp()` replaces the stub
   transparently once the PyTorch MLP is available, with no changes to the
   ablation or evaluation code.

3. **Standard evaluation** — all conditions are scored with the shared
   `evaluate()` from `src/evaluate_api.py`, returning the same MAE / RMSE / R²
   that every other member uses.

4. **Ablation table** (`run_ablation`) — four conditions:
   - XGBoost alone (Phase 1 model, re-run with the same seed)
   - MLP alone (learns target directly, not residuals)
   - Plain average (simple 50/50 blend — the bar a "real" hybrid must beat)
   - Residual hybrid (Jivit's design)

   Results are reported as mean ± std over N seeds (default 5), logged to
   `reports/hybrid_ablation.csv` (aggregated) and
   `reports/hybrid_ablation_raw.csv` (every seed).

---

## Why I Chose the Residual Hybrid

**XGBoost's strength and weakness**  
XGBoost captures the coarse structure of the CCME formula almost perfectly — in
Regime 1 (all 8 chemical inputs available), the R² exceeds 0.99 because the
formula is a nearly-deterministic function of those inputs.  In Regime 2 (no
chemical inputs, metadata only), the residual is much larger, which is exactly
where a correction network can help.

**Why not stacking or embedding?**  
- *Stacking* (XGBoost prediction as MLP input feature) risks the MLP learning
  to trust a signal that is only accurate because XGBoost saw the training row
  — OOF is still required, and the MLP has a harder optimization problem because
  the prediction is entangled with the features.
- *Embedding* (MLP hidden layer as XGBoost features) requires a fully trained
  MLP first, adding a dependency on Himanshu's architecture tuning before
  anything can run.

The residual approach is clean:
- The MLP has a single, narrow task.
- The anti-leakage rule is structurally easy to verify.
- It degrades gracefully: if the MLP adds nothing, the residual predictions
  cluster near zero and the hybrid converges to XGBoost alone.

**Expected outcome**  
In Regime 1 the XGBoost already explains ~99 % of the variance, so the
MLP's marginal gain will be small (and that is an honest finding).  
In Regime 2 there is genuine room for the MLP to learn non-linear
interactions between metadata features and the residual.

---

## Leakage Controls

| Risk | Mitigation |
|------|-----------|
| MLP sees test rows during training | MLP is fitted only on `X_train` and `train_residuals`; validation and test rows are only passed to `predict()` |
| XGBoost residuals on train row it was trained on | Fixed by OOF: each training row's XGBoost prediction comes from a fold that excluded it |
| Scaler fit on val/test | `_fit_numeric_scaler` is called once on `X_train_arr`; val and test use `scaler.transform()` |
| Hyperparameter selection on test | All tuning uses the validation set only; test is touched once for the final number |

---

## How to Reproduce

```bash
# Single run, quick subsample check
.venv/bin/python -m src.dl.hybrid --subsample 10000

# Full run, Regime 1
.venv/bin/python -m src.dl.hybrid

# Full 5-seed ablation table
.venv/bin/python -m src.dl.hybrid --ablation --seeds 5

# Regime 2 (metadata only)
.venv/bin/python -m src.dl.hybrid --regime 2
```

Output files:
- `reports/hybrid_results.csv` — single-run results (append-only)
- `reports/hybrid_ablation.csv` — mean ± std over seeds
- `reports/hybrid_ablation_raw.csv` — every seed's raw metrics

---

## Theory Note: Bias–Variance Trade-off, Residual Learning, and Stacking

### Bias–variance decomposition
For a regression estimator f̂, the expected squared error on a test point decomposes as:

$$\mathbb{E}[(y - \hat{f}(x))^2] = \underbrace{\text{Bias}^2[\hat{f}(x)]}_{\text{systematic error}} + \underbrace{\text{Var}[\hat{f}(x)]}_{\text{sensitivity to training data}} + \sigma^2_\varepsilon$$

XGBoost, as an ensemble of deep trees, has **low bias** (it can fit complex
functions) but non-trivial **variance** (it may overfit region-specific splits).
An MLP with regularization (dropout, early stopping) has intermediate bias and
lower variance on smooth functions.  Combining them reduces both when their errors
are uncorrelated — the prerequisite for any ensemble gain.

### Residual learning
If f̂_1(x) is XGBoost's prediction, then the residual r = y − f̂_1(x)
contains everything XGBoost could not explain.  A second model f̂_2 trained
directly on (x, r) learns what the first model missed.  The combined prediction

$$\hat{f}_{hybrid}(x) = \hat{f}_1(x) + \hat{f}_2(x)$$

is equivalent to gradient boosting with one manual boosting step — except the
"weak learner" here is a neural network rather than a shallow tree.  This
connection is why XGBoost + MLP residual hybrids are more principled than naive
averaging: each component has a defined role.

### Stacking (meta-learning)
Stacking trains a meta-learner (here, the MLP) on the *predictions* of base
models rather than on the raw features.  OOF predictions are essential to avoid
target leakage: a base model that memorises training labels will produce
near-zero residuals for training rows it saw, so the meta-learner would learn
to ignore those rows and would fail on new data.  The K-fold OOF procedure used
here generates honest training residuals for the meta-learner at the cost of
training K + 1 XGBoost models instead of one.

---

*Last updated: {{ date }}*
