# What I built and why — Satya Yadav

| | |
| :--- | :--- |
| **GitHub** | [@ACE-pwd](https://github.com/ACE-pwd) |
| **Role** | IEEE LaTeX report, presentation, metrics, naive baseline |
| **Branch prefix** | `feat/satya-*` |
| **Theory note for the report** | Metrics: what each one measures and why it fits this target |

> **Draft prepared by the team lead from the repository evidence — Satya to verify and
> rewrite in his own words before submission.** This document is individually assessed
> and read alongside commit history, so the reasoning below must be yours. Anything you
> disagree with is wrong by definition; correct it. The *What I would do differently*
> section is deliberately left blank because nobody else can write it for you.

## Files I own

`src/metrics.py`, `notebooks/03_naive_baseline.ipynb`,
`reports/naive_baseline_results.csv`, `docs/viva_prep.md`, the presentation deck
(`presentation/phase1_presentation.pdf`), and the IEEE report content.

## What I built

1. **`src/metrics.py` (312 lines)** — the single scoring implementation every model uses,
   so that no module defines its own metric and comparisons between models are
   meaningful.
   - `regression_metrics` — MAE, RMSE, R², MAPE and median absolute error, as pure
     functions of `(y_true, y_pred)`.
   - `classification_metrics` and `calibration_data` — for the five-class `CCME_WQI`
     scheme, retained for Phase 2 and for calibration analysis even though Phase 1 is
     formulated as regression.
   - `ceiling_split_metrics` — reports errors separately for records at the index ceiling
     and below it.
   - Input handling: `_to_array` and `_validate_lengths` coerce any array-like and fail
     loudly on mismatched lengths, rather than letting NumPy broadcast silently.
2. **`notebooks/03_naive_baseline.ipynb`** — mean and median predictors on the shared
   split, establishing the floor every model must beat.
3. **The presentation deck**, now at v2.
4. **Report sections** and `docs/viva_prep.md`.

## Why I built it that way

**Metrics as pure functions, tested on dummy data.** `regression_metrics` takes only
`y_true` and `y_pred` — no model, no dataframe, no configuration. That makes it testable
without the dataset, which matters because CI never has the 305 MiB raw file. The
alternative, letting each model module compute its own scores, is how two models end up
reported on subtly different definitions.

**MAE *and* RMSE, not one or the other.** They answer different questions on this data.
The measurements are extremely heavy-tailed — ammonia skewness 11.98, kurtosis 222 — so
RMSE is dominated by a small number of large excursions while MAE describes the typical
record. Reporting only one hides the shape of the error.

**A median baseline as the floor, not a mean baseline.** Both are reported. The median
gives the lower MAE (13.605 against 14.900) and the mean gives the lower RMSE (18.656
against 19.492), because each minimises the error it is built for. We quote the median
because MAE is the primary metric. The mean baseline's $R^2 = -0.0006$ is worth
understanding rather than glossing: R² is *defined* against the mean predictor, so ≈0 is
the expected value, not a sign that the problem is unusually hard.

**R² returns `nan`, not 0 or 1, when the target has no variance.** This occurs for the
at-ceiling group, where every true value is exactly 100. Reporting a number there would
be an artefact of a zero denominator; `nan` is the honest answer and forces the reader to
use MAE and RMSE for that group.

**Ceiling-split reporting.** 35.92 % of records sit at exactly 100. A pooled statistic
therefore answers two questions at once — whether a record is fully guideline-compliant,
and how far the rest fall short — and a model can be strong on one and weak on the other.
`ceiling_split_metrics` separates them and adds two diagnostics: mean shortfall for
records that belong on the ceiling, and the rate at which non-ceiling records are
predicted at or above it.

The target is **bounded with a point mass, not censored**. Censoring implies a latent
value beyond the threshold that recording conceals; here a fully compliant record gives
$F_1 = F_2 = F_3 = 0$ and therefore exactly 100, so nothing is hidden above the bound.
That rules out Tobit-style censored regression and makes stratified reporting the correct
remedy.

## Theory note — metrics and what they measure

**MAE** is the mean of $|y - \hat{y}|$: the expected size of a typical error, in index
points. It weights every record equally and is therefore robust to the heavy tails in
this dataset.

**RMSE** is $\sqrt{\mathbb{E}[(y-\hat{y})^2]}$. Squaring before averaging means a single
record wrong by 10 points contributes as much as a hundred wrong by 1. On a skewed target
this makes RMSE a statement about worst-case behaviour more than typical behaviour.
RMSE ≥ MAE always, and the ratio between them indicates how much of the error is
concentrated in a few records.

**R²** is $1 - \mathrm{SS_{res}}/\mathrm{SS_{tot}}$: the fraction of target variance the
model explains, relative to predicting the mean. Its reference point is what makes it
useful — and also what makes it easy to misread. R² = 0 is the mean predictor, not a
useless model in some absolute sense, and negative R² simply means worse than that
reference. R² is also not comparable across our availability regimes without care,
because its denominator is the target variance of whichever subset is being scored.

**Bootstrap confidence intervals** turn each point estimate into an interval by
resampling the test set with replacement. They answer "would this ordering survive a
different sample of the same size?" — and for Random Forest versus XGBoost the intervals
do not overlap on any metric, so the ordering is not a sampling artefact. They say
nothing about whether the comparison is *fair*, which is a separate issue given the
unequal training budgets.

**Why regression rather than classification.** The five CCME classes discard information
exactly where it matters: a record moving from 79.9 to 80.1 crosses a class boundary and
is penalised identically to one moving from 50 to 95 within a class. The continuous
formulation also avoids a severe imbalance problem, since the worst-quality class is
under 5 % of records.

## How it connects to the rest of the pipeline

- **From Taj:** the cleaned table, and the finding that 35.92 % of the target sits at the
  ceiling — which is what prompted `ceiling_split_metrics`.
- **From Parth:** the shared split and the `evaluate()` contract in `src/evaluate_api.py`
  that my functions sit behind.
- **To Himanshu and Jivit:** both model modules score through `regression_metrics`, so
  their reported numbers are directly comparable.
- **To the report:** `results/metrics.csv` and `results/bootstrap_ci_results.csv` become
  Tables I and IV, generated by `report/make_tables.py`.

## What I would do differently

_To be written by Satya._
