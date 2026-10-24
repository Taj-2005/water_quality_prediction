# What I built and why — Shaik Tajuddin

| | |
| :--- | :--- |
| **GitHub** | [@Taj-2005](https://github.com/Taj-2005) |
| **Role** | Team lead · repository, data dictionary, cleaning, EDA |
| **Branch prefix** | `feat/taj-*` |
| **Theory note for the report** | Missing data handling |

## What I built

1. **The repository** (`feat/taj-repo-scaffold`, PR #2) — structure, README, `.gitignore`,
   `requirements.txt`, the branch rule and the fixed seed in `src/config.py`, plus the
   work plan as editable markdown so ownership changes are reviewable instead of locked
   inside a PDF.
2. **`docs/data_dictionary.md`** — all 14 columns profiled on the full 2,827,977 rows:
   units, meanings, ranges, categorical breakdowns, and the data-quality findings below.
3. **`src/clean.py` + `notebooks/01_cleaning.ipynb`** — unit check, detection-limit check,
   impossible-value flags, duplicate removal, upstream-imputation flags. 2,818,006 × 30
   written to `clean.parquet`.
4. **`notebooks/02_eda.ipynb`** — distributions, correlations, composition and coverage,
   ending in nine findings mapped to the owner each one affects.

## Why I built it that way

**Flag, never delete.** Every questionable row is marked and kept: 505 pH readings outside
0–14, 15 negative concentrations, 7,284 temperatures above 40 °C. Deleting them would hide
a data-quality problem the report has to discuss, and it would make a cleaning script the
place where modelling decisions get taken silently. Flags move that decision to Parth's
pipeline, where it belongs and where it is visible.

**No imputation in cleaning.** Imputation must be fitted on the training split only, and
the split does not exist at cleaning time. Doing it here would fit statistics across rows
destined for the test set.

**Column names kept verbatim.** `src/config.py` binds `TARGET_COLUMN = "CCME_Values"`.
Tidier names would have broken Parth's interface and every downstream module — a cosmetic
gain for a real cost.

**Parquet as the shared artifact.** The cleaned table is 559 MiB as CSV, which GitHub
refuses outright above 100 MB, and in LFS it would have consumed most of the free 1 GiB
quota. As zstd parquet it is 45 MB in plain git, loads in 0.3 s instead of 8.5 s, and
preserves the category and boolean dtypes a CSV round trip destroys. Everyone now models
identical bytes rather than trusting that each rerun matched.

## Theory note — missing data handling

The standard taxonomy is MCAR, MAR and MNAR, and the usual Phase 1 task is choosing an
imputer and fitting it on the training split only. **This dataset did not allow that task,
and finding out why was the work.**

The file reports **zero** missing values. That looked like good fortune until I profiled
the modal values: five of the eight measurement columns carry their own **exact median** as
the single most frequent value — dissolved oxygen 10.2 at 31.60 % of rows, nitrate 4.5 at
25.33 %, BOD 2.7 at 24.28 %, pH 7.78 at 12.98 %, temperature 11.46 at 11.02 %. No natural
measurement distribution places that much mass on its own median. It is the signature of a
median fill applied upstream, during a harmonization step we did not run.

Three consequences follow, and they are the substance of this note:

1. **The missingness mechanism is unrecoverable.** Once a gap is filled with a median, the
   fact that it was ever missing is gone. We cannot classify it as MCAR, MAR or MNAR, and
   we cannot undo it. The only honest response is to detect it and mark it, which is what
   `flag_upstream_imputation()` does.
2. **The train-only rule was broken before we received the data.** The upstream fill used
   medians computed over every row, including rows that will become our test set. Parth's
   pipeline can still fit *its* imputer on train only, but that guarantee is now partial,
   and the report must say so rather than claiming a cleanliness we do not have.
3. **The fill is not random — it is an England artefact, and it is confounded with the
   target.** England averages 2.04 suspect measurements per row against ≤0.04 for every
   other country, on 75.3 % of the dataset. Mean `CCME_Values` is 91.31 for fully measured
   rows against 75.89 at three suspect measurements. So fill status carries information
   about both country and target, and a model given these columns can score well by
   detecting which country filled its gaps rather than by learning chemistry. No split
   design fixes this, because the confound follows the sites.

The practical outcome: one boolean per affected column plus `n_imputed_upstream` (0–8), so
Jivit's ablation can measure the cost rather than the team inheriting it invisibly.

A related negative result belongs here. The planned detection-limit rule — one stated
treatment for censored values like `<0.5` — turned out to be unnecessary: 5,508 rows
contain a `<`, and every one is inside an `Area` site name such as
`BURNHAM MARKET STW F/E <=2028M3/D`. All eight measurement columns parse as pure floats.
The upstream pipeline had already resolved censoring too, by means we cannot inspect. I
kept the check as executable code rather than deleting it, so a future data refresh reports
censored values instead of coercing them to `NaN` unnoticed.

## How it connects to the rest of the pipeline

- **To Parth:** `clean.parquet` is what `load_data()` returns. The imputation confound and
  the country-dependent target ceiling both constrain his split design.
- **To Satya:** the target is censored — 35.92 % of rows sit exactly at 100.0 — so metrics
  need reporting split by at-ceiling versus below-ceiling, or a model looks good by
  predicting the boundary.
- **To Himanshu:** skewness reaches 11.98 with kurtosis 222, which is the evidence for his
  log-transform features. Trees are invariant to monotone transforms, so the gain should
  show up on the Phase 2 MLP rather than on Random Forest.
- **To Jivit:** the `__imputed_upstream` flags are built for his ablation.

## What I would do differently

I accepted "zero missing values" at face value for the first pass and only caught the fill
when checking modal values for the data dictionary. A distribution check — modal value
share against the median — should be the *first* thing run on any dataset that claims no
missingness, not a late step. It is now the first check in `01_cleaning.ipynb`.
