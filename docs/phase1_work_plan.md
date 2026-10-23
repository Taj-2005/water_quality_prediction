# Water Quality Prediction: Phase 1 Work Plan

Revised 23 October 2026 · Team lead: Shaik Tajuddin · Original plan by Parthraj Singh Bhati

> The editable source of this plan is `docs/phase1_work_plan.md` in the project
> repository, and the PDF is generated from it. Edit the markdown and regenerate rather
> than editing the PDF, so the two never disagree.

## Overview

Phase 1 delivers the complete machine learning half of Project 18 (Water Quality
Prediction): clean data, a leakage-safe split, Random Forest and XGBoost models, and a
graded report, deck and repo, split across five people.

Phase 1 is graded on nine items: literature review, dataset quality and EDA, feature
engineering, theoretical rigor, model application, GitHub and code quality, presentation,
viva and the IEEE LaTeX report. The viva is individual, so everyone must be able to
explain the whole pipeline, not only their own part.

- **Dataset:** the one provided in the project PDF (finalised).
- **Target:** confirm the exact target column first (a continuous index or a defined class
  scheme) and write it down. If the target is an index computed from the input columns,
  that is leakage: drop the contributing inputs or choose a different target, and state it
  in the report.
- **Models:** naive baseline, Random Forest, XGBoost. The MLP comes in Phase 2.
- **Fairness rules:** one shared split with a fixed seed, preprocessing fitted on train
  only, tuning on validation only, the test set touched once at the end.

## Roles at a glance

Taj sets up the repository and leads the team, Parth defines the target and the shared
interface, Taj then feeds the data, Himanshu and Jivit run the two models side by side,
and Satya owns the report and deck.

| Member | Role | Main files | Others need it |
| :--- | :--- | :--- | :--- |
| Taj *(team lead)* | Repository, branch rules, seed, PR review, data dictionary, cleaning, EDA | README.md, requirements.txt, .gitignore, src/config.py, docs/data_dictionary.md, src/clean.py, 01_cleaning.ipynb, 02_eda.ipynb | Day 1, then day 2 to 4 |
| Parth | Target, split, pipeline, final evaluation | docs/target_definition.md, src/split.py, src/preprocess.py, src/evaluate_api.py, 06_evaluation.ipynb, 07_shap.ipynb | Day 1 to 2, then again after cleaning |
| Himanshu | Literature review, feature engineering, Random Forest | refs.bib, src/features.py, src/models_rf.py, 04_random_forest.ipynb | After split and pipeline |
| Jivit | XGBoost, ablation | src/models_xgb.py, 05_xgboost.ipynb, reports/xgb_results.csv | After split and pipeline |
| Satya | IEEE LaTeX report, presentation, metrics, naive baseline | Overleaf project, deck, src/metrics.py, 03_naive_baseline.ipynb, docs/viva_prep.md | Metrics early, report at the end |

## Taj: repository, team lead, data dictionary, cleaning, EDA

Taj owns the repository and leads the team, so the shared foundation exists on day 1 and
nobody is blocked. He then turns the raw dataset into a clean, documented table that every
other step depends on.

### Step 1: repository and project standards (day 1)

- Create the GitHub repository and add all five members. Add README.md,
  requirements.txt, .gitignore and the folders `data/`, `notebooks/`, `src/`, `reports/`
  and `docs/`.
- Set the branch rule: one branch per person (for example `feat/taj-cleaning`), pull
  requests into `main`, **Taj reviews**. Nobody pushes straight to `main`.
- Fix the random seed in one place (`src/config.py`) and have every module import it, so
  results are reproducible across members.
- Keep README.md current: the exact commands to reproduce every result, the roster, and
  the ownership table.
- Act as the point of contact for the faculty.

### Step 2: data dictionary and cleaning (day 2 to 4)

- `docs/data_dictionary.md`: every column with its unit, meaning, source and license.
- `src/clean.py` and `notebooks/01_cleaning.ipynb`:
  - make units consistent
  - handle detection-limit values such as "<0.5" with one stated rule
  - flag impossible values (for example pH outside 0 to 14, negative turbidity) instead of
    silently deleting them
  - remove exact duplicates
- Save the output to `data/processed/clean.csv`. Do not impute here; imputation belongs in
  Parth's pipeline and must be fitted on train only.
- Log every cleaning decision with its reason in the notebook, because examiners will ask.

### Step 3: EDA (in parallel with Parth's pipeline)

- `notebooks/02_eda.ipynb`: distributions, missingness by column and by site, correlation
  heatmap, target skew or class balance, and outliers.
- End the notebook with a short list of findings that change modelling decisions (for
  example skewed features that may need a log transform).
- Write the report notes for the Dataset and EDA sections and the theory note on missing
  data handling.

**Done when:** the repository is live with all five members and the branch rule in place,
`clean.csv` is merged into `main`, Parth's pipeline runs on it, and the EDA notebook runs
top to bottom without errors.

## Parth: target, split, pipeline, evaluation

Parth defines the target and the shared interface on day 1 and 2, because nobody else can
code against a moving target. He returns once Taj pushes clean data, and again at the end
for evaluation.

### Step 1: target and interface (day 1 to 2)

- Confirm the target column in the provided dataset and write `docs/target_definition.md`:
  the target, the standard or index definition used, and the leakage risks.
- Push `src/evaluate_api.py` as a stub with `load_data()`, `get_splits()` and
  `evaluate(y_true, y_pred)` so everyone codes against the same interface.

### Step 2: split and pipeline (after Taj's clean data lands)

- `src/split.py`: split by site or time if the data has those columns, otherwise a seeded
  random split. Use the shared seed and save the train, validation and test indices to disk.
- `src/preprocess.py`: an sklearn pipeline for imputation, scaling and missingness flags,
  fitted on the training set only.
- Write the theory note on split design and leakage for the report.

### Step 3: final evaluation (after models are saved)

- `06_evaluation.ipynb`: one results table for naive, Random Forest and XGBoost, residual
  plots (or a confusion matrix for classes), and error by site or time period.
- `07_shap.ipynb`: SHAP and feature importance for the best model.
- Write the theory note on SHAP.

**Done when:** everyone can run `get_splits()` and get identical data, and the final
results table has all three models on the same test set.

## Himanshu: literature review, feature engineering, Random Forest

Himanshu can start the literature review on day 1, then builds features and the Random
Forest once Parth's split and pipeline are merged.

### Step 1: literature review (start day 1)

- Read and summarise 8 to 12 papers on water quality prediction with machine learning,
  water quality index definitions, and Random Forest and gradient boosting on tabular data.
- Keep one `refs.bib` file in the repo and tell Satya when it is updated, since Satya cites
  from it in LaTeX.
- Write the Literature Review notes: what others did, which metrics they used, and the gap
  this project fills.

### Step 2: feature engineering (after the split and pipeline are pushed)

- `src/features.py`: derived features such as ratios between chemical parameters, log
  transforms for skewed columns, and time features if a date exists. Any aggregate (for
  example per-site averages) must be computed from training rows only.
- Run a leakage check: confirm no feature is a formula of the target. Document the result.
- Hand Jivit the feature list so the ablation uses the same features.

### Step 3: Random Forest

- `src/models_rf.py` and `notebooks/04_random_forest.ipynb`: train a Random Forest with
  hyperparameter search (number of trees, depth, minimum leaf size, features per split)
  using the validation set only.
- Save the best model and `reports/rf_results.csv` using the shared `evaluate()` function.
- Write the theory note on Random Forest: bagging, variance reduction and out-of-bag error.

**Done when:** the literature review and `refs.bib` are merged, and `rf_results.csv` exists
with metrics on the shared validation and test split.

## Jivit: XGBoost and ablation

Jivit owns the XGBoost model and the experiments that justify the feature engineering. He
can write the training code early against Parth's interface stub and run it for real once
the split and pipeline land.

### Step 1: prepare (while waiting for the pipeline)

- Read the XGBoost paper and draft the training function against `evaluate_api.py`, using
  a small random subset as a stand-in.

### Step 2: XGBoost (after the split and pipeline are pushed)

- `src/models_xgb.py` and `notebooks/05_xgboost.ipynb`: train XGBoost with Optuna or
  randomized search on the validation set only. Use early stopping, and tune learning rate,
  depth, number of trees, subsample and regularization.
- Save the best model and `reports/xgb_results.csv` using the shared `evaluate()` function.
- Keep the training code in the same function format as Himanshu's, so the Phase 2 MLP can
  plug into the same evaluation.

### Step 3: ablation

- Compare XGBoost with and without Himanshu's engineered features on the identical split.
- Compare Random Forest against XGBoost on the same split and report the difference with a
  short explanation of why one wins.
- Write the theory note on gradient boosting: loss function, additive trees, shrinkage and
  regularization.

**Done when:** `xgb_results.csv` is merged, and the ablation table shows results with and
without engineered features.

## Satya: LaTeX report, presentation, metrics, naive baseline

Satya owns every written and presented deliverable, plus the metrics code and the naive
baseline, so his GitHub contribution is real code and he can explain it in the viva. Most
of his early work needs nobody else.

### Step 1: metrics and baseline (start day 1)

- `src/metrics.py`: MAE, RMSE and R squared for a continuous target, or macro F1 and a
  calibration curve for classes. Write them as functions of `y_true`, `y_pred` and test
  them on dummy data.
- `notebooks/03_naive_baseline.ipynb`: a mean or median predictor (or majority class) on
  the shared split. This is the floor every model must beat.
- Write the theory note on the metrics: what each one measures and why it fits this target.

### Step 2: report setup (start day 1)

- Create the IEEE LaTeX project on Overleaf with one `.tex` file per section, a shared
  `refs.bib` and a `figures/` folder.
- Agree a deadline with each member for their section text. Members send markdown or text,
  or push a `.tex` file directly.

### Step 3: assemble (as sections arrive)

- Turn each member's notes into report sections: Introduction, Literature Review, Dataset
  and EDA, Methodology, Experiments and Results, Limitations, Conclusion.
- Add figures and tables from the notebooks.
- Fix references, formatting and numbering, then export the final PDF.
- Build the presentation deck. Each member presents their own slides, so Satya keeps the
  style consistent and sets the order.
- Maintain `docs/viva_prep.md` with likely questions for every part of the project.

**Done when:** the PDF compiles without errors, every cited paper is in `refs.bib`, and the
deck covers all five members' parts.

## Order of work and dependencies

Taj's repository and Parth's target come first, then Taj's clean data, because everyone
else waits on them. Himanshu, Jivit and Satya then work side by side, Parth evaluates the
saved models, and Satya assembles the report and deck from everyone's sections.

```
Taj: repo + branch rules + seed
   |
   +-> Parth: target definition + shared interface
            |
            +-> Taj: data dictionary + cleaning
                     |
                     +-> Parth: split + preprocessing pipeline
                              |
                              +-> Himanshu: features + Random Forest
                              +-> Jivit: XGBoost + ablation
                              +-> Satya: metrics + naive baseline  (can start day 1)
                                       |
                                       +-> Parth: final evaluation + SHAP
                                                |
                                                +-> Satya: IEEE report + deck
```

## GitHub workflow and viva prep

Every member pushes their own work through a pull request, and every member must be able to
explain a teammate's notebook, because the viva is individual.

- One branch per person, named `feat/<name>-<task>`. Open a pull request into `main` and
  wait for **Taj's review** (team lead) before merging. Never push straight to `main`.
- Commit small and often with clear messages, so each member's contribution is visible in
  the history.
- Fixed random seed everywhere, and a README.md with the exact commands to reproduce every
  result.
- Each member adds a one-page `docs/notes_<name>.md` titled "what I built and why".
- Hold at least two mock vivas where each person explains another member's part and answers
  questions on it.
- The evaluation dates are not fixed yet and fall after the mid-semester exams, so finish
  the Phase 1 code well before mid-sems and use the remaining time for the report, deck and
  viva practice.

## Phase 2 preview

Phase 2 adds the deep learning model on the same split and preprocessing, then compares it
fairly against the Phase 1 tree models.

- Deep learning literature review: MLPs on tabular data, regularization, uncertainty.
- A regularized MLP in PyTorch (dropout, weight decay, early stopping, batch norm), with
  the output head matched to the target.
- Architecture justification: why this depth, width and head suit this problem.
- Ablations on depth, width, regularization, and with or without engineered features.
- Optional hybrid for extra marks, only if it can be justified, for example an MLP trained
  on XGBoost residuals, compared fairly against each model alone.
- Technical validation: calibration, uncertainty, and leakage checks.
- Deep learning theory section, a second IEEE report, a new presentation and another round
  of viva practice.

## Revision history

| Date | Change |
| :--- | :--- |
| 22 Oct 2026 | Original plan (Parthraj Singh Bhati) |
| 23 Oct 2026 | Repository ownership moved from Parth to Taj; Taj named team lead and PR reviewer; Parth's section narrowed to target, split, pipeline and evaluation; seed centralised in src/config.py; dependency order updated |
