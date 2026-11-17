# Water Quality Prediction

Predicting a documented water quality target from physical and chemical measurements, and
comparing interpretable tree ensembles against a neural model on an identical,
leakage-safe split.

**Course project — CSAI (AML / DL), Project 18.** Delivered in two graded phases:

| Phase | Scope | Models | Status |
| :--- | :--- | :--- | :--- |
| **Phase 1** | Classical machine learning | Naive baseline, Random Forest, XGBoost | **Completed** |
| **Phase 2** (planned) | Deep learning + optional hybrid | Regularized MLP (PyTorch), optional MLP-on-residuals hybrid | **Next** |

---

## Current Status

**Phase 1 (Classical Machine Learning) is COMPLETE.** All models, benchmarks, evaluations, reports, presentations, and viva preparation are finalized:

- **Target Confirmed**: `CCME_Values` (Continuous regression target ranging 31.304–100.0, documented in [docs/target_definition.md](docs/target_definition.md)).
- **Cleaned Dataset Committed**: 2,818,006 rows × 30 columns committed to `data/processed/clean.parquet` with full column-by-column profiling in [docs/data_dictionary.md](docs/data_dictionary.md).
- **Leakage-Safe Spatial Split**: 70/15/15 area-grouped split across 61,689 monitoring stations (`Area`), stratified by country, persisted at `data/splits/splits.parquet` ([src/split.py](src/split.py)). Strictly zero station overlap across train, val, and test partitions ($A_{\text{train}} \cap A_{\text{test}} = \emptyset$).
- **Models Evaluated**: Naive Median Baseline, Random Forest (100k subsample, engineered features), and XGBoost (full 1.98M training set, engineered features).
- **Central Scientific Result (Partial-Measurement Ladder)**: Across 4 measurement regimes (8/8, 7/8, 6/8, 4/8), demonstrated that the all-measurements (8/8) regime is mathematical formula emulation ($R^2 = 0.9997$); true environmental prediction occurs when parameters are missing ($R^2 = 0.834$ at 7/8; $R^2 = 0.573$ at 4/8).
- **Authoritative Results**: Generated in a single pass via [src/run_evaluation.py](src/run_evaluation.py) and committed in [results/](results/).
- **Automated Verification**: 135 unit tests in [tests/](tests/) verifying data integrity, spatial disjointness, preprocessing, and model pipelines with zero synthetic failures.
- **Formal Academic Deliverables**:
  - **IEEE Conference Report**: Full 8-page paper compiled at [report/water_quality_prediction_phase1_report.pdf](report/water_quality_prediction_phase1_report.pdf).
  - **Presentation Deck**: 7-slide presentation committed at [presentation/phase1_presentation.pdf](presentation/phase1_presentation.pdf).
  - **Viva Preparation**: 77 defense questions across 9 areas in [docs/viva_prep.md](docs/viva_prep.md) and rehearsal schedule in [docs/mock_viva_plan.md](docs/mock_viva_plan.md).
- **Next Phase**: **Phase 2 (Deep Learning / PyTorch MLP)** on the identical split, preprocessing, and evaluation contracts.

---

## 1. Problem statement

Predict the continuous water quality index (`CCME_Values`) from routine sensor and
lab measurements. The team commits to **one** target definition and does not mix
incompatible water quality standards.

- **Target Column:** `CCME_Values` (see [docs/target_definition.md](docs/target_definition.md))
- **Problem Type:** **Regression** (continuous CCME Water Quality Index ranging from 31.304 to 100.0)
- **Gap being addressed:** tree models are strong on tabular data but may not capture
  smooth multi-sensor interactions or produce calibrated uncertainty as efficiently.
  Phase 2 tests whether a regularized MLP closes that gap on the same split.
- **Deliverable:** a comparison of Random Forest / XGBoost (Phase 1) and an MLP
  (Phase 2), with an explicit target definition, missing-data treatment, and feature
  importance analysis.

### Target definition and leakage

> Full analysis and mathematical formula in [docs/target_definition.md](docs/target_definition.md).

- **`CCME_WQI`** is excluded as direct label leakage (post-hoc categorical binning of the target).
- **The 8 chemical parameters** contributing directly to the CCME calculation formula are documented for pipeline handling and ablation.

---

## 2. Dataset

- **Source:** Harmonized global surface-water monitoring dataset with CCME Water Quality Index values across 5 countries (England, USA, Ireland, China, Canada), totaling 2,827,977 rows × 14 columns.
- **Location:** `data/raw/Combined_dataset.csv`
- **Documentation:** every column — unit, meaning, source, license — is recorded in
  [docs/data_dictionary.md](docs/data_dictionary.md).
- **Raw data is never edited in place.** `data/raw/Combined_dataset.csv` (305 MiB) is
  tracked with **Git LFS**, so `git lfs install` is required before cloning or you will
  receive a pointer file instead of the data.
- **The cleaned table is committed** as `data/processed/clean.parquet` (45 MB, plain
  git, no LFS), so every member models identical bytes rather than relying on each
  person's rerun. It loads in 0.3 s against the CSV's 8.5 s and preserves the category
  and boolean dtypes a CSV round trip loses.
- **`clean.csv` is not committed** — 559 MiB exceeds GitHub's 100 MB plain-git limit,
  and via LFS it would consume most of the free 1 GiB quota. Generate it locally when
  you need it: see [Generating `clean.csv`](#generating-cleancsv). `data/splits/` is
  likewise git-ignored and regenerated.

### Data handling rules

| Stage | Rule |
| :--- | :--- |
| Units | verified consistent per column (mg/l, pH units, °C); no mixed-unit rows found |
| Detection limits | none present — verified in [docs/data_dictionary.md](docs/data_dictionary.md); the upstream harmonization already resolved censored values |
| Upstream imputation | 5 of 8 measurement columns carry their exact median at 11–32 % of rows; flagged per column rather than silently inherited |
| Impossible values | **flagged**, not silently deleted (pH outside 0–14, negative concentrations, temperature above 40 °C) |
| Duplicates | 9,971 exact duplicate rows (0.35 %) removed |
| Imputation | **not** done during cleaning — it lives in the preprocessing pipeline and is fitted on **train only** |

---

## 3. Methodology

### Experimental fairness rules

These are non-negotiable and apply to every model in both phases:

- **One shared split**, created once, with a **fixed random seed**; train/validation/test
  indices are saved to disk so every member trains on byte-identical data.
- Split **by site or by time** when the metadata permits, otherwise a seeded random split.
- All preprocessing (imputation, scaling, missingness flags) is **fitted on the training
  set only**.
- All hyperparameter tuning happens on the **validation set only**.
- The **test set is touched exactly once**, at the end, for the final results table.
- Any aggregate feature (e.g. per-site averages) is computed from **training rows only**.

### Models

1. **Naive baseline** — mean/median predictor (or majority class). The floor every model must beat.
2. **Random Forest** — hyperparameter search over number of trees, depth, min leaf size, features per split.
3. **XGBoost** — Optuna or randomized search, early stopping; tuning learning rate, depth, n_estimators, subsample, regularization.

Both tree models expose the **same training-function signature**, so the Phase 2 MLP
plugs into the identical evaluation path.

### Metrics

| Target type | Metrics |
| :--- | :--- |
| Continuous index | MAE, RMSE, R² |
| Defined class scheme | Macro F1 + calibration curve |

Implemented in `src/metrics.py` as pure functions of `(y_true, y_pred)` and unit-tested
on dummy data.

### Interpretation

SHAP values and feature importance for the best model, plus error broken down by site or
time period, so the model's behaviour is explainable in the viva and not just scored.

---

## 4. Repository structure

```
water_quality_prediction/
├── data/
│   ├── raw/                     # Combined_dataset.csv, 305 MiB, Git LFS
│   ├── processed/               # clean.parquet (committed); clean.csv generated locally
│   └── splits/                  # persisted train/val/test indices (git-ignored)
├── notebooks/
│   ├── 01_cleaning.ipynb        # cleaning decisions, each with its reason
│   ├── 02_eda.ipynb             # distributions, imputation finding, nine conclusions
│   ├── 03_naive_baseline.ipynb  # mean and median floors
│   ├── 04_random_forest.ipynb   # RF training + hyperparameter search
│   ├── 05_xgboost.ipynb         # XGBoost tuning + ablation
│   └── 06_evaluation.ipynb      # final comparison, residuals, SHAP
├── src/
│   ├── config.py                # single seed and all paths
│   ├── evaluate_api.py          # shared contract: load_data(), get_splits(), evaluate()
│   ├── clean.py                 # raw -> clean.parquet, flags, no imputation
│   ├── split.py                 # Area-grouped 70/15/15, stratified by country
│   ├── preprocess.py            # train-only pipeline + partial-measurement masking
│   ├── features.py              # log transforms, ratios, leakage-checked
│   ├── metrics.py               # MAE / RMSE / R2 + ceiling-split reporting
│   ├── models_rf.py             # Random Forest
│   ├── models_xgb.py            # XGBoost
│   ├── evaluation.py            # shared evaluation helpers
│   └── run_evaluation.py        # one pass producing everything in results/
├── results/                     # AUTHORITATIVE - cite these. See results/README.md
│   ├── metrics.csv              # headline comparison of the three models
│   ├── partial_measurement_results.csv   # the four availability regimes
│   ├── ablation_results.csv     # engineered features, imputation flags
│   ├── bootstrap_ci_results.csv # 95% confidence intervals
│   ├── country_results.csv      # test error by country (China has 0 test records)
│   ├── country_insample_china.csv
│   └── shap_importance.csv      # feature attribution
├── reports/                     # figures + per-module training logs. NOT for citation
│   ├── figures/                 # every figure used by the report and deck
│   └── *.csv                    # training-time logs, see reports/README.md
├── report/                      # IEEE LaTeX source; make_tables.py generates tables
│   └── water_quality_prediction_phase1_report.pdf
├── presentation/
│   └── phase1_presentation.pdf  # Phase 1 deck
├── docs/
│   ├── target_definition.md     # target, CCME formula, leakage analysis
│   ├── data_dictionary.md       # every column + the upstream-imputation finding
│   ├── literature_review.md     # related work and the experimental regimes
│   ├── viva_prep.md             # 77 questions across nine areas
│   ├── mock_viva_plan.md        # two-round rehearsal agenda
│   ├── phase1_work_plan.md      # editable source of the work plan PDF
│   ├── refs.bib                 # single source of truth for citations
│   └── notes_<name>.md          # one page per member: "what I built and why"
├── tests/                       # 135 tests; run without the dataset
├── .github/workflows/ci.yml     # lint, format, tests, notebook hygiene
├── pyproject.toml               # ruff + pytest configuration
├── requirements.txt
└── README.md
```

### The shared interface

Everyone codes against `src/evaluate_api.py`, never against their own ad-hoc loader:

```python
from src.evaluate_api import load_data, get_splits, evaluate

df = load_data()                                  # cleaned, documented table
X_train, X_val, X_test, y_train, y_val, y_test = get_splits()   # identical for everyone
metrics = evaluate(y_true, y_pred)                # identical metric definitions
```

If `get_splits()` returns different data for two members, that is a bug to fix before
any result is reported.

---

## 5. Setup

Requires Python 3.10+.

```bash
git clone https://github.com/Taj-2005/water_quality_prediction.git
cd water_quality_prediction

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Notebook kernel

Register the virtual environment as a Jupyter kernel, once per machine:

```bash
python -m ipykernel install --user --name water-quality \
    --display-name "Python (water_quality .venv)"
```

Then select **Python (water_quality .venv)** as the kernel in every notebook. A kernel
pointing at a global interpreter is the usual cause of:

```
ImportError: Unable to find a usable engine; tried using: 'pyarrow', 'fastparquet'.
```

The packages are in `requirements.txt`, but they are installed into the venv — not into
whichever interpreter your notebook happens to be running. `load_data()` falls back to
`clean.csv` when no parquet engine is present, so the notebook still works; it is just
slower and loses the category and boolean dtypes.

### Raw data

The raw dataset is tracked with Git LFS, so a fresh clone needs:

```bash
git lfs install && git lfs pull
```

---

## 6. Reproducing results

Run in this order — each step depends on the previous one.

`data/processed/clean.parquet` is **committed**, so a fresh clone can skip step 1 and go
straight to modelling. Regenerate it only when the cleaning logic changes.

```bash
# 1. raw -> cleaned, documented table (no imputation here)
#    see "Generating clean.csv" below if you need the CSV as well
python -m src.clean --no-csv

# 2. create the shared split once; writes indices to data/splits/
python -m src.split

# 3. train individual baseline and tree models (optional)
python -m src.models_rf        # -> reports/rf_results.csv
python -m src.models_xgb       # -> reports/xgb_results.csv

# 4. Master unified evaluation — evaluates all models across all regimes (8/8, 7/8, 6/8, 4/8),
#    runs ablations, computes bootstrap CIs, decomposes country errors, extracts SHAP,
#    and writes all authoritative CSVs to results/ and figures to reports/figures/
python -m src.run_evaluation

# 5. Rebuild LaTeX tables for the IEEE report directly from results/*.csv
python report/make_tables.py

# 6. Run unit test suite (135 tests across 7 modules, runs without dataset)
pytest tests/
```

### Generating `clean.csv`

`clean.csv` is **git-ignored and never committed** — at 559 MiB it exceeds GitHub's
100 MB plain-git limit, and tracking it in LFS would consume most of the free 1 GiB
quota, blocking pushes for the whole team after a single regeneration.

Most work does not need it: `load_data()` reads the committed `clean.parquet`, which
holds the identical 2,818,006 × 30 table. Generate the CSV only if you specifically need
a plain-text copy — to open it in Excel, to diff it, or to hand it to a tool that cannot
read parquet.

```bash
# both formats: clean.parquet (45 MB) and clean.csv (559 MiB)
python -m src.clean

# parquet only - the default for everyday work
python -m src.clean --no-csv
```

The pipeline is deterministic — a fixed cleaning rule set and no sampling — so every
member's regenerated file is byte-identical. Measured on the full 2,827,977 rows: about
**32 seconds** and **1.4 GiB** of peak memory for both formats.

| Command | Writes | Committed |
| :--- | :--- | :--- |
| `python -m src.clean` | `clean.parquet` + `clean.csv` | parquet only |
| `python -m src.clean --no-csv` | `clean.parquet` | yes |
| `python -m src.clean --profile` | nothing — prints the figures behind [docs/data_dictionary.md](docs/data_dictionary.md) | — |

**Never run `git add data/processed/clean.csv`.** The `.gitignore` prevents it by
default; forcing it past that is what would exhaust the LFS quota.

Then run the notebooks in numeric order for the EDA, baseline, evaluation table and SHAP
analysis. The random seed is fixed in one place and used everywhere, so a clean run
reproduces every number in the report.

---

## 7. Results

Authoritative results generated by `src/run_evaluation.py` on the held-out, leakage-safe test set (415,804 records across 9,154 unseen monitoring stations). All models are scored through the identical `evaluate()` contract on identical partitions. The numbers below match `results/*.csv` and the IEEE conference report (`report/water_quality_prediction_phase1_report.pdf`).

### 7.1 Headline Model Comparison (Held-Out Test Set, 8/8 Regime)

In the all-measurements (8/8) regime, all 8 physical and chemical parameters contributing to the CCME calculation formula are present.

| Model | Training Samples | Train MAE | Train RMSE | Test MAE | Test RMSE | Test $R^2$ | Gen Gap (RMSE) | Train Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Naive Baseline (Median)** | 1,986,397 | 12.610 | 18.011 | 13.605 | 19.492 | -0.0923 | +1.481 | 0.02 s |
| **Random Forest (100k, Eng)** | 100,000 | 0.281 | 0.604 | 0.408 | 0.920 | 0.9976 | +0.316 | 1.7 s |
| **XGBoost (Full, Eng)** | 1,986,397 | 0.096 | 0.304 | **0.100** | **0.328** | **0.9997** | **+0.024** | 49.3 s |

> [!NOTE]
> **Formula Emulation vs. True Prediction**: An $R^2$ of 0.9997 is a verification of pipeline integrity and tree expressiveness, **not** genuine predictive discovery. Because the CCME Water Quality Index is a deterministic algebraic function of these 8 chemical parameters against statutory guidelines, tree models act as universal approximators of the formula. True environmental prediction only occurs when input parameters are missing, motivating the partial-measurement ladder below.

---

### 7.2 The Partial-Measurement Ladder (Central Scientific Result)

In real-world deployment, monitoring stations frequently lack the full suite of chemical sensors. We systematically evaluate model degradation across four availability regimes by progressively withholding parameters and dynamically masking all derived interaction features:

1. **8/8 Regime**: All 8 parameters present (formula emulation benchmark).
2. **7/8 Regime**: Orthophosphate withheld (highest SHAP feature; genuine environmental estimation begins).
3. **6/8 Regime**: Orthophosphate and Ammonia withheld.
4. **4/8 Regime**: Orthophosphate, Ammonia, BOD, and Dissolved Oxygen withheld (only routine physical sensors: pH, Temperature, Nitrate, and Total Nitrogen).

| Regime | Available Parameters | Masked Parameters | Model | Features ($p$) | Val MAE | Val RMSE | Val $R^2$ | Test MAE | Test RMSE | Test $R^2$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **8/8** | All 8 measurements | None (Benchmark) | Random Forest | 38 | 0.403 | 0.908 | 0.9976 | 0.408 | 0.920 | 0.9976 |
| *(Benchmark)* | | | **XGBoost** | 23 | 0.104 | 0.355 | 0.9996 | **0.100** | **0.328** | **0.9997** |
| **7/8** | 7 measurements | Orthophosphate | Random Forest | 35 | 5.337 | 8.184 | 0.8054 | 5.328 | 8.145 | 0.8093 |
| *(Genuine)* | | | **XGBoost** | 20 | 4.787 | 7.646 | 0.8302 | **4.776** | **7.597** | **0.8341** |
| **6/8** | 6 measurements | Orthophosphate, Ammonia | Random Forest | 32 | 5.659 | 8.557 | 0.7873 | 5.675 | 8.544 | 0.7901 |
| | | | **XGBoost** | 17 | 5.207 | 8.090 | 0.8099 | **5.209** | **8.045** | **0.8139** |
| **4/8** | 4 sensors (routine) | Ortho-P, Ammonia, BOD, DO | Random Forest | 25 | 9.259 | 12.752 | 0.5276 | 9.305 | 12.781 | 0.5304 |
| *(Routine)* | | | **XGBoost** | 10 | 8.566 | 12.168 | 0.5699 | **8.596** | **12.190** | **0.5728** |

**Key Takeaways**:
- **The Reconstruction Drop**: Withholding orthophosphate alone drops $R^2$ from 0.9997 to 0.8341 and multiplies MAE by 48× (0.100 → 4.776). This step rigorously measures how much apparent accuracy in published literature is algebraic reconstruction versus real ecological inference.
- **XGBoost Superiority**: XGBoost leads Random Forest across every availability level on identical test samples (+0.0248 $R^2$ at 7/8, +0.0238 at 6/8, +0.0424 at 4/8).
- **Practical Accuracy**: Under severe missingness (4/8 regime), test MAE rises to 8.596 index points — roughly the full width of an entire CCME water quality classification band (e.g., Fair vs Good).

---

### 7.3 Ablation Studies (XGBoost, 8/8 Regime)

All ablations evaluated on the identical held-out test split (`results/ablation_results.csv`):

| Domain | Configuration | Features ($p$) | Val MAE | Val RMSE | Val $R^2$ | Test MAE | Test RMSE | Test $R^2$ | Key Finding |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Feature Engineering** | Raw features only | 10 | 0.1561 | 0.4241 | 0.99948 | 0.1517 | 0.4005 | 0.99954 | Baseline raw features |
| | Raw + Engineered features | 23 | 0.1040 | 0.3549 | 0.99963 | **0.1000** | **0.3282** | **0.99969** | **18% RMSE reduction, 34% MAE reduction** |
| **Upstream Imputation Flags** | Without flags (Primary) | 10 | 0.1561 | 0.4241 | 0.99948 | 0.1517 | 0.4005 | 0.99954 | Primary model; avoids spatial confounding |
| | With flags (Ablation) | 20 | 0.1524 | 0.4226 | 0.99948 | 0.1474 | 0.3979 | 0.99954 | Negligible gain ($\Delta R^2 < 0.00001$); justified exclusion |

- **Feature Engineering Impact**: Logarithmic transforms (`log_Orthophosphate`, `log_Ammonia`, `log_BOD`) and stoichiometric nutrient ratios (`N_P_ratio`, `BOD_DO_ratio`, `Inorganic_N_ratio`) significantly assist tree splits along non-linear biochemical decay trajectories.
- **Exclusion of Administrative Flags**: Imputation flags are concentrated in England (representing 99.8% of flagged rows). Including them risks allowing models to exploit collection artifacts as geographic shortcuts with near-zero accuracy improvement, justifying their exclusion from the primary feature matrix.

---

### 7.4 Statistical Uncertainty (95% Bootstrap Confidence Intervals)

Empirical bootstrap resampling ($B = 1{,}000$) on the test set (`results/bootstrap_ci_results.csv`):

| Model | Test MAE [95% CI] | Test RMSE [95% CI] | Test $R^2$ [95% CI] |
| :--- | :--- | :--- | :--- |
| **Naive Baseline (Median)** | 13.6047 [13.5631, 13.6457] | 19.4916 [19.4378, 19.5410] | -0.0923 [-0.0936, -0.0907] |
| **Random Forest (100k, Eng)** | 0.4078 [0.4053, 0.4103] | 0.9197 [0.9102, 0.9297] | 0.9976 [0.9975, 0.9976] |
| **XGBoost (Full, Eng)** | **0.1000 [0.0991, 0.1009]** | **0.3282 [0.3228, 0.3337]** | **0.9997 [0.9997, 0.9997]** |

The 95% confidence intervals between Random Forest and XGBoost are completely non-overlapping, confirming that XGBoost's lead is statistically significant and not an artifact of test sampling.

---

### 7.5 Geographic Generalization & Decomposition by Country

Test error evaluated across jurisdictions over all 415,804 held-out test samples (`results/country_results.csv`):

| Country | Test Samples ($n$) | % of Test Partition | Test MAE | Test RMSE | Test $R^2$ | MedAE | Max Error | Residual Mean | Residual Std |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **England** | 319,184 | 76.76% | 0.116 | 0.329 | 0.9997 | 0.026 | 6.667 | -0.0015 | 0.329 |
| **USA** | 60,787 | 14.62% | 0.062 | 0.394 | 0.9937 | 0.001 | 8.662 | -0.0014 | 0.394 |
| **Ireland** | 35,249 | 8.48% | 0.019 | 0.123 | 0.9996 | 0.002 | 6.069 | +0.0023 | 0.123 |
| **Canada** | 584 | 0.14% | 0.176 | 0.561 | 0.9982 | 0.034 | 4.740 | -0.0416 | 0.559 |
| **Overall Test** | **415,804** | **100.00%** | **0.100** | **0.328** | **0.9997** | **0.018** | **8.662** | **-0.0011** | **0.328** |

> [!IMPORTANT]
> **Why China is Reported Separately**:
> All 45,977 Chinese observations originate from a single monitoring station (`Hou Bay`). Under strict zero-spatial-leakage constraints ($A_{\text{train}} \cap A_{\text{test}} = \emptyset$), individual stations cannot be split across partitions. Greedy bin-packing deterministically allocated `Hou Bay` to the training partition. China therefore has **0 test samples**. In-sample training fit on China (`results/country_insample_china.csv`) yields MAE = 0.183, RMSE = 0.659, $R^2$ = 0.980. Isolating China in a separate diagnostic table ensures the out-of-sample test proportions sum exactly to 100.00%.

---

### 7.6 Feature Attribution & Interpretability (SHAP)

Feature importance calculated via TreeSHAP on 10,000 sampled test instances (`results/shap_importance.csv`, XGBoost 8/8 regime):

| Rank | Feature | Mean $|$SHAP$|$ | Environmental & Domain Interpretation |
| :---: | :--- | :---: | :--- |
| 1 | `Orthophosphate (mg/l)` | **8.813** | Primary nutrient limiting factor; exceedances severely penalize the CCME index |
| 2 | `log_Orthophosphate` | **3.062** | Non-linear sensitivity across low-concentration nutrient ranges |
| 3 | `Ammonia (mg/l)` | **2.455** | Direct ecotoxicity indicator; severe threshold penalties |
| 4 | `Biochemical Oxygen Demand (mg/l)` | **1.061** | Primary measure of organic pollution and oxygen depletion |
| 5 | `log_Ammonia` | **0.691** | Non-linear scaling of ammonia concentration excursions |
| 6 | `Nitrogen (mg/l)` | **0.267** | Total nitrogen enrichment and eutrophication driver |
| 7 | `pH (ph units)` | **0.229** | Acid-base balance; excursion outside 6.5–9.0 bounds penalizes score |
| 8 | `log_BOD` | **0.178** | Logarithmic transform of BOD |
| 9 | `Temperature (cel)` | **0.160** | Physical driver governing reaction kinetics and gas solubility |
| 10 | `N_P_ratio` | **0.110** | Stoichiometric Redfield balance indicating nitrogen vs phosphorus limitation |

> [!NOTE]
> **Dissolved Oxygen Diagnostic**: Dissolved Oxygen ranked 14th (Mean $|$SHAP$|$ = 0.062) despite its biological criticality in aquatic ecosystems. Exploratory analysis revealed that the upstream dataset's CCME implementation applied the DO guideline in the wrong direction (treating low DO as non-violating). This discovery highlights the value of interpretability audits in identifying latent dataset defects.

---

## 8. Team and ownership

Five members; each owns code, writes a theory note for the report, and must be able to
explain **the whole pipeline** in the viva, not only their own part.

### Roster

**Team lead:** Shaik Tajuddin ([@Taj-2005](https://github.com/Taj-2005)) — owns the
repository, reviews every pull request, and is the point of contact for the faculty.

| Name | Enrollment | Email | GitHub |
| :--- | :--- | :--- | :--- |
| Shaik Tajuddin *(team lead)* | 2401010436 | shaik.tajuddin2024@nst.rishihood.edu.in | [@Taj-2005](https://github.com/Taj-2005) |
| Parthraj Singh Bhati | 2401010320 | parthraj.bhati2024@nst.rishihood.edu.in | [@parthrajsinghbhati](https://github.com/parthrajsinghbhati) |
| Himanshu Pandey | 2401020095 | himanshu.h2024@nst.rishihood.edu.in | [@14-himanshu](https://github.com/14-himanshu) |
| Jivit Rana | 2401010202 | jivit.rana2024@nst.rishihood.edu.in | [@Jivit87](https://github.com/Jivit87) |
| Satya Yadav | 2401010427 | satya.yadav2024@nst.rishihood.edu.in | [@ACE-pwd](https://github.com/ACE-pwd) |

### Ownership

| Member | Role | Owns | Branch prefix |
| :--- | :--- | :--- | :--- |
| **Tajuddin** *(lead)* | Team lead, repository, data dictionary, cleaning, EDA | `README.md`, `requirements.txt`, `.gitignore`, `src/config.py`, `docs/data_dictionary.md`, `src/clean.py`, `01_cleaning.ipynb`, `02_eda.ipynb` | `feat/taj-*` |
| **Parthraj** | Target, split, pipeline, master evaluation | `docs/target_definition.md`, `src/split.py`, `src/preprocess.py`, `src/evaluate_api.py`, `src/evaluation.py`, `src/run_evaluation.py`, `06_evaluation.ipynb`, `results/` | `feat/parth-*` |
| **Himanshu** | Literature review, feature engineering, Random Forest | `docs/refs.bib`, `src/features.py`, `src/models_rf.py`, `04_random_forest.ipynb` | `feat/himanshu-*` |
| **Jivit** | XGBoost, ablation | `src/models_xgb.py`, `05_xgboost.ipynb`, `reports/xgb_results.csv` | `feat/jivit-*` |
| **Satya** | IEEE report, presentation, metrics, naive baseline | `report/`, `presentation/`, `src/metrics.py`, `03_naive_baseline.ipynb`, `docs/viva_prep.md`, `docs/mock_viva_plan.md` | `feat/satya-*` |

Per-member write-ups: [notes_taj.md](docs/notes_taj.md) · [notes_parth.md](docs/notes_parth.md) ·
[notes_himanshu.md](docs/notes_himanshu.md) · [notes_jivit.md](docs/notes_jivit.md) ·
[notes_satya.md](docs/notes_satya.md)

### Order of work

The repository and the shared interface come first, because nobody can code without
them; then the cleaned data, because every model depends on it. Himanshu, Jivit and Satya
then work in parallel, Parth evaluates the saved models, and Satya assembles the report
and deck from everyone's sections.

```
Taj: repo + branch rules + seed
   └─> Parth: target + shared interface
        └─> Taj: data dictionary + cleaning
                 └─> Parth: split + preprocessing pipeline
                          ├─> Himanshu: features + Random Forest
                          ├─> Jivit: XGBoost + ablation
                          └─> Satya: metrics + naive baseline   (can start on day 1)
                                   └─> Parth: final evaluation + SHAP
                                            └─> Satya: IEEE report + deck
```

---

## 9. Contributing workflow

### Quality gate

Every push and pull request runs [`.github/workflows/ci.yml`](.github/workflows/ci.yml):
lint, format check, unit tests, and a check that no notebook was committed carrying an
error traceback in its outputs. Run the same checks locally before pushing:

```bash
ruff check src/ tests/        # lint
ruff format src/ tests/       # format (use --check to verify without writing)
pytest tests/                 # unit tests
```

The tests run on small synthetic frames and **never touch the dataset**. That is
deliberate: CI checks out with `lfs: false`, because pulling the 305 MiB raw file on
every push would exhaust the repository's 1 GiB/month LFS bandwidth allowance — which is
shared by all five of us. Correctness on the real data is demonstrated by
[`notebooks/01_cleaning.ipynb`](notebooks/01_cleaning.ipynb), whose outputs are committed.


- **One branch per person**, named `feat/<name>-<task>` (e.g. `feat/taj-cleaning`).
- Open a **pull request into `main`** and wait for the team lead's review (Taj). **Never push straight to `main`.**
- **Commit small and often** with clear messages — each member's contribution must be
  visible in the history, since the viva is individually assessed.
- Fixed random seed everywhere; no result is reported that the commands above cannot reproduce.
- Every member maintains a one-page `docs/notes_<name>.md` titled *"what I built and why"*.
- Every cleaning, feature and tuning decision is logged **with its reason** — examiners
  will ask why, not just what.

---

## 10. Phase 2 preview

Phase 2 adds the deep learning model **on the same split and preprocessing**, then
compares it fairly against the Phase 1 tree models:

- Deep learning literature review: MLPs on tabular data, regularization, uncertainty.
- A regularized MLP in PyTorch — dropout, weight decay, early stopping, batch norm —
  with the output head matched to the target (regression head for an index,
  classification head only for a defined class scheme).
- Architecture justification: why this depth, width and head suit this problem.
- Ablations on depth, width, regularization, and with/without engineered features.
- Optional **hybrid** for additional marks, only where technically justified — e.g. an
  MLP trained on XGBoost residuals, compared fairly against each model alone.
- Technical validation: calibration, uncertainty, and leakage checks.

---

## 11. References

Citations are maintained in a single [docs/refs.bib](docs/refs.bib), which the LaTeX
report cites from directly. Core pre-reads for Phase 1:

- Breiman, L. (2001). *Random Forests.* Machine Learning, 45(1), 5–32.
- Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* KDD.
- Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions.* NeurIPS.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning* — feed-forward networks (Phase 2).

---

## 12. Stack

Python · pandas · NumPy · scikit-learn · XGBoost · Optuna · SHAP · matplotlib · seaborn ·
PyTorch (Phase 2) · Jupyter
