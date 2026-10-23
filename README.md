# Water Quality Prediction

Predicting a documented water quality target from physical and chemical measurements, and
comparing interpretable tree ensembles against a neural model on an identical,
leakage-safe split.

**Course project — CSAI (AML / DL), Project 18.** Delivered in two graded phases:

| Phase | Scope | Models |
| :--- | :--- | :--- |
| **Phase 1** (this repo, current) | Classical machine learning | Naive baseline, Random Forest, XGBoost |
| **Phase 2** (planned) | Deep learning + optional hybrid | Regularized MLP (PyTorch), optional MLP-on-residuals hybrid |

---

## Current Status

The initial shared foundation (Phase 1, Stage 1) is **complete** (Parth):
- **Target Confirmed**: `CCME_Values`
- **Problem Type**: **Regression**
- **Target Documentation & Leakage Analysis**: Documented in [docs/target_definition.md](docs/target_definition.md)
- **Shared Evaluation API**: Established in [src/evaluate_api.py](src/evaluate_api.py)
- **Configuration**: Configured in [src/config.py](src/config.py)

**Pending next steps:**
- **Taj** → data cleaning (`src/clean.py`, `notebooks/01_cleaning.ipynb`, `data/processed/clean.csv`) and EDA (`notebooks/02_eda.ipynb`, `docs/data_dictionary.md`)
- **Parth** → shared split (`src/split.py`) and preprocessing pipeline (`src/preprocess.py`) once cleaned data is available
- **Himanshu** → literature review (`refs.bib`), feature engineering (`src/features.py`), and Random Forest (`src/models_rf.py`, `04_random_forest.ipynb`)
- **Jivit** → XGBoost model training, tuning, and ablation (`src/models_xgb.py`, `05_xgboost.ipynb`)
- **Satya** → metric implementations (`src/metrics.py`), naive baseline (`03_naive_baseline.ipynb`), LaTeX report & presentation deck

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
- **Raw data is never edited in place and never committed.** `data/` is git-ignored
  except for its folder structure; see [Reproducing results](#6-reproducing-results).

### Data handling rules

| Stage | Rule |
| :--- | :--- |
| Units | made consistent across all rows before anything else |
| Detection limits | values like `<0.5` handled by **one stated rule**, documented in the cleaning notebook |
| Impossible values | **flagged**, not silently deleted (e.g. pH outside 0–14, negative turbidity) |
| Duplicates | exact duplicates removed |
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
│   ├── raw/                     # original dataset, never edited (git-ignored)
│   ├── processed/               # clean.csv, output of src/clean.py (git-ignored)
│   └── splits/                  # saved train/val/test indices (git-ignored)
├── notebooks/
│   ├── 01_cleaning.ipynb        # cleaning decisions, each with its reason
│   ├── 02_eda.ipynb             # distributions, missingness, correlation, outliers
│   ├── 03_naive_baseline.ipynb  # mean / median / majority-class floor
│   ├── 04_random_forest.ipynb   # RF training + hyperparameter search
│   ├── 05_xgboost.ipynb         # XGBoost training + tuning + ablation
│   ├── 06_evaluation.ipynb      # final results table, residuals, error by site/time
│   └── 07_shap.ipynb            # SHAP + feature importance for the best model
├── src/
│   ├── evaluate_api.py          # shared interface: load_data(), get_splits(), evaluate()
│   ├── clean.py                 # raw -> data/processed/clean.csv
│   ├── split.py                 # seeded site/time/random split, indices saved to disk
│   ├── preprocess.py            # sklearn pipeline, fitted on train only
│   ├── features.py              # derived features, leakage-checked
│   ├── metrics.py               # MAE / RMSE / R2, or macro F1 + calibration
│   ├── models_rf.py             # Random Forest training + tuning
│   └── models_xgb.py            # XGBoost training + tuning
├── reports/
│   ├── figures/                 # figures used in the IEEE report and deck
│   ├── rf_results.csv           # Random Forest metrics on shared val/test
│   └── xgb_results.csv          # XGBoost metrics + ablation
├── docs/
│   ├── target_definition.md     # the target, its standard/index definition, leakage risks
│   ├── data_dictionary.md       # every column: unit, meaning, source, license
│   ├── viva_prep.md             # likely questions for every part of the project
│   ├── refs.bib                 # single source of truth for citations
│   └── notes_<name>.md          # one page per member: "what I built and why"
├── requirements.txt
├── .gitignore
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

Then place the raw dataset in `data/raw/` under the filename recorded in
[docs/data_dictionary.md](docs/data_dictionary.md).

---

## 6. Reproducing results

Run in this order — each step depends on the previous one.

```bash
# 1. raw -> cleaned, documented table (no imputation here)
python -m src.clean

# 2. create the shared split once; writes indices to data/splits/
python -m src.split

# 3. models, each evaluated through the shared evaluate() function
python -m src.models_rf        # -> reports/rf_results.csv
python -m src.models_xgb       # -> reports/xgb_results.csv
```

Then run the notebooks in numeric order for the EDA, baseline, evaluation table and SHAP
analysis. The random seed is fixed in one place and used everywhere, so a clean run
reproduces every number in the report.

---

## 7. Results

Filled in once the final evaluation runs on the untouched test set. All three models are
scored on the **same** test set via the same `evaluate()` call.

| Model | Metric 1 | Metric 2 | Metric 3 |
| :--- | :--- | :--- | :--- |
| Naive baseline | — | — | — |
| Random Forest | — | — | — |
| XGBoost | — | — | — |

**Ablation** — XGBoost with and without engineered features, on the identical split:

| Configuration | Metric 1 | Metric 2 |
| :--- | :--- | :--- |
| Base features only | — | — |
| Base + engineered features | — | — |

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
| **Parthraj** | Target, split, pipeline, final evaluation | `docs/target_definition.md`, `src/split.py`, `src/preprocess.py`, `src/evaluate_api.py`, `06_evaluation.ipynb`, `07_shap.ipynb` | `feat/parth-*` |
| **Himanshu** | Literature review, feature engineering, Random Forest | `docs/refs.bib`, `src/features.py`, `src/models_rf.py`, `04_random_forest.ipynb` | `feat/himanshu-*` |
| **Jivit** | XGBoost, ablation | `src/models_xgb.py`, `05_xgboost.ipynb`, `reports/xgb_results.csv` | `feat/jivit-*` |
| **Satya** | IEEE report, presentation, metrics, naive baseline | Overleaf project, deck, `src/metrics.py`, `03_naive_baseline.ipynb`, `docs/viva_prep.md` | `feat/satya-*` |

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
# water_quality_prediction
