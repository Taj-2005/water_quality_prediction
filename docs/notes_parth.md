# What I built and why — Parthraj Singh Bhati

| | |
| :--- | :--- |
| **GitHub** | [@parthrajsinghbhati](https://github.com/parthrajsinghbhati) |
| **Role** | Target, split, pipeline, final evaluation |
| **Branch prefix** | `feat/parth-*` |
| **Theory note for the report** | Split design and leakage; SHAP and feature importance |

One page, maintained by Parthraj Singh Bhati. Examiners read this alongside the commit history to
see individual contribution, so write it as you go rather than at the end.

## Files I own

`docs/target_definition.md`, `src/split.py`, `src/preprocess.py`, `src/evaluate_api.py`, `notebooks/06_evaluation.ipynb`, `notebooks/07_shap.ipynb`

## What I built

In Phase 1, Stage 1 (Initial Foundation):
1. **Target Confirmation and Documentation (`docs/target_definition.md`)**:
   - Confirmed the target as `CCME_Values` (continuous numerical score, 31.304 to 100.0).
   - Classified the task as **Regression**.
   - Verified the exact CCME mathematical formula used during dataset harmonization (`Data_Processing_Harmonization.ipynb`), matching Scope ($F_1$), Frequency ($F_2$), and Amplitude ($F_3$) excursions against Ireland WQI guidelines.
2. **Target Leakage Analysis**:
   - Discovered and proved exact mathematical formula inversion: `CCME_Values` was calculated directly from 8 chemical/physical input parameters (`Ammonia`, `BOD`, `DO`, `Orthophosphate`, `pH`, `Temperature`, `Nitrogen`, `Nitrate`).
   - Identified `CCME_WQI` as direct label leakage (a deterministic 5-class binning of the target).
   - Documented the leakage risk and formal exclusion decisions in `docs/target_definition.md` without prematurely deleting raw columns.
3. **Shared Interface Stub (`src/evaluate_api.py`)**:
   - Defined `load_data()`, `get_splits()`, and `evaluate(y_true, y_pred)` with typed signatures, robust docstrings, and clean exception handling pointing to upstream dependencies.
4. **Configuration (`src/config.py`)**:
   - Bound `TARGET_COLUMN = "CCME_Values"`, `TARGET_KIND = "regression"`, and `RAW_CSV`.

In Phase 1, Stage 2 (Split & Preprocessing Pipeline):
5. **Grouped Split Pipeline (`src/split.py`)**:
   - **Area-Based Grouping:** Implemented `make_grouped_split()`, grouping observations by monitoring station (`Area`, 61,689 unique sites) to prevent spatial autocorrelation leakage. Stations in train NEVER appear in validation or test.
   - **Country Stratification:** Partitioned areas within each country using greedy bin-packing, preserving representative country distributions (England 70/15/15, USA 70/15/15, Ireland 70/15/15, Canada 70/15/15, China 100% train due to its single bay).
   - **Deterministic & Compact:** Uses `SEED = 42`. Persists lightweight index assignments to `data/splits/splits.parquet` (429 KB) without duplicating the 300+ MB dataset.
6. **Train-Only Preprocessing Pipeline (`src/preprocess.py`)**:
   - **Strict Target Isolation:** Decoupled `y = df['CCME_Values']`. Ensured `CCME_Values`, `CCME_WQI`, `Area`, and raw date columns are completely stripped from feature matrix $X$.
   - **Train-Only Fitting:** Implemented `WaterQualityPreprocessor`, learning imputation medians, category encodings, and optional standard scalers strictly on $X_{\text{train}}$. Applied to validation and test without refitting.
   - **Upstream Flag Handling:** By default excludes `<feature>__imputed_upstream`, `n_imputed_upstream`, and `flag_any_imputed_upstream` from primary features to avoid exploiting the England imputation confound, while exposing `include_imputation_flags=True` for Jivit's ablation.
7. **Partial-Measurement Masking Mechanism (`mask_features`)**:
   - Created controlled masking supporting `8/8`, `7/8`, `6/8`, and `4/8` parameter suites.
   - **Dependency Propagation:** Registered chemical interaction dependencies (e.g. $N:P$ ratio, Inorganic N ratio, Organic N fraction, BOD/DO ratio). If an underlying measurement is masked, all dependent derived features are automatically dropped to prevent indirect leakage.
   - Non-destructive: leaves the clean dataset on disk completely unaltered.
8. **Shared API Integration (`src/evaluate_api.py`)**:
   - Implemented `get_splits()`, providing Himanshu (RF), Jivit (XGBoost), and Satya (Baseline) with ready-to-model `(X_train, X_val, X_test, y_train, y_val, y_test)` in 1.48 seconds.
9. **Automated Unit Testing (`tests/test_split.py`, `tests/test_preprocess.py`)**:
   - Added 20 comprehensive unit tests covering zero row overlap, zero Area leakage, train-only fitting, immutability, and partial feature masking. (34 passing tests repository-wide).

## Why I built it that way

1. **Why `Area` grouping instead of simple random k-fold**:
   - Monitoring stations along river networks exhibit spatial autocorrelation. Standard random splitting allows models to memorize localized catchment features rather than learning water chemistry (Roberts et al., 2017). Grouping by `Area` forces genuine spatial generalization to unseen monitoring sites.
2. **Why greedy bin-packing within countries**:
   - Station sizes vary widely (from 1 observation to >45,000). A naive group shuffle can randomly assign large stations into test, skewing proportions. Bin-packing guarantees that the 70/15/15 proportions are closely matched ($70.49\%$ train, $14.76\%$ val, $14.76\%$ test) while maintaining strict zero area leakage.
3. **Why lightweight split indices instead of saving duplicate CSVs**:
   - Copying 2.8M rows across separate train, val, and test files would consume >1.5 GB of disk space. Storing a 429 KB categorical parquet index allows fast indexing in RAM without disk bloat.
4. **Why dependency propagation in partial masking**:
   - If an experiment masks `Orthophosphate` to test partial measurement availability, leaving the $N:P$ ratio in $X$ allows tree ensembles to back-calculate Orthophosphate: $\text{Orthophosphate} \approx \text{Nitrogen} / \text{Ratio}_{N:P}$. Dropping dependent derived features prevents indirect leakage.

## How it connects to the rest of the pipeline

- **Supplied to Himanshu (Random Forest)**:
  - Himanshu calls `X_train, X_val, X_test, y_train, y_val, y_test = get_splits()`.
  - He builds candidate features in `src/features.py`. If derived features are registered in `DEFAULT_FEATURE_DEPENDENCIES`, the partial masking harness handles them safely.
- **Supplied to Jivit (XGBoost & Ablation)**:
  - Jivit executes the upstream imputation ablation by calling `get_splits(include_imputation_flags=True)`.
  - Jivit executes the partial-measurement ablation across `8/8`, `7/8`, `6/8`, `4/8` via `get_splits(partial_scenario=...)`.
- **Supplied to Satya (Baseline & Metrics)**:
  - Satya imports `get_splits()` in `notebooks/03_baseline.ipynb` to evaluate the naive baseline (train mean/median) on the identical test partition.

## What I would do differently

- For countries with only a single station (e.g. China's Hou Bay), station-level grouping forces all observations into one partition (train). In future datasets with broader catchment coverage, multi-station networks for all territories would allow out-of-country validation blocks.
