# Engineering Contribution Log & Architecture Record — Parthraj Singh Bhati

| | |
| :--- | :--- |
| **Author** | Parthraj Singh Bhati ([@parthrajsinghbhati](https://github.com/parthrajsinghbhati)) |
| **Project Role** | Target Specification, Spatial Splitting, Train-Only Preprocessing, Leakage Prevention, Partial-Measurement Framework, Shared Evaluation API, Master Evaluation Integration |
| **Git Branch** | `feat/parth` (merged upstream via [PR #7](https://github.com/Taj-2005/water_quality_prediction/pull/7) + Final Integration Phase) |
| **Owned Code** | `src/split.py`, `src/preprocess.py`, `src/evaluate_api.py`, `src/evaluation.py`, `src/run_evaluation.py` |
| **Owned Tests** | `tests/test_split.py`, `tests/test_preprocess.py`, `tests/test_evaluation.py` (26 unit tests) |
| **Owned Artifacts** | `data/splits/splits.parquet`, `results/*.csv`, `reports/figures/06_*.png`, `notebooks/06_evaluation.ipynb` |

---

## 1. Parth's Overall Contribution

Parthraj Singh Bhati engineered the foundational data architecture, leakage-prevention governance, partial-measurement experiment framework, shared data contracts, and final evaluation orchestration for the Water Quality Prediction project (`CCME_Values` regression).

### Division of Responsibility Across the Team
- **Taj**: Data cleaning, harmonization, and missing-data profiling (`src/clean.py`, `notebooks/01_cleaning.ipynb`, `notebooks/02_eda.ipynb`).
- **Himanshu Pandey**: Literature review, domain feature engineering definitions, and Random Forest baseline modeling/tuning (`src/features.py`, `src/models_rf.py`, `notebooks/04_random_forest.ipynb`).
- **Jivit Rana**: XGBoost model training, Optuna hyperparameter optimization, and initial model exploration (`src/models_xgb.py`, `notebooks/05_xgboost.ipynb`).
- **Satya Yadav**: Metric definitions, Naive Baseline constant-prediction implementation, and IEEE report documentation (`src/metrics.py`, `notebooks/03_naive_baseline.ipynb`).
- **Parthraj Singh Bhati (This Record)**: Target formula specification and algebraic inversion analysis (`docs/target_definition.md`); spatial grouped data splitting (`src/split.py`); train-only preprocessing and leakage governance (`src/preprocess.py`); dynamic partial-measurement framework and chemical dependency propagation (`src/preprocess.py`); shared data contract (`src/evaluate_api.py`); common evaluation interface (`src/evaluation.py`); and master evaluation execution, full-data ablations, error analysis, and figure generation (`src/run_evaluation.py`).

*Note*: Parth did not implement or tune Random Forest or XGBoost, but built the shared infrastructure that made all models leak-free, reproducible, and directly comparable.

---

## 2. Area-Grouped Data Splitting (`src/split.py`)

- **What I Did**: Implemented `make_grouped_split()` and `save_splits()` in `src/split.py`. The module partitions 2,818,006 rows across 61,689 monitoring stations (`Area`) into Train (70%), Validation (15%), and Test (15%) splits, stratified by `Country` using greedy bin-packing (`SEED = 42`). Split assignments are persisted to `data/splits/splits.parquet` (429 KB).
- **Why I Did It**: Environmental monitoring stations collect repetitive water samples over time. Random row splitting places samples from the same geographical site into both train and test partitions. Machine learning models exploit this spatial autocorrelation (Roberts et al., 2017) to memorize site-specific baseline chemistry rather than learning transferable patterns, producing artificially optimistic test metrics.
- **How I Implemented It**: Stations within each country are sorted descending by observation count and placed iteratively into whichever partition is furthest below its target proportion. This guarantees zero station overlap across splits while preserving geopolitical balance.
- **Why Preferred**: Guarantees $A_{\text{train}} \cap A_{\text{val}} = \emptyset$ and $A_{\text{train}} \cap A_{\text{test}} = \emptyset$ while matching target proportions within 0.49% across 2.8M rows.
- **Alternatives Considered & Rejected**:
  1. *Random Row Split*: Rejected because spatial autocorrelation across identical sites leaks local site characteristics into the test set.
  2. *Purely Temporal Split*: Rejected because sensor deployment across countries is historically unbalanced (e.g., Canada spans 1940–2018, USA spans 1968–2021, England dominates post-1990). A hard date cutoff allocates whole countries into single partitions.
  3. *Unstratified Group Split*: Rejected because international sample size disparities cause smaller jurisdictions (e.g., Canada, Ireland) to be randomly excluded from train or test partitions.
- **Special Case: China**: All 45,977 Chinese observations originate from a single monitoring station (`Hou Bay`). Under strict zero-spatial-leakage constraints, individual stations cannot be split across partitions. Greedy bin-packing deterministically allocated `Hou Bay` to the largest partition (Train). Consequently, China has **0 test samples**; reported China metrics reflect **in-sample domain fitting**, not out-of-station generalization.
- **Final Outcome**:
  - **Train**: 1,986,397 rows (70.49%) across 43,382 stations.
  - **Validation**: 415,805 rows (14.76%) across 9,153 stations.
  - **Test**: 415,804 rows (14.76%) across 9,154 stations.
- **Limitations**: Station-level grouping does not prevent basin-wide correlations between hydrologically connected adjacent stations.

---

## 3. Train-Only Preprocessing (`src/preprocess.py`)

- **What I Did**: Authored `WaterQualityPreprocessor` in `src/preprocess.py`, supporting target separation, train-learned fallback median imputation, categorical level learning (`Country`, `Waterbody Type`), and optional standard scaling.
- **Why I Did It**: Computing transformation parameters (medians, categorical levels, scalers) across the entire dataset leaks distribution statistics from validation/test sets into training (statistical snooping; Kaufman et al., 2012).
- **How I Implemented It**: Strict `fit()` vs. `transform()` separation: parameters are learned strictly on `X_train` inside `fit()` and frozen. `transform()` applies learned values to validation and test sets without updating parameters. Calling `transform()` before `fit()` raises a `RuntimeError`.
- **Why Preferred**: Ensures mathematical isolation of holdout partitions while providing a standard scikit-learn transformer interface.
- **Alternatives Considered & Rejected**:
  1. *Global Preprocessing Before Split*: Rejected because full-dataset statistics leak test distribution information into train.
  2. *Fitting Separately on Each Split*: Rejected because independent transformations shift partitions by unaligned offsets, invalidating evaluation.
- **Final Outcome**: Downstream models receive identically scaled and encoded features with zero statistical snooping, validated by `tests/test_preprocess.py`.

---

## 4. Leakage Prevention Governance

Every potential leakage vector was identified, formally evaluated, and blocked in `src/preprocess.py` and `src/features.py`:

1. **`CCME_Values` (Target Leakage)**:
   - *Problem → Decision*: Continuous regression target. Decoupled `y = df["CCME_Values"]` immediately on loading.
   - *Why → Implementation*: Prevents trivial identity learning ($y=x$). `extract_features_target()` drops target; `validate_no_leakage()` raises `ValueError` if present in $X$.
   - *Alternative Rejected*: Relying on model scripts to drop target manually (high risk of human error).
   - *Outcome*: Target strictly decoupled.
2. **`CCME_WQI` (Post-Hoc Label Leakage)**:
   - *Problem → Decision*: Categorical 5-class binning of `CCME_Values`. Hard-coded exclusion in `EXCLUDED_COLUMNS`.
   - *Why → Implementation*: Trees use categorical intervals to bound continuous predictions with artificial precision. Blocked in `src/preprocess.py`.
   - *Alternative Rejected*: Keeping as input (represents post-hoc label unavailable at true inference).
   - *Outcome*: Direct label leakage completely prevented.
3. **`Area` (Spatial Site Memorization)**:
   - *Problem → Decision*: 61,689 monitoring site IDs. Used strictly as split key; excluded from predictive features $X$.
   - *Why → Implementation*: High-cardinality site IDs allow trees to memorize location baselines rather than chemical interactions, failing on unseen stations. Filtered in `extract_features_target()`.
   - *Alternative Rejected*: One-hot or target encoding `Area` (fails on unseen test sites).
   - *Outcome*: Models forced to generalize based solely on physical-chemical inputs.
4. **`Date` / Raw Timestamp Strings (Temporal Memorization)**:
   - *Problem → Decision*: Raw timestamps. Excluded from $X$; only cyclical features (`sin_day`, `cos_day`, `month`) allowed.
   - *Why → Implementation*: Trees split on raw timestamps to learn historical sample windows. Stripped during preprocessing.
   - *Alternative Rejected*: Passing raw dates (causes overfitting to collection periods).
   - *Outcome*: Models learn seasonal cycles instead of chronological boundaries.
5. **Upstream Imputation Flags (Administrative Confounding)**:
   - *Problem → Decision*: Flags indicating upstream national median substitution (predominantly England). Excluded from primary feature matrix.
   - *Why → Implementation*: Trees split on flags as proxies for England's river network. Filtered in `extract_features_target()`; exposed only via `include_imputation_flags=True` for ablations.
   - *Alternative Rejected*: Including flags in primary feature set (pollutes physical model with collection artifacts).
   - *Outcome*: Primary model learns purely from physical-chemical observations.
6. **Train-Only Preprocessing (Statistical Snooping)**:
   - *Problem → Decision*: Holdout data influencing transformations. Imputers and encoders fit strictly on $X_{\text{train}}$.
   - *Why → Implementation*: Guarantees holdout sets remain strictly unseen. Enforced by `WaterQualityPreprocessor.fit()`.
   - *Alternative Rejected*: Global dataset fitting (leaks test distributions).
   - *Outcome*: Verified zero statistical leakage across all partitions.
7. **Area-Grouped Splitting (Spatial Station Overlap)**:
   - *Problem → Decision*: Same-station samples in train and test. Enforced site-grouped greedy bin-packing.
   - *Why → Implementation*: Ensures holdout metrics evaluate genuine spatial generalization. Enforced in `src/split.py`.
   - *Alternative Rejected*: Random row splitting (yields overly optimistic test scores).
   - *Outcome*: Guaranteed disjoint station sets across all partitions.
8. **Partial-Measurement Dependency Masking (Indirect Reconstruction)**:
   - *Problem → Decision*: Derived features (e.g., $N:P$) retaining masked parent parameters. Automated cascading pruning via dependency registry.
   - *Why → Implementation*: Trees algebraically recover masked parameters from ratios. Automated in `mask_features()`.
   - *Alternative Rejected*: Manual feature lists per scenario (error-prone, unmaintainable).
   - *Outcome*: Completely leak-free partial measurement experiments.

---

## 5. The Upstream Imputation Problem

- **What I Discovered**: Profiling revealed that 1,715,965 rows (**60.89%**) contain suspected upstream median fill values, concentrated in England's river monitoring network (BOD, Ammonia, and Nitrogen exhibit sharp spikes at national medians). Only 39.11% of records are unflagged.
- **Why It Matters**: Upstream imputation occurred before raw data collection and cannot be removed. Feeding imputation indicators to models enables them to exploit administrative artifacts rather than genuine water chemistry.
- **Design Decision**: Exclude all imputation indicators (`*__imputed_upstream`, `n_imputed_upstream`, `flag_any_imputed_upstream`) from primary feature matrices, while providing an explicit ablation toggle (`include_imputation_flags=True`) to quantify artifact reliance.
- **Alternative Rejected**: Including indicators in primary models was rejected because models learn regulatory collection patterns instead of physical water dynamics.
- **Outcome**: Primary models remain scientifically grounded; ablation verifies that flags provide negligible incremental predictive value ($-0.65\%$ RMSE delta).

---

## 6. Partial-Measurement Framework (`mask_features`)

- **What I Did**: Implemented `mask_features()` in `src/preprocess.py` to simulate real-world sensor suite unavailability across four standard regimes:
  - **8/8 (Full Suite)**: All 8 core chemical parameters available.
  - **7/8 (Missing Orthophosphate)**: Orthophosphate omitted.
  - **6/8 (Missing Nutrients & Organics)**: Orthophosphate and BOD omitted.
  - **4/8 (Core Physical-Chemical Only)**: Only pH, Dissolved Oxygen, Temperature, and Nitrate available.
- **Why I Did It**: In operational field monitoring, complete 8-parameter suites are rarely available due to sensor costs. Furthermore, because the 8/8 model emulates a deterministic formula, evaluating performance under incomplete measurements is the true test of machine learning utility.
- **How I Implemented It**: In-memory dynamic masking: unavailable measurement columns and all dependent engineered features are removed on the fly without modifying disk data.
- **Why Preferred**: Eliminates file duplication and guarantees that all scenarios share the exact same underlying split, validation set, test set, and preprocessing rules.
- **Alternative Rejected**: Generating separate static datasets for each scenario was rejected due to disk bloat, maintenance overhead, and split inconsistency risks.

---

## 7. Dependency Propagation for Engineered Features

- **The Rule**: When a raw measurement is masked, every engineered feature requiring that measurement must also be dropped.
- **Why It Matters**: Retaining stoichiometric ratios (e.g., $N:P$ ratio) when Orthophosphate is masked allows decision trees to invert the ratio ($\text{Orthophosphate} \approx \text{Nitrogen} / \text{Ratio}_{N:P}$), creating indirect target leakage.
- **Implementation**: Maintained a central dependency registry in `src/preprocess.py` matching `src/features.py`:
  ```python
  DEFAULT_FEATURE_DEPENDENCIES = {
      "N_P_ratio": ["Nitrogen (mg/l)", "Orthophosphate (mg/l)"],
      "Inorganic_N_ratio": ["Ammonia (mg/l)", "Nitrate (mg/l)"],
      "Organic_N_fraction": ["Nitrogen (mg/l)", "Nitrate (mg/l)", "Ammonia (mg/l)"],
      "BOD_DO_ratio": ["Biochemical Oxygen Demand (mg/l)", "Dissolved Oxygen (mg/l)"],
      "log_Ammonia": ["Ammonia (mg/l)"],
      "log_BOD": ["Biochemical Oxygen Demand (mg/l)"],
      "log_Orthophosphate": ["Orthophosphate (mg/l)"],
      "log_Nitrate": ["Nitrate (mg/l)"],
      "log_Nitrogen": ["Nitrogen (mg/l)"],
  }
  ```
  `mask_features()` inspects all columns against this registry and purges any feature whose parent measurements are missing.
- **Alternative Rejected**: Manually writing hardcoded column lists for each partial scenario was rejected as brittle, unmaintainable, and prone to leakage.

---

## 8. Shared Data Access Contract (`get_splits()`)

- **What I Did**: Developed `get_splits()` in `src/evaluate_api.py` as the unified data access interface:
  ```python
  X_train, X_val, X_test, y_train, y_val, y_test = get_splits(
      splits_dir=None,
      include_metadata=True,
      include_imputation_flags=False,
      scale_numeric=False,
      partial_scenario="8/8",
  )
  ```
- **Why I Did It**: Guarantees that every model (Baseline, Random Forest, XGBoost) is trained and evaluated on identical partitions, identical targets, and identical preprocessing rules.
- **Why Preferred**: Loads and serves the complete 2.8M row dataset in ~1.5 seconds via memory-mapped parquet reads.
- **Alternative Rejected**: Independent data loading by individual teammates was rejected because slight differences in random seeds or preprocessing order invalidate model comparisons.

---

## 9. Final Evaluation & Integration Work

Parth engineered the evaluation architecture to unify disparate model benchmarks:
- **Common Evaluation Interface (`src/evaluation.py`)**:
  - `evaluate_regression()`: Computes MAE, RMSE, $R^2$, Explained Variance, MedAE, Max Error, and residual distribution statistics.
  - `evaluate_by_group()`: Slices performance across countries and waterbody types.
  - `bootstrap_test_metrics()`: Generates 1,000-draw non-parametric 95% bootstrap confidence intervals on fixed test predictions.
- **Master Evaluation Runner (`src/run_evaluation.py`)**:
  - Evaluates Baseline, Random Forest, and XGBoost on holdout test data ($N_{\text{test}} = 415,804$).
  - Executes full-data partial measurement sweeps and feature engineering/imputation ablations.
  - Generates residual diagnostics, TreeSHAP feature importance, results CSVs (`results/`), and publication plots (`reports/figures/`).
- **Interactive Reporting Notebook (`notebooks/06_evaluation.ipynb`)**:
  - Consolidation notebook displaying all diagnostic tables and figures.

*Ownership Clarification*: Parth engineered the evaluation framework, pipeline orchestration, and analysis scripts. Model implementations and hyperparameters were authored by Satya (Baseline), Himanshu (Random Forest), and Jivit (XGBoost).

---

## 10. Final Harmonized Experiment Design

- **The Sample-Size Confound**: Early exploratory runs evaluated partial measurements on a 100k subsample (`subsample=100_000`) for rapid iteration, producing an 8/8 Test RMSE of **0.4036**. However, the primary XGBoost model was trained on the full dataset ($N_{\text{train}} = 1,986,397$), achieving Test RMSE **0.3282**. Comparing a full-data 8/8 model against 100k-sample partial models confounded measurement availability with training set size.
- **The Harmonized Solution**: Updated `src/run_evaluation.py` so that XGBoost partial-measurement experiments train on the **full dataset ($N_{\text{train}} = 1,986,397$)** across all scenarios (`8/8`, `7/8`, `6/8`, `4/8`). The 8/8 partial anchor now directly reflects the primary 0.3282 benchmark.
- **RF Alignment**: Random Forest was standardized to $N_{\text{train}} = 100,000$ across all partial runs, matching its primary benchmark ($0.9197$) due to scikit-learn memory constraints.
- **Schema Update**: Added an explicit `n_train_samples` column to all output CSVs to eliminate sample-size ambiguity.

---

## 11. Current Primary Model Results

*Evaluated on the identical 415,804 holdout test samples using `src/evaluation.py`. 95% CIs derived from 1,000 bootstrap draws.*

| Model | $N_{\text{train}}$ | Val MAE | Val RMSE | Val R² | Test MAE | Test RMSE | Test R² | Test RMSE 95% CI | Train Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Naive Baseline (Median)** | 1,986,397 | 13.5192 | 19.3993 | -0.0932 | 13.6047 | 19.4916 | -0.0923 | [19.4378, 19.5410] | 0.02s |
| **Random Forest (100k, Eng)** | 100,000 | 0.4031 | 0.9075 | 0.9976 | 0.4078 | 0.9197 | 0.9976 | [0.9102, 0.9297] | 1.84s |
| **XGBoost (Full, Eng)** | **1,986,397** | **0.1040** | **0.3549** | **0.9996** | **0.1000** | **0.3282** | **0.9997** | **[0.3228, 0.3337]** | **50.24s** |

*Sample Size Note*: Random Forest was trained on a 100k subsample due to scikit-learn multi-core memory limits (>16 GB RAM for 1.98M rows). XGBoost uses histogram-based binning (`tree_method='hist'`), allowing full 1.98M row training in 50.2s.

---

## 12. Controlled Partial-Measurement Results

*Evaluated on holdout test partition ($N_{\text{test}} = 415,804$). XGBoost trained on full data ($N_{\text{train}} = 1,986,397$); RF trained on $N_{\text{train}} = 100,000$.*

| Availability | Missing Parameters | Model | $N_{\text{train}}$ | $p$ | Test MAE | Test RMSE | Test R² |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **8/8 (Full Suite)** | None | **RF**<br>**XGBoost** | 100,000<br>**1,986,397** | 38<br>23 | 0.4078<br>**0.1000** | 0.9197<br>**0.3282** | 0.9976<br>**0.9997** |
| **7/8 (Missing Orthophosphate)** | Orthophosphate | **RF**<br>**XGBoost** | 100,000<br>**1,986,397** | 35<br>20 | 5.3283<br>**4.7761** | 8.1450<br>**7.5966** | 0.8093<br>**0.8341** |
| **6/8 (Missing Nutrients/Organics)** | Orthophosphate, BOD | **RF**<br>**XGBoost** | 100,000<br>**1,986,397** | 32<br>17 | 5.6751<br>**5.2089** | 8.5442<br>**8.0451** | 0.7901<br>**0.8139** |
| **4/8 (Core Physical-Chemical)** | Orthophosphate, BOD, NH3, N | **RF**<br>**XGBoost** | 100,000<br>**1,986,397** | 25<br>10 | 9.3046<br>**8.5957** | 12.7807<br>**12.1898** | 0.5304<br>**0.5728** |

*Takeaway*: Dropping Orthophosphate (7/8) causes the largest error surge (Test RMSE $0.3282 \to 7.5966$, $+2215\%$), empirically proving that phosphorus exceedances are the primary mathematical driver of water quality index degradation.

---

## 13. Controlled Feature Engineering Ablation

*Both configurations evaluated on full training ($N_{\text{train}} = 1,986,397$) and test ($N_{\text{test}} = 415,804$) sets.*

| Configuration | Features ($p$) | Val MAE | Val RMSE | Val R² | Test MAE | Test RMSE | Test R² | $\Delta \text{Test RMSE}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Raw Features** | 10 | 0.1561 | 0.4241 | 0.9995 | 0.1517 | 0.4005 | 0.9995 | Reference |
| **Raw + Engineered Features** | 23 | **0.1040** | **0.3549** | **0.9996** | **0.1000** | **0.3282** | **0.9997** | **-0.0723 (-18.1%)** |

*Takeaway*: Stoichiometric ratios ($N:P$, $BOD:DO$) and log transforms reduce test RMSE by **18.1%** and test MAE by **34.1%**, demonstrating that domain feature representations provide valuable inductive structure for tree ensembles.

---

## 14. Controlled Upstream Imputation Ablation

*Both configurations evaluated on full training ($N_{\text{train}} = 1,986,397$) and test ($N_{\text{test}} = 415,804$) sets.*

| Configuration | Features ($p$) | Val MAE | Val RMSE | Val R² | Test MAE | Test RMSE | Test R² | $\Delta \text{Test RMSE}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Without Imputation Flags (Primary)** | 10 | 0.1561 | 0.4241 | 0.9995 | 0.1517 | 0.4005 | 0.9995 | Reference |
| **With Imputation Flags (Ablation)** | 20 | 0.1524 | 0.4226 | 0.9995 | 0.1474 | 0.3979 | 0.9995 | -0.0026 (-0.65%) |

*Takeaway*: Including upstream imputation flags yielded an incremental test RMSE reduction of only $0.0026$ index points ($-0.65\%$). The small performance improvement suggests limited incremental predictive value from upstream indicators, supporting the decision to exclude them from primary models.

---

## 15. The Historical 0.4036 Result

Early partial-measurement scripts logged an XGBoost 8/8 Test RMSE of **0.4036** because they used an exploratory 100k subsample (`subsample=100_000`). When trained on the full training set ($N_{\text{train}} = 1,986,397$), the exact same model achieves **0.3282**. All partial-measurement experiments have been harmonized to full-data training, establishing **0.3282** as the canonical 8/8 anchor.

---

## 16. Limitations and Trade-offs

1. **Deterministic Target Construction (Formula Emulation)**: In Regime 1 (8/8), `CCME_Values` is derived deterministically from the 8 inputs. High $R^2$ ($0.9997$) demonstrates accurate formula emulation, not independent environmental forecasting.
2. **Pre-Existing Upstream Median Imputation**: 60.89% of records contain upstream median fill values that cannot be excised from raw data.
3. **China Single-Site Constraint**: All 45,977 China rows originate from station `Hou Bay`. Strict spatial grouping forces all China data into Train (0 out-of-station test samples).
4. **Geographic Imbalance**: England accounts for 75.5% of records; Canada represents only 0.14%.
5. **Target Ceiling Censorship ($CCME = 100.0$)**: 35.92% of samples sit at 100.0. Tree leaf averaging regresses pristine water predictions toward $\approx 99.0 - 99.8$, creating asymmetric negative residuals driven by target censorship.

---

## 17. Testing and Quality Assurance

The repository test suite stands at **130 passed, 0 failed, 67 warnings**:
- `test_split.py` [Parth]: 8 passed (zero row overlap, zero station overlap, country allocation, deterministic persistence).
- `test_preprocess.py` [Parth]: 12 passed (train-only fitting, fit-before-transform safety, partial masking, dependency pruning).
- `test_evaluation.py` [Parth]: 6 passed (regression metrics, residual statistics, group stratification, bootstrap CIs).
- Teammate test suites (`clean`, `features`, `metrics`, `rf`, `xgb`): 104 passed.
- **Total tests directly authored by Parth**: **26 unit tests**.
- **Code Quality**: `ruff check src/ tests/` passes with zero errors.

---

## 18. File-by-File Work Log

| File | What Parth Did | Why It Was Needed | Outcome |
| :--- | :--- | :--- | :--- |
| `src/split.py` | Primary Author | Prevent spatial autocorrelation leakage across stations | Grouped 2.8M rows across 61,689 sites into 70.49/14.76/14.76% splits with zero site overlap |
| `src/preprocess.py` | Primary Author | Prevent statistical snooping and provide dynamic partial masking | Train-only fitted preprocessor; automated dependency pruning |
| `src/evaluate_api.py` | Primary Author | Provide shared data access contract for all teammate models | `get_splits()` serves preprocessed partitions in ~1.5s |
| `src/evaluation.py` | Primary Author | Provide common evaluation metrics and diagnostics | Unified calculation of MAE, RMSE, $R^2$, bootstrap CIs, and subgroup metrics |
| `src/run_evaluation.py` | Primary Author | Orchestrate multi-model evaluation, full-data ablations, and figures | Generated standardized `results/*.csv` and publication figures |
| `tests/test_split.py` | Primary Author | Automated verification of data splitting rules | 8 unit tests verifying disjoint station sets and bin-packing |
| `tests/test_preprocess.py` | Primary Author | Automated verification of preprocessing isolation and masking | 12 unit tests verifying train-only fitting and dependency pruning |
| `tests/test_evaluation.py` | Primary Author | Automated verification of evaluation metrics | 6 unit tests verifying metrics, group slicing, and bootstrap CIs |
| `data/splits/splits.parquet` | Primary Author | Persist deterministic split assignments without disk duplication | Compact 429 KB parquet index mapping all 2,818,006 rows |
| `results/*.csv` | Primary Author | Standardized, machine-readable benchmark tables | Populated canonical benchmark and ablation tables |
| `notebooks/06_evaluation.ipynb` | Primary Author | Consolidation notebook rendering all tables and plots | Fully reproducible interactive evaluation report |

---

## 19. Design Decision Summary Table

| Decision | Preferred Approach | Why | Alternative Rejected | Why Alternative Not Preferred |
| :--- | :--- | :--- | :--- | :--- |
| **Data Splitting** | Group by `Area` with country bin-packing | Prevents spatial autocorrelation leakage across monitoring stations | Random row split | Leaks site-specific water chemistry, producing artificially optimistic test scores |
| **Preprocessing Fitting** | Fit strictly on $X_{\text{train}}$; transform val/test | Prevents statistical snooping into holdout partitions | Global preprocessing before split | Validation and test distributions influence transformation parameters |
| **Station ID (`Area`)** | Grouping variable only; barred from $X$ | Forces models to learn water chemistry rather than station IDs | Include `Area` in $X$ | Overfits to known stations; fails completely on unseen test stations |
| **Target Label (`CCME_WQI`)** | Hard-coded exclusion from $X$ | Prevents direct label leakage | Retain `CCME_WQI` in $X$ | Allows decision trees to invert continuous target trivially |
| **Upstream Imputation Flags** | Exclude from primary model; use in ablation | Prevents models from exploiting England's median fill artifact | Include flags in primary model | Models learn procedural data generation shortcuts instead of water quality |
| **Partial Masking** | Dynamic non-destructive masking in RAM | Keeps underlying split, test set, and preprocessing identical | Separate static files per scenario | Causes file bloat, maintenance overhead, and split inconsistency risks |
| **Engineered Dependencies** | Central dependency registry (`CHEMICAL_FEATURE_DEPENDENCIES`) | Automatically purges derived features when parent is masked | Manual feature lists per scenario | Error-prone; risks indirect reconstruction leakage through derived ratios |
| **Data Access Contract** | Central `get_splits()` API | Guarantees all teammates evaluate models on identical data | Independent teammate split scripts | Produces inconsistent splits, preprocessing, and incomparable results |
| **Partial Training Size** | Full data ($N = 1,986,397$) for all XGBoost runs | Isolates measurement availability as sole experimental variable | 100k subsample for partial runs | Confounders training budget with parameter missingness |
| **Result Tracking** | Explicit `n_train_samples` column in result schemas | Eliminates ambiguity between full-data runs and subsamples | Omitting sample size from tables | Leads to confusion between 100k runs (0.4036) and full runs (0.3282) |
| **Split Persistence** | Lightweight categorical parquet index (429 KB) | Eliminates disk duplication while ensuring exact reproducibility | Saving full train/val/test CSVs | Consumes >1.5 GB of disk space unnecessarily |

---

## 20. Final Contribution Summary

Parthraj Singh Bhati designed and implemented the foundational data architecture that made the Water Quality Prediction project methodologically sound, leak-free, and comparable across all models:
1. **Engineered Data Foundation**: Built the Area-grouped, country-stratified splitting pipeline (`src/split.py`) that eliminated spatial autocorrelation leakage across 2.8 million observations and 61,689 monitoring sites.
2. **Enforced Leakage Governance**: Authored `WaterQualityPreprocessor` (`src/preprocess.py`) and multi-layer validation checks, isolating targets, post-hoc labels, spatial IDs, and upstream-imputation artifacts.
3. **Built Partial-Measurement Framework**: Implemented dynamic masking and the chemical dependency propagation registry, preventing indirect reconstruction leakage when chemical parameters are omitted.
4. **Unified Data Access**: Created `get_splits()` (`src/evaluate_api.py`), serving standardized preprocessed partitions to all teammate models in ~1.5s.
5. **Orchestrated Final Evaluation**: Engineered `src/evaluation.py` and `src/run_evaluation.py`, producing verified benchmarks, harmonized full-data partial measurement sweeps, full-data feature engineering ablations ($-18.1\%$ RMSE), full-data upstream imputation ablations ($-0.65\%$ delta), error analyses, bootstrap confidence intervals, and publication figures.
6. **Maintained Strict QA**: Authored 26 dedicated unit tests, ensuring 100% test pass rates across the 130-test repository suite.
