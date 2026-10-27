# What I built and why — Himanshu Pandey

| | |
| :--- | :--- |
| **GitHub** | [@14-himanshu](https://github.com/14-himanshu) |
| **Role** | Feature engineering, Random Forest, model evaluation |
| **Branch prefix** | `himanshu` |
| **Theory note for the report** | Random Forest: bagging, variance reduction, out-of-bag error |

One page, maintained by Himanshu Pandey. Examiners read this alongside the commit history to
see individual contribution, so write it as you go rather than at the end.

## Files I own

`src/features.py`, `tests/test_features.py`, `src/models_rf.py`, `tests/test_models_rf.py`, `notebooks/04_random_forest.ipynb`

## What I built

1. **Target Formulation & Domain Analysis**:
   - Analyzed the target definition (`CCME_Values`), Ireland guideline thresholds, and mathematical mechanics of the CCME index equation.
   - Formulated the critical experimental distinction between **Regime 1 (Formula Emulation / Benchmark Surrogate)** and **Regime 2 (Independent Environmental Prediction)** to ensure academic honesty and avoid conflating algebraic approximation with ecological prediction.
   - Audited candidate features against the verified dataset schema in `docs/data_dictionary.md` (e.g., removing unfeasible $BOD:COD$ proposals due to absence of COD in the schema, establishing numerical safeguards for near-zero and negative readings).
   - Identified the target boundary condition: 35.92% of samples sit at the maximum ceiling of 100.0, establishing a hypothesis on tree regression behavior at censored boundaries.

2. **Phase 1 Upstream Audit & Synchronization**:
   - Reconciled repository metadata following the merge of Taj's PR #4 and PR #5 into `main`, confirming `data/processed/clean.parquet` (45.3 MB), `pyarrow>=14.0`, and Figshare DOI `10.6084/m9.figshare.27800394`.
   - Merged `main` cleanly into `himanshu`.

3. **Feature Engineering & Leakage Governance (`src/features.py`, `tests/test_features.py`)**:
   - Implemented hydrochemical and stoichiometric domain features: `N_P_ratio`, `Inorganic_N_ratio`, `Organic_N_estimate` (mg/l concentration difference), `Organic_N_fraction` (dimensionless $[0, 1]$), `BOD_DO_ratio`.
   - Implemented monotonic log transforms ($\log(1 + \max(0, x))$) for positively skewed, heavy-tailed water quality distributions (`Ammonia`, `BOD`, `Orthophosphate`, `Nitrate`, `Nitrogen`).
   - Implemented cyclic temporal annual seasonality features: `sin_day`, `cos_day`, and `month` derived from `Date_parsed`.
   - Built a scikit-learn compatible transformer `WaterQualityFeatureEngineer(regime=1 | 2)` adhering to a strict input contract.
   - Programmed programmatic leakage rejection in `validate_regime()`: rejects `CCME_Values`, `CCME_WQI`, `Area`, intermediate formula components (`F1`, `F2`, `F3`, `Scope`, `Frequency`, `Amplitude`, `excursions`, `nse`), and rejects all chemical columns under Regime 2.
   - Optimized dataset I/O in `get_regime_splits()` by caching and reusing the date series, reducing runtime across 2.8M rows from 18.0s to 2.49s.
   - Created 26 rigorous unit tests covering mathematical boundary values, NaN propagation, division-by-zero protection, sub-zero preservation, and partial-measurement masking propagation.

4. **Random Forest Implementation & Validation Protocol (`src/models_rf.py`, `tests/test_models_rf.py`)**:
   - Developed configurable `train()` supporting raw vs. engineered feature sets, regimes 1 and 2, and all four measurement scenarios (`8/8`, `7/8`, `6/8`, `4/8`).
   - Built train-only `CategoricalEncoder`: fits `OneHotEncoder(handle_unknown="ignore")` strictly on `X_train` and applies it to validation and test sets with zero data leakage.
   - Implemented safe evaluation metric fallback (`compute_metrics`): calls `evaluate()`, gracefully calculating `mae`, `rmse`, and `r2` via scikit-learn when Satya's function raises `NotImplementedError`.
   - Enforced validation-only model selection: hyperparameter search evaluates strictly on `X_val`; the holdout test set (`X_test`) is evaluated **strictly once** on the selected model.
   - Supported deterministic subsampling (`subsample=...`, seed=42) for rapid experimentation without memory exhaustion on the 1.98M row training partition.
   - Implemented automated report logging appending run metadata and validation/test metrics to `reports/rf_results.csv`.
   - Created 21 unit tests verifying categorical fitting, tuning isolation, metric fallback, subsampling, and scenario filtering.

5. **Diagnostic Analysis & Visualizations (`notebooks/04_random_forest.ipynb`)**:
   - Explains Bagging (Bootstrap Aggregation) and variance reduction mathematically (Breiman, 2001).
   - Trained baseline model (`subsample=100_000`, 100 trees, depth 15, leaf 5):
     - Validation: MAE = 0.4031, RMSE = 0.9075, R² = 0.9976
     - Out-of-Bag (OOB): R² = 0.9979
     - Holdout Test (single evaluation): MAE = 0.4078, RMSE = 0.9197, R² = 0.9976
   - Conducted systematic validation hyperparameter sweep across tree depth and regularization (`reports/figures/rf_hyperparameter_tuning.png`).
   - Executed systematic 8-experiment ablation matrix comparing raw vs. engineered features across 8/8, 7/8, 6/8, and 4/8 scenarios (`reports/figures/rf_ablation_comparison.png`).
   - Diagnosed feature importance: compared Mean Decrease in Impurity (MDI) with validation Permutation Feature Importance, explicitly citing and demonstrating the continuous/collinear feature bias proved by Strobl et al. (2007) (`reports/figures/rf_feature_importance_comparison.png`).
   - Conducted partitioned residual analysis separating samples below ceiling (`CCME < 100.0`) from samples at the ceiling (`CCME == 100.0`), demonstrating that the positive residual spike at 100.0 is an intrinsic artifact of regressing on censored target distributions (`reports/figures/rf_residuals_analysis.png`).

## Why I built it that way

1. **Why distinguish formula emulation from true environmental prediction**:
   - In Regime 1 (8/8 scenario), models achieve $R^2 \approx 0.998$ and $\text{RMSE} < 1.0$ because they are learning to approximate a closed-form algebraic formula from its 8 input variables. Presenting this as "high predictive accuracy for water quality forecasting" is scientifically misleading. Explicitly labeling Regime 1 as a **surrogate/emulator benchmark** and isolating Regime 2 as **independent prediction** ensures academic honesty.
2. **Why fit categorical encodings strictly on training data**:
   - Categorical encodings (e.g. `Country`, `Waterbody Type`) must not encode frequencies or categories present only in validation or test data. Setting `handle_unknown="ignore"` ensures that novel test categories map safely to all zeros rather than crashing or leaking label information.
3. **Why isolate hyperparameter selection to validation only**:
   - Selecting tree count, depth, or leaf size using test performance produces optimistically biased results. In `src/models_rf.py`, candidate models are compared exclusively on `X_val`, and `X_test` is evaluated exactly once after final model selection.
4. **Why evaluate MDI alongside Permutation Importance (Strobl et al., 2007)**:
   - Gini/MDI importance favors features with high cardinality and continuous variation, and arbitrarily splits importance between correlated inputs (`Nitrogen` and `Nitrate`). Permutation importance breaks feature-target association directly on validation data, providing an unbiased estimate of true predictive degradation.
5. **Why partition residuals by the 35.92% target ceiling**:
   - 35.92% of all samples sit at $\text{CCME} = 100.0$ (samples satisfying all 8 guidelines). Because regression tree leaves compute local conditional means ($y_i \le 100.0 \implies \hat{y} \le 100.0$), predictions for pristine water regress toward the mean ($\approx 98-99$), generating an asymmetric positive residual cluster ($y - \hat{y} > 0$). Explaining this mathematically prevents examiners from mistaking target censorship for model deficiency.

## How it connects to the rest of the pipeline

- **Supplied to Jivit**: Established the baseline Random Forest implementation and validation results in `reports/rf_results.csv`, against which XGBoost will be benchmarked in `notebooks/05_xgboost.ipynb`.
- **Supplied to Parth**: Logged standard metrics (`mae`, `rmse`, `r2`) and generated figures in `reports/figures/` for consolidation into the final multi-model comparison table in `notebooks/06_evaluation.ipynb`.
- **Supplied to Satya**: Verified metric schema compatibility (`compute_metrics` matches `regression_metrics` specifications) and provided empirical results, figures, and discussion sections for the IEEE report.

## Experimental Results Summary

*All runs executed on the shared, Area-grouped split (`data/splits/splits.parquet`). Validation metrics evaluated on 415,805 validation samples. Test metrics evaluated strictly once on 415,804 holdout test samples for the final selected baseline model.*

| Experiment / Configuration | Feature Set | Features ($p$) | Val MAE | Val RMSE | Val R² | Test MAE | Test RMSE | Test R² | Notes |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **8/8 Surrogate Benchmark** | Engineered | 38 | **0.4031** | **0.9075** | **0.9976** | **0.4078** | **0.9197** | **0.9976** | Primary surrogate baseline (100k samples, 100 trees, depth 15) |
| **8/8 Surrogate Baseline** | Raw | 25 | 1.1243 | 1.9665 | 0.9888 | — | — | — | Raw Ireland inputs + metadata (50k samples, 50 trees, depth 15) |
| **7/8 Partial Scenario** | Engineered | 35 | **5.4397** | **8.2888** | **0.8004** | — | — | — | Missing Orthophosphate & dependencies (50k samples) |
| **7/8 Partial Scenario** | Raw | 24 | 5.8753 | 8.5565 | 0.7873 | — | — | — | Missing Orthophosphate (50k samples) |
| **6/8 Moderate Missingness** | Engineered | 32 | **5.7553** | **8.6606** | **0.7821** | — | — | — | Missing Orthophosphate & BOD (50k samples) |
| **6/8 Moderate Missingness** | Raw | 23 | 6.1816 | 8.8673 | 0.7716 | — | — | — | Missing Orthophosphate & BOD (50k samples) |
| **4/8 Severe Missingness** | Engineered | 25 | 9.3248 | 12.8412 | 0.5210 | — | — | — | pH, DO, Temp, Nitrate + log_Nitrate + metadata + temporal (50k) |
| **4/8 Severe Missingness** | Raw | 21 | **9.2564** | **12.7329** | **0.5290** | — | — | — | pH, DO, Temp, Nitrate + metadata (50k samples) |
| **Full-Dataset Baseline** | Engineered | 38 | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* | Full 1.98M row training reserved for final consolidation |

## Limitations & Outstanding Work

1. **Full Dataset Training Cost**: Initial verification and diagnostic plots were trained on 50,000 to 100,000 subsamples. Full training on all 1,979,584 training rows with 100+ trees takes ~10–15 minutes and 8–16 GB RAM; this final fit will be executed once Jivit completes the XGBoost setup for direct parity.
2. **Regime 2 Spatiotemporal Generalization**: The Regime 2 pipeline is implemented and tested in code (`regime=2`), strictly excluding chemical features. Further exploration of spatial autocorrelation across monitoring stations (`Area`) can be incorporated into final comparative discussions.
