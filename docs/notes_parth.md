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

## Why I built it that way

1. **Why `CCME_Values` as regression instead of `CCME_WQI` as classification**:
   - The reference benchmark implementation in the dataset repository (`Technical_Validation.ipynb`) explicitly predicts `y = df['CCME_Values']` using regression models (`LinearRegression`, `RandomForestRegressor`, ANN with MSE loss).
   - The project work plan explicitly directs: *"If the target is continuous, classify it as regression. Do not convert a continuous target into classes unless the project documentation explicitly requires that."* Discretizing continuous metrics throws away valuable gradient information and variance.
2. **Why document leakage rather than deleting raw columns now**:
   - Deleting raw columns at stage 1 would disrupt Taj's exploratory data analysis (`02_eda.ipynb`) and data dictionary (`docs/data_dictionary.md`).
   - Establishing documentation first allows the team to explicitly cite the CCME formula and justify feature selection / surrogate modeling in the IEEE report.
3. **Why a unified interface stub in `src/evaluate_api.py`**:
   - To prevent teammates (Himanshu, Jivit, Satya) from building disparate, incompatible data-loading or metric functions. Providing the contract stub now unblocks parallel development.

## How it connects to the rest of the pipeline

- **Consumed by Taj**: Taj reads `docs/target_definition.md` and `src/config.py` to structure cleaning rules in `src/clean.py` and analyze target distributions in `notebooks/02_eda.ipynb`.
- **Consumed by Satya**: Satya implements `regression_metrics` in `src/metrics.py` matching the regression problem type and interfaces with `evaluate()`.
- **Consumed by Himanshu and Jivit**: Both modelers write training routines conforming to `get_splits()` and score via `evaluate()`.
- **Awaiting from Taj**: My next stage (`src/split.py` and `src/preprocess.py`) will execute as soon as Taj delivers `data/processed/clean.csv`.

## What I would do differently

- In future datasets, index targets should ideally be accompanied by raw guideline metadata directly inside the table rather than having to reverse-engineer formula divisors from Jupyter notebook checkpoints.
