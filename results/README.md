# `results/` — authoritative Phase 1 results

**Cite these files.** Everything here is produced by a single run of
`python -m src.run_evaluation`, which evaluates all models through the shared
`evaluate()` contract on the same persisted split. These are the numbers in the IEEE
report, and `report/make_tables.py` generates the report's tables directly from them.

| File | Contents |
| :--- | :--- |
| `metrics.csv` | Headline comparison: naive baseline, Random Forest, XGBoost — train/val/test MAE, RMSE, R², timings |
| `partial_measurement_results.csv` | **The central result.** All four availability regimes (8/8, 7/8, 6/8, 4/8) × both models |
| `ablation_results.csv` | Engineered features, and upstream-imputation flags |
| `bootstrap_ci_results.csv` | 95 % confidence intervals on the test set |
| `country_results.csv` | Test error by country. Four rows; percentages sum to 100 over all 415,804 test records |
| `country_insample_china.csv` | China diagnostic, kept separate because its single site (Hou Bay) is allocated wholly to training and it has **zero test records** |
| `shap_importance.csv` | Mean absolute SHAP, XGBoost, 8/8 regime |

## Why China is in its own file

China contributes one monitoring site. It cannot be divided across partitions without
breaking the grouping guarantee, so it goes entirely to training. Its figures are
in-sample and are not evidence of generalisation. They were previously appended to
`country_results.csv`, which mixed train rows into a test table and made the percentage
column sum to 102.31. `src/run_evaluation.py` now asserts that column sums to 100.

## Relationship to `reports/`

`reports/` holds **training-time logs** written by individual model modules and
notebooks, plus all figures. Those CSVs cover only the configuration each module
happened to train and are not kept in step with each other. When the two disagree,
`results/` is correct. See `reports/README.md`.
