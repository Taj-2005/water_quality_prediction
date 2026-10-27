# `reports/` — figures and training-time logs

**Do not cite the CSVs here.** They are per-module logs written during training, not a
coordinated evaluation. For the authoritative numbers — and for anything that goes in the
report — use `results/`, which is produced in one pass by `src/run_evaluation.py` through
the shared `evaluate()` contract.

| Path | Written by | Scope |
| :--- | :--- | :--- |
| `figures/*.png` | notebooks 01–06 | **All report figures live here**, referenced by `report/main.tex` via `\graphicspath` |
| `rf_results.csv` | `src/models_rf.py`, `04_random_forest.ipynb` | Random Forest training log — 8/8 configuration only |
| `xgb_results.csv` | `src/models_xgb.py`, `05_xgboost.ipynb` | XGBoost training log — 8/8 configuration only |
| `xgb_ablation.csv` | `src/models_xgb.py` | Feature ablation as run during training |
| `naive_baseline_results.csv` | `03_naive_baseline.ipynb` | Mean and median baselines, val and test |

## Why the two directories were not merged

The files answer different questions. A training log records what one module did on one
configuration; `results/` records the coordinated comparison across every model and
regime. Merging them would hide that distinction, and moving these paths would require
re-executing three notebooks to no benefit.

The practical rule: **figures come from here, numbers come from `results/`.**
