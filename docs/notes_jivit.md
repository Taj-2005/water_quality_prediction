# What I built and why — Jivit Rana

| | |
| :--- | :--- |
| **GitHub** | [@Jivit87](https://github.com/Jivit87) |
| **Role** | XGBoost, ablation |
| **Branch prefix** | `feat/jivit-*` |
| **Theory note for the report** | Gradient boosting: loss function, additive trees, shrinkage, regularization |

One page, maintained by Jivit Rana. Examiners read this alongside the commit history to
see individual contribution, so write it as you go rather than at the end.

## Files I own

`src/models_xgb.py`, `notebooks/05_xgboost.ipynb`, `reports/xgb_results.csv`

## What I built

- `src/models_xgb.py`: loads the shared split, trains XGBoost with early stopping, optionally
  tunes hyperparameters with Optuna, scores the test set once and appends to
  `reports/xgb_results.csv`. `train()` has the same shape as Himanshu's Random Forest `train()`,
  so the Phase 2 MLP and Parth's final evaluation can treat every model the same way.
- `run_ablation()`: trains XGBoost on raw vs engineered features with the same split and
  seed, and writes `reports/xgb_ablation.csv`.
- `notebooks/05_xgboost.ipynb`: tuned run, feature importance, the ablation, and the
  Random Forest vs XGBoost comparison (full training data, about 20 minutes to run).
- `tests/test_models_xgb.py`: checks categories line up across splits, results are repeatable,
  the CSV is written, tuning only sees the validation set, and the ablation runs both arms.

Full-data results (Regime 1, 8/8 measurements, about 2M training rows):

| Model / features | Test MAE | Test RMSE | Test R2 |
| :--- | ---: | ---: | ---: |
| XGBoost, raw (10 features) | 0.152 | 0.400 | 0.99954 |
| XGBoost, engineered (23 features) | 0.100 | 0.328 | 0.99969 |
| Random Forest, engineered (100k train rows) | 0.408 | 0.920 | 0.99757 |

## Why I built it that way

- **Validation only for every choice.** Optuna scores trials on validation RMSE and early
  stopping watches the validation set. The test set is scored once, after everything is frozen.
- **Optuna over a grid.** There are seven hyperparameters; a smart sampler (TPE) finds good
  values in 15 trials, where a grid would need hundreds of fits on 2M rows. Trials fit on
  300k rows for speed; the winner is refitted on all training rows.
- **Native categories.** `Country` and `Waterbody Type` go in as categories (categories are
  learned from train only) instead of one-hot columns, which keeps the feature count small.
- **Same settings in both ablation arms.** Only the feature set changes, so a score
  difference comes from the features alone.
- **Rejected:** a grid search (too slow), tuning on the test set (leakage), and a random
  split of rows (the shared split groups by site, so I use it as given).

## How it connects to the rest of the pipeline

- **I consume:** `get_splits()` (Parth), the cleaned data (Taj), engineered features
  (Himanshu), and the metrics (Satya, via `evaluate()`; until that stub is finished I fall
  back to the scikit-learn metrics Himanshu's code already uses).
- **They consume:** `reports/xgb_results.csv` for Parth's final results table and SHAP
  notebook, and the ablation table and RF-vs-XGBoost comparison for Satya's report.
- **Regime 1 caveat:** the 8 chemical inputs define `CCME_Values`, so these scores show how
  well trees imitate the CCME formula. They are not independent water-quality prediction.

## Theory note: gradient boosting

Gradient boosting builds a model as a sum of small trees, added one at a time:
`F_m(x) = F_(m-1)(x) + learning_rate * h_m(x)`.

- **Loss function.** We minimise squared error, `L = (y - F(x))^2`. Each new tree `h_m` is
  fitted to the negative gradient of the loss, which for squared error is just the current
  residual `y - F_(m-1)(x)`. So every tree learns what the ensemble still gets wrong.
- **Additive trees.** The final prediction is the sum of all trees. Shallow trees are weak
  alone, but many of them correct each other's mistakes step by step.
- **Shrinkage.** The learning rate (about 0.055 in my tuned model) scales each tree down, so no
  single tree changes the answer much. A small rate needs more trees but generalises better.
- **Regularization.** XGBoost penalises complexity with `reg_lambda` (L2) and `reg_alpha`
  (L1) on leaf values, `min_child_weight` limits tiny leaves, and `subsample` /
  `colsample_bytree` train each tree on random rows and columns. It also uses the
  second derivative of the loss for better split decisions (Chen and Guestrin, 2016).
- **Why XGBoost beat Random Forest here.** A forest averages independent deep trees, which
  reduces variance but each tree must approximate the whole CCME formula alone. Boosting
  removes error step by step, which suits a smooth, low-noise target. Caveat: the forest
  used 100k training rows and XGBoost used all 2M, so part of the gap is data size.

## What I would do differently

- The best models used about 2,000 trees, right at my cap, so a higher cap (and a small
  extra gain) is worth trying.
- Run the Random Forest on the same full training set for a like-for-like comparison.
- Run Regime 2 (non-chemical covariates), which is the honest independent-prediction test.
