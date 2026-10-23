# Notebooks

Run in numeric order. Each notebook must execute top to bottom without errors, and
every decision is logged **with its reason** in the notebook itself.

| Notebook | Owner | Purpose |
| :--- | :--- | :--- |
| `01_cleaning.ipynb` | Shaik Tajuddin | Cleaning decisions and their reasons; produces `data/processed/clean.csv` |
| `02_eda.ipynb` | Shaik Tajuddin | Distributions, missingness by column and site, correlation heatmap, target skew / class balance, outliers. Ends with the findings that change modelling decisions |
| `03_naive_baseline.ipynb` | Satya Yadav | Mean / median / majority-class floor on the shared split |
| `04_random_forest.ipynb` | Himanshu Pandey | RF training and hyperparameter search on validation only |
| `05_xgboost.ipynb` | Jivit Rana | XGBoost tuning, early stopping, and the feature ablation |
| `06_evaluation.ipynb` | Parthraj Singh Bhati | Final results table for all three models, residual plots or confusion matrix, error by site / time |
| `07_shap.ipynb` | Parthraj Singh Bhati | SHAP and feature importance for the best model |

Notebooks import from `src/` and never redefine loading, splitting or metric logic —
see `src/evaluate_api.py`.
