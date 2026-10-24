# What I built and why — Himanshu Pandey

| | |
| :--- | :--- |
| **GitHub** | [@14-himanshu](https://github.com/14-himanshu) |
| **Role** | Literature review, feature engineering, Random Forest |
| **Branch prefix** | `feat/himanshu-*` |
| **Theory note for the report** | Random Forest: bagging, variance reduction, out-of-bag error |

One page, maintained by Himanshu Pandey. Examiners read this alongside the commit history to
see individual contribution, so write it as you go rather than at the end.

## Files I own

`docs/refs.bib`, `src/features.py`, `src/models_rf.py`, `notebooks/04_random_forest.ipynb`

## What I built

1. **Literature Review & Citation Repository (`docs/refs.bib`)**:
   - Curated and validated 14 peer-reviewed and foundational citations covering:
     - Water quality assessment and WQI mathematical formulations (CCME 2001, Brown et al. 1970, Abbasi & Abbasi 2012, Lumb et al. 2011).
     - Applied machine learning benchmarks for water resource modeling (Ahmed et al. 2019, Bui et al. 2020, Aldhyani et al. 2020).
     - Core ensemble theory: Random Forest (Breiman 2001), Gradient Boosting (Friedman 2001), XGBoost (Chen & Guestrin 2016).
     - Explainability and game-theoretic attribution: SHAP (Lundberg & Lee 2017).
     - Methodology, data leakage, and spatiotemporal validation (Kaufman et al. 2012, Roberts et al. 2017, Goodfellow et al. 2016).
   - Produced the comprehensive 6-part Literature Review document detailing WQI derivation, tree-ensemble variance reduction vs. gradient descent in function space, chemical feature engineering, and the distinction between index emulation versus environmental prediction.
2. **Audit & Upstream Verification**:
   - Performed Phase 1 pre-flight audit of Parth's and Taj's deliverables, pinpointing the critical target formula leakage (Spearman rho = -0.917 for Orthophosphate) and upstream median imputation confound in England.

## Why I built it that way

1. **Why distinguish index emulation from environmental prediction**:
   - Many published works report near-perfect R^2 (>0.98) predicting a WQI from the exact chemical constituents that define the index formula, without acknowledging this as closed-form formula recovery rather than true physical prediction. Documenting this distinction upfront establishes the theoretical rigor required for the IEEE report and viva.
2. **Why centralize BibTeX in `docs/refs.bib`**:
   - Satya requires clean, verified citation keys (`@article`, `@inproceedings`, `@techreport`) to compile the Overleaf LaTeX report without missing references or corrupted formatting.

## How it connects to the rest of the pipeline

- **Supplied to Satya**: Complete BibTeX entries in `docs/refs.bib` and structured Literature Review synthesis for Section II of the IEEE report.
- **Supplied to Jivit**: Theoretical foundation for Random Forest vs. XGBoost comparison and feature ablation rationale.
- **Next Stage (Awaiting Parth's Split & Pipeline)**: Once Parth lands `src/split.py` and `src/preprocess.py`, I will implement chemical ratios, log transforms, and leakage-safe aggregations in `src/features.py`, followed by Random Forest tuning in `src/models_rf.py`.

## What I would do differently

- In academic papers on WQI prediction, authors rarely publish their exact cross-validation split indices, making direct experimental replication impossible. In our project, all splits are strictly persisted to `data/splits/` to enforce reproducible benchmarks across Random Forest, XGBoost, and the naive baseline.
