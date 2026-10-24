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

1. **Rigorous Literature Review & Verified BibTeX Repository (`docs/literature_review.md`, `docs/refs.bib`)**:
   - Curated and verified 17 foundational and empirical references across WQI theory (CCME 2001, Horton 1965, Brown et al. 1970, Abbasi & Abbasi 2012, Lumb et al. 2011), machine learning applications in water science (Ahmed et al. 2019, Bui et al. 2020, Aldhyani et al. 2020, Tyralis et al. 2019), ensemble theory (Breiman 2001, Friedman 2001, Chen & Guestrin 2016), feature importance bias (Strobl et al. 2007), game-theoretic explainability (Lundberg & Lee 2017), deep learning fundamentals (Goodfellow et al. 2016), and data leakage prevention (Kaufman et al. 2012, Roberts et al. 2017).
   - Audited and corrected all bibliographic metadata (fixed Aldhyani et al. DOI and journal from a misattributed pesticide paper to *Applied Bionics and Biomechanics*, corrected Bui et al. and Ahmed et al. co-authors, and added missing Horton and Strobl citations).
   - Removed unverified empirical claims, replacing them with accurate descriptions of model architectures (e.g., Bui et al.'s bagging-tree ensembles in Iran, Ahmed et al.'s WDT-ANFIS in Malaysia, and Aldhyani et al.'s NARNET/LSTM vs. SVM on Indian rivers).
   - Harmonized feature engineering proposals with the actual 14-column dataset schema (removed non-existent $BOD:COD$ ratio, added $\epsilon$-smoothing for zero denominators in nutrient ratios, and verified that Temperature is an active formula input).
   - Formally clarified the distinction between the standard multi-period CCME protocol and the dataset-specific point-in-time ($F_2 \equiv F_1$) row adaptation.

2. **Phase 1 Upstream Pre-Flight Audit**:
   - Inspected upstream branches (`origin/feat/parth`, `origin/feat/taj-cleaning`, `origin/feat/taj-eda`).
   - Identified unmerged PR blockers on Taj's branch and confirmed that Parth's split and preprocessing scripts remain unimplemented stubs, preventing premature model implementation.

## Why I built it that way

1. **Why distinguish formula emulation from true environmental prediction**:
   - Many published works claim high predictive accuracy ($R^2 > 0.95$) without disclosing that the model is merely learning to approximate a closed-form algebraic formula from its contributing inputs. Framing the project around two distinct evaluation regimes (Regime 1: formula emulation; Regime 2: feature ablation on non-contributing spatiotemporal covariates) ensures scientific honesty for the IEEE report and viva.
2. **Why enforce strict schema and numerical safeguards in feature proposals**:
   - Environmental sensor data contains values near zero and isolated negative readings from sensor baseline drift (e.g., Orthophosphate min $-0.004$). Defining explicit clipping ($\max(0, x)$) and smoothing denominators ($x / (y + \epsilon)$) prevents fatal `ZeroDivisionError` and `NaN` propagation during model training.
3. **Why centralize verified BibTeX in `docs/refs.bib`**:
   - Satya requires verified citation keys to compile the IEEE LaTeX report without citation compilation warnings or inaccurate citations during examination.

## How it connects to the rest of the pipeline

- **Supplied to Satya**: Verified BibTeX entries in `docs/refs.bib` and literature review synthesis for Section II of the IEEE report.
- **Supplied to Jivit**: Theoretical foundation for Random Forest vs. XGBoost comparison and feature ablation design.
- **Next Stage (Awaiting Parth's Split & Pipeline)**: Once Parth merges `src/split.py` and `src/preprocess.py`, I will implement chemical ratios, log transforms, and leakage-safe aggregations in `src/features.py`, followed by Random Forest tuning in `src/models_rf.py`.

## What I would do differently

- In academic literature on WQI prediction, authors rarely publish their exact cross-validation split indices, making direct experimental replication impossible. In our project, all splits are strictly persisted to `data/splits/` to enforce reproducible benchmarks across Random Forest, XGBoost, and the naive baseline.
