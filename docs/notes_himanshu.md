# What I built and why — Himanshu Pandey

| | |
| :--- | :--- |
| **GitHub** | [@14-himanshu](https://github.com/14-himanshu) |
| **Role** | Literature review, feature engineering, Random Forest |
| **Branch prefix** | `feat/himanshu-*` (working branch: `feat/himanshu-lit-review`) |
| **Theory note for the report** | Random Forest: bagging, variance reduction, out-of-bag error |

One page, maintained by Himanshu Pandey. Examiners read this alongside the commit history to
see individual contribution, so write it as you go rather than at the end.

## Files I own

`docs/refs.bib`, `src/features.py`, `src/models_rf.py`, `notebooks/04_random_forest.ipynb`

## What I built

1. **Rigorous Literature Review & Verified BibTeX Repository (`docs/literature_review.md`, `docs/refs.bib`)**:
   - Curated and verified 17 foundational and empirical references across WQI theory (CCME 2001, Horton 1965, Brown et al. 1970, Abbasi & Abbasi 2012, Lumb et al. 2011), machine learning in water science (Ahmed et al. 2019, Bui et al. 2020, Tyralis et al. 2019), ensemble theory (Breiman 2001, Friedman 2001, Chen & Guestrin 2016), feature importance bias (Strobl et al. 2007), game-theoretic explainability (Lundberg & Lee 2017), deep learning foundations (Goodfellow et al. 2016), data leakage prevention (Kaufman et al. 2012, Roberts et al. 2017), and the Figshare dataset record (DOI: 10.6084/m9.figshare.27800394).
   - **Retraction handling:** Removed the retracted Aldhyani et al. (2020) paper (`10.1155/2020/6659314`) and all associated claims.
   - **Bui et al. (2020) correction:** Corrected author 5 to `Nerantzis Kazakis`. Accurately documented that the best performance in the Talar catchment, Iran, belonged to the hybrid Bagging-Random Tree (BA-RT) model ($R^2 = 0.941$, $	ext{RMSE} = 2.71$), rather than standalone Random Forest alone.
   - **Ahmed et al. (2019) correction:** Removed unsupported claims that Random Forest or SVM were benchmarked, accurately reporting their actual evaluation of MLP, RBF, and hybrid Wavelet-ANFIS (WDT-ANFIS) on the Johor River Basin, Malaysia.
   - **CCME WQI Protocol qualification:** Formally distinguished the official aggregative multi-period CCME standard ($\ge 4$ variables sampled $\ge 4$ times) from the dataset's point-in-time single-row adaptation ($F_2 \equiv F_1$). Kept the zero reconstruction error claim explicitly provisional pending local execution of the raw 305 MB dataset and upstream harmonization scripts.
   - **Schema & Feature qualification:** Verified available columns from `docs/data_dictionary.md`, removed unfeasible $BOD:COD$ proposals (no COD in schema), added $\epsilon$-smoothing to candidate stoichiometric ratios, explained the statistical caveats of clipping negative readings, and verified that Temperature is an active formula-contributing parameter (25.0 °C threshold).
   - **Research gap tightening:** Categorized gaps into literature-demonstrated limitations (eclipsing, MDI bias), general methodological risks (leakage, autocorrelation), and project-specific hypotheses (35.92% target ceiling, England upstream median imputation, formula emulation vs. generalization). Structured the two evaluation regimes as proposed comparative designs without claiming pre-experimental results.

2. **Phase 1 Upstream Audit & Synchronization**:
   - Reconciled repository metadata following the merge of Taj's PR #4 and PR #5 into `main`, confirming `data/processed/clean.parquet` (45.3 MB), `pyarrow>=14.0`, and Figshare DOI `10.6084/m9.figshare.27800394`.
   - Merged `main` cleanly into `feat/himanshu-lit-review`.

## Why I built it that way

1. **Why distinguish formula emulation from true environmental prediction**:
   - Many published works report high predictive accuracy ($R^2 > 0.95$) without disclosing that the model is learning to approximate an algebraic equation from its constituent variables. Framing the project around two distinct proposed regimes (Regime 1: formula emulation; Regime 2: feature ablation on non-contributing spatiotemporal covariates) ensures scientific transparency before examiners.
2. **Why purge retracted literature and verify exact authors/models**:
   - Citing retracted papers or misattributing algorithms undermines academic integrity. Every retained score and author was cross-checked against publisher DOI landing pages.
3. **Why qualify negative-value clipping and ratio smoothing**:
   - Adding $\epsilon$ to a denominator prevents division-by-zero errors in code, but does not prove scientific or ecological validity. Similarly, clipping negative sensor drift values to zero alters the empirical noise structure and requires domain justification.

## How it connects to the rest of the pipeline

- **Supplied to Satya**: Fully verified BibTeX entries in `docs/refs.bib` and literature review synthesis for Section II of the IEEE report.
- **Supplied to Jivit**: Theoretical foundation for Random Forest vs. XGBoost comparison and feature ablation design.
- **Next Stage (Awaiting Parth's Split & Pipeline)**: Once Parth merges `src/split.py` and `src/preprocess.py`, I will implement chemical ratios, log transforms, and leakage-safe aggregations in `src/features.py`, followed by Random Forest tuning in `src/models_rf.py`.

## What I would do differently

- Future literature reviews should perform automated DOI validation and retraction database queries prior to drafting text, preventing propagation of misattributed or retracted sources.
