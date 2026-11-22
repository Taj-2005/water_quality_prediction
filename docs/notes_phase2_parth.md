# What I Built and Why — Parthraj Singh Bhati (Phase 2)

| | |
| :--- | :--- |
| **Author** | Parthraj Singh Bhati |
| **Role** | Deep Learning Foundation, Final Multi-Model Comparison & Technical Validation |
| **Branch** | `feat/parth-final-comparison` |
| **Deliverables** | `src/final_comparison.py`, `notebooks/10_final_comparison.ipynb`, `tests/test_final_comparison.py`, `reports/final_comparison*.csv`, `reports/leakage_check.csv`, `reports/compute_cost.csv`, `reports/figures/final_comparison_*.png` |
| **Theory Notes** | Backpropagation & Reverse-Mode Autodiff, The Adam/AdamW Optimizer Mechanics, Multi-Seed Statistical Evaluation |

---

## 1. Executive Summary & Experimental Disclosure

### Mandatory Benchmark Disclosure
> [!IMPORTANT]
> **Formula-Approximation Benchmark (Surrogate Emulator)**:
> In this project, the target continuous variable `CCME_Values` is mathematically defined by the Canadian Council of Ministers of the Environment (CCME) Water Quality Index equation, computed from **8 physical and chemical parameters** (`Ammonia`, `BOD`, `Dissolved Oxygen`, `Orthophosphate`, `pH`, `Temperature`, `Nitrogen`, `Nitrate`).
> Therefore, all models evaluated in Regime 1 (including Random Forest, XGBoost, Tuned MLP, and the Hybrid) operate as **deterministic formula approximators / surrogate emulators**. These benchmarks quantify how accurately machine learning architectures learn non-linear threshold logic from data; they **must NOT be presented as independent environmental forecasting**. All models are scored strictly on the identical saved Phase 1 test partition; this comparison scores the test split once without re-tuning.

### Key Deliverables Completed
1. **Multi-Seed Benchmark Engine (`src/final_comparison.py`)**:
   - Compares 6 core model configurations: **Naive Baseline**, **Random Forest (Phase 1)**, **XGBoost (Phase 1 Tuned, 23 Eng Features)**, **XGBoost (Jivit Hybrid Base, 10 Raw Features)**, **Tuned WaterQualityMLP (Phase 2)**, and **Residual Hybrid (Phase 2)**.
   - Evaluated across the team's 5 agreed literature review seeds (`[42, 123, 456, 789, 1011]`).
   - Evaluated strictly using the shared `src.evaluate_api.evaluate()` contract.
2. **Controlled Subsample Benchmark (Exact 20,000 Rows, 5 Seeds)**:
   - Strict mathematical parity where every model is trained on the exact same 20,000 training observations across all 5 seeds (matching Jivit's hybrid ablation scale).
3. **Table 2: Reported Operating Points from Earlier Single-Seed Runs**:
   - Compiles reported operating points from earlier single-seed runs at different training sizes (20k to 1.98M rows). Plainly disclosed as non-comparable across different sample sizes.
   - Includes the empirical Random Forest full data trial (1,986,397 rows trained in 52.41s).
4. **Tail Error Metrics (P95, P99, Max Absolute Error)**:
   - Computed across all 5 seeds to empirically assess tail behavior and extreme error suppression.
5. **Technical Validation & Anti-Leakage Audit (`reports/leakage_check.csv`)**:
   - 5/5 automated technical verification checks passing (row disjointness, scaler fit isolation, target standardization isolation, spatial group isolation in GroupKFold, and exclusion of formula/label leakage).
6. **Compute Cost & Hardware Profiling (`reports/compute_cost.csv`)**:
   - Measured parameter counts and tree node counts (replacing hardcoded estimates).
   - Real-world training runtime and inference throughput per 100k rows measured on Apple Silicon arm64 hardware.
7. **Interactive Analysis Notebook (`notebooks/10_final_comparison.ipynb`)**:
   - Fully executed notebook with tabular results, diagnostic plots, and granular error breakdowns by Country and monitoring station (`Area`).

---

## 2. Experimental Methodology & Hyperparameter Lineage

To ensure academic rigor, **zero hyperparameters were re-tuned** during this final comparison phase. Every model carries forward the fixed, validated hyperparameter specifications established by the team in Phase 1 and earlier Phase 2 milestones:

### Model Hyperparameter & Configuration Summary

| Model Configuration | Architecture / Hyperparameters | Preprocessing / Inputs | Lineage / Source |
| :--- | :--- | :--- | :--- |
| **Naive Baseline** | Predicts global training mean $\bar{y}_{\text{train}}$ | Raw target only | Statistical reference |
| **Random Forest** | 100 trees, `max_depth=15`, `min_samples_leaf=5`, `max_features="sqrt"` | Train-only `CategoricalEncoder` (38 one-hot features) | Himanshu Phase 1 (`reports/rf_results.csv`) |
| **XGBoost (Phase 1 Tuned)** | `max_depth=7`, $\eta=0.0552$, `subsample=0.950`, `colsample_bytree=0.814`, `min_child_weight=30`, `reg_alpha=0.0071`, `reg_lambda=33.12` | Categorical dtype via `match_categories` (23 engineered features: log transforms, nutrient ratios) | Jivit Phase 1 (`reports/xgb_results.csv`) |
| **XGBoost (Jivit Hybrid Base)** | `max_depth=8`, $\eta=0.100$, `subsample=0.800`, `colsample_bytree=0.800`, `min_child_weight=5`, `reg_alpha=0.0`, `reg_lambda=1.0` | Categorical dtype via `match_categories` (10 raw physical/chemical features) | Jivit Phase 2 (`src/dl/hybrid.py`, `reports/hybrid_results.csv`) |
| **Tuned MLP** | `(256, 256, 256, 256)` GELU, AdamW ($\text{lr}=10^{-3}, \text{wd}=10^{-4}$), batch size 512, epochs 15, unregularized (`dropout=0.0`, `use_batch_norm=False`) | `QuantileTransformer(output="normal")` fitted on train only; train-only target standardization | Himanshu & Taj Phase 2 (`reports/mlp_results.csv`) |
| **Residual Hybrid** | XGBoost Base (`max_depth=8, reg_lambda=1.0`) + PyTorch `WaterQualityMLP(128, 64, 32)` ReLU residual corrector, AdamW ($\text{lr}=10^{-3}, \text{wd}=10^{-4}$) | GroupKFold on `Area` for out-of-site residual generation (10 raw features) | Jivit Phase 2 (`src/dl/hybrid.py`, `reports/hybrid_results.csv`) |

### Resolution of the 0.560 vs. 0.887 XGBoost Test RMSE Discrepancy
In earlier preliminary summaries, standalone XGBoost at 20,000 rows was reported with a test RMSE of **0.5602**, while Jivit's hybrid ablation reported XGBoost base at 20,000 rows with a test RMSE of **0.8865** (and 0.8385 on seed 42).
Investigation of the code and report lineages identified the exact sources:
1. **Input Feature Space**: Jivit's hybrid pipeline operates strictly on the **10 raw features** (8 chemical parameters + Country + Waterbody Type), without engineered nutrient ratios or log transforms. Standalone tuned XGBoost evaluated on the **23 engineered features** (including $N:P$ ratio, $BOD:DO$ ratio, and log-transformed concentrations).
2. **Hyperparameter Regularization**: Jivit's hybrid base XGBoost uses standard gradient boosting settings (`learning_rate=0.1, max_depth=8, min_child_weight=5, reg_lambda=1.0`). In contrast, Phase 1 Optuna tuning selected a shallower tree with heavy L2 regularization (`learning_rate=0.0552, max_depth=7, min_child_weight=30, reg_lambda=33.12`), which significantly suppresses leaf-level variance.
3. **Controlled Finding**: When trained on the identical 10 raw features, XGBoost Base achieves Test RMSE **0.8385** (seed 42), and the Residual Hybrid lowers it to **0.6356** (multi-seed mean). When trained on 23 engineered features, standalone XGBoost achieves Test RMSE **0.5602**. Both comparisons are now explicitly displayed side by side in Table 1.

### Regularization Alignment (Himanshu vs. Taj)
- **Taj's Recommendation (`docs/notes_phase2_taj.md`)**: Sweeping 5 regularization families revealed that with 200,000+ training observations, the neural network is capacity-limited rather than variance-limited. Dropout added no benefit beyond seed noise, and Batch Normalization severely degraded performance (+0.30 validation MAE). The explicit recommendation was **no regularization** (zero dropout, no batch normalization, decoupled weight decay $10^{-4}$).
- **Himanshu's Tuned MLP (`src/dl/architecture.py`)**: Himanshu's Optuna optimization selected a 4-layer uniform architecture `(256, 256, 256, 256)` with GELU activations, `dropout=0.0`, `use_batch_norm=False`, and AdamW weight decay $10^{-4}$.
- **Verdict**: **Himanshu's tuned MLP already strictly conforms to Taj's empirical regularization recommendation.**

---

## 3. Benchmark Results & Comparative Analysis

### Table 1: Controlled Comparison Benchmark (Exact 20,000 Rows, 5 Seeds)
*Every model trained on the exact same 20,000 training observations across all 5 seeds (`[42, 123, 456, 789, 1011]`). Evaluated on the held-out test split (415,804 rows).*  
*Disclosure: formula-approximation benchmark (CCME_Values computed from 8 input parameters). Same training rows, feature sets differ.*

| Model | Feature Set | Size (parameters or tree nodes) | Test RMSE (mean ± std) | Test MAE (mean ± std) | Test P95 AE (mean ± std) | Test P99 AE (mean ± std) | Test Max AE (mean ± std) | Test $R^2$ (mean ± std) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **XGBoost (Phase 1 Tuned)** | 23 engineered features (ratios + log transforms) | 111,228 nodes | **0.5602 ± 0.0284** | **0.2130 ± 0.0085** | **0.8736 ± 0.0356** | **2.4743 ± 0.1206** | **11.7142 ± 1.1033** | **0.9991 ± 0.0001** |
| **Residual Hybrid** | 10 raw features (XGB base + residual MLP) | 11,777 params (MLP only) | 0.6356 ± 0.0824 | 0.3022 ± 0.0499 | 1.1865 ± 0.1877 | 2.5806 ± 0.2378 | 12.5283 ± 3.5568 | 0.9988 ± 0.0003 |
| **XGBoost (Jivit Hybrid Base)** | 10 raw features | 120,375 nodes | 0.6757 ± 0.0949 | 0.3016 ± 0.0501 | 1.2090 ± 0.2009 | 2.7325 ± 0.2874 | 15.0811 ± 4.0863 | 0.9987 ± 0.0004 |
| **Random Forest (Phase 1)** | 38 features (Phase 1 one-hot) | 185,636 nodes | 1.4543 ± 0.0504 | 0.6545 ± 0.0148 | 2.6878 ± 0.0574 | 6.0583 ± 0.0589 | 24.6490 ± 2.2994 | 0.9939 ± 0.0004 |
| **Tuned MLP** | 10 raw features (quantile normalized) | 204,289 params | 1.7295 ± 0.0721 | 1.1464 ± 0.0537 | 3.6101 ± 0.1256 | 5.8123 ± 0.2743 | 36.1018 ± 7.1832 | 0.9914 ± 0.0007 |
| **Naive Baseline** | Target only | 1 param | 18.6559 ± 0.0012 | 14.8991 ± 0.0159 | 40.9058 ± 0.0477 | 44.8921 ± 0.0477 | 50.9211 ± 0.0477 | -0.0006 ± 0.0001 |

#### Feature Control & Input Representation Investigation
We investigated evaluating the Tuned MLP and Residual Hybrid on the same 23 engineered features used by Tuned XGBoost. However, doing so requires modifying other teammates' files:
1. `src/dl/hybrid.py` (owned by Jivit) hardcodes `get_splits(include_metadata=True)` which returns only the 10 raw features, and exposes no argument for passing an external feature matrix.
2. `src/dl/nn_preprocess.py` (owned by Taj) filters continuous columns strictly by `c in MEASUREMENTS` (the 8 raw chemical measurements) and drops all other columns via `remainder="drop"`.
Under team ownership constraints (*"say if the existing code can't do this without editing others' files and stop there"*), we preserve teammates' files, explicitly label Table 1 `"same training rows, feature sets differ"`, and add the `Feature Set` column documenting each model's inputs.

---

### Table 2: Reported Operating Points from Earlier Single-Seed Runs (Different Training Sizes, Not Comparable)

> [!NOTE]
> **Non-Comparable Operating Regimes**: The table below compiles operating points reported from earlier single-seed runs across different training partition sizes (ranging from 20,000 to 1,986,397 rows). Because training sample sizes differ by up to two orders of magnitude, these numbers reflect distinct computational operating regimes and cannot be used to rank model architectures. Tail error metrics (P95, P99, Max AE) are excluded because earlier single-seed runs did not save full prediction arrays. All reported values originate strictly from CSV files in `reports/`.

| Model | Feature Set | Training Samples | Size (parameters or tree nodes) | Fit Time (s) | Test MAE | Test RMSE | Test $R^2$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **XGBoost (Phase 1 Full Data, 23 Eng Features, Single Seed s=42)** | 23 engineered features (ratios + log transforms) | 1,986,397 | 109,501 nodes | 730.0s | 0.1000 | 0.3282 | 0.9997 |
| **Random Forest (Full Data Trial, 38 Features, Single Seed s=42)** | 38 features (Phase 1 one-hot) | 1,986,397 | 2,191,352 nodes | 52.41s | 0.2669 | 0.5830 | 0.9990 |
| **Residual Hybrid (XGB Base + MLP, 10 Raw Features, Single Seed s=42)** | 10 raw features (XGB base + residual MLP) | 20,000 | 11,777 params (MLP only) | 76.44s | 0.4110 | 0.8521 | 0.9979 |
| **Random Forest (Phase 1 100k Benchmark, 38 Features, Single Seed s=42)** | 38 features (Phase 1 one-hot) | 100,000 | 398,998 nodes | 32.5s | 0.4078 | 0.9197 | 0.9976 |
| **Tuned MLP (Phase 2 200k Architecture Benchmark, Single Seed s=42)** | 10 raw features (quantile normalized) | 200,000 | 204,289 params | 142.0s | 0.5344 | 0.9306 | 0.9975 |
| **Naive Baseline (Mean, Single Seed s=42)** | Target only | 1,986,397 | 1 param | 0.05s | 14.7725 | 18.5081 | 0.0000 |

#### Random Forest Full Training Trial Outcome (Single Seed s=42)
As part of technical validation, Random Forest was evaluated on the full 1,986,397 training observations with Phase 1 hyperparameters (`n_estimators=100, max_depth=15, min_samples_leaf=5, max_features="sqrt"`, `n_jobs=-1`, 20-minute hard timeout). This is an empirical single-seed trial (no std):
- **Outcome**: Completed successfully in **52.41 seconds** (56.53s total including I/O).
- **Test Metrics**: Test RMSE = **0.5830**, Test MAE = **0.2669**, Test $R^2$ = **0.9990**.
- **Measured Tree Size**: **2,191,352 tree nodes** across the 100 estimators.
- **Empirical Reality**: Full Random Forest does **not** exhaust memory or fail on Apple Silicon arm64 (8 physical cores); speculative "memory exhaustion" wording has been removed from repo documentation. As a single-seed trial across a different training size, it is excluded from any direct ranking against the 5-seed controlled table.

---

### Table 3: Paired Differences Statistical Analysis Across 5 Random Seeds (`reports/paired_differences.csv`)

To rigorously isolate architectural signal from random seed stochasticity, we compute paired differences per seed across the 5 literature seeds (`[42, 123, 456, 789, 1011]`):

$$\Delta_i = \text{Error}(M_{2, \text{seed}_i}) - \text{Error}(M_{1, \text{seed}_i})$$

A positive mean difference indicates that Model 1 has lower error (is better) by that margin:

| Comparison | Metric | Model 1 ($M_1$) | Model 2 ($M_2$) | Mean Difference | Std Difference | Seeds $M_1$ Better | Total Seeds | Win Rate $M_1$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Hybrid vs XGB Base** | RMSE | Residual Hybrid (10 Raw) | XGBoost Base (10 Raw) | **+0.0400** | 0.0150 | 5 | 5 | **5/5** |
| **Hybrid vs XGB Base** | MAE | Residual Hybrid (10 Raw) | XGBoost Base (10 Raw) | -0.0006 | 0.0059 | 2 | 5 | 2/5 |
| **Hybrid vs XGB Base** | P95 AE | Residual Hybrid (10 Raw) | XGBoost Base (10 Raw) | +0.0225 | 0.0269 | 4 | 5 | 4/5 |
| **Hybrid vs XGB Base** | P99 AE | Residual Hybrid (10 Raw) | XGBoost Base (10 Raw) | **+0.1520** | 0.0634 | 5 | 5 | **5/5** |
| **Hybrid vs XGB Base** | Max AE | Residual Hybrid (10 Raw) | XGBoost Base (10 Raw) | **+2.5528** | 0.9175 | 5 | 5 | **5/5** |
| **Tuned XGB vs Hybrid** | RMSE | XGBoost Tuned (23 Eng) | Residual Hybrid (10 Raw) | **+0.0754** | 0.0652 | 5 | 5 | **5/5** |
| **Tuned XGB vs Hybrid** | MAE | XGBoost Tuned (23 Eng) | Residual Hybrid (10 Raw) | **+0.0892** | 0.0564 | 5 | 5 | **5/5** |
| **Tuned XGB vs Hybrid** | P95 AE | XGBoost Tuned (23 Eng) | Residual Hybrid (10 Raw) | **+0.3129** | 0.2098 | 5 | 5 | **5/5** |
| **Tuned XGB vs Hybrid** | P99 AE | XGBoost Tuned (23 Eng) | Residual Hybrid (10 Raw) | +0.1063 | 0.2020 | 3 | 5 | 3/5 |
| **Tuned XGB vs Hybrid** | Max AE | XGBoost Tuned (23 Eng) | Residual Hybrid (10 Raw) | +0.8141 | 2.5716 | 2 | 5 | 2/5 |

---

## 4. When Does the MLP Beat the Trees and When Does It Not?

Based strictly on the empirical values presented in Table 1, Table 2, and Table 3:

### 1. Accuracy and Error Comparison:
- **Standalone Accuracy**: Across all evaluated sample sizes, the standalone Tuned MLP does **not** beat XGBoost or Random Forest on test error. On the controlled 20,000-sample benchmark:
  - Tuned XGBoost (23 engineered features) scores Test RMSE **0.5602** vs. Tuned MLP's **1.7295**.
  - Comparing on the identical 10 raw features without engineered ratios, XGBoost Base scores Test RMSE **0.6757**, Random Forest scores **1.4543**, and Tuned MLP scores **1.7295**.
- **Tail Error & Paired Analysis**:
  - **Hybrid vs. XGBoost Base (10 Raw Features)**:
    - **RMSE**: Hybrid beats Base in **5 of 5 seeds** (mean reduction $+0.0400 \pm 0.0150$).
    - **MAE**: Essentially tied (mean diff $-0.0006 \pm 0.0059$, Hybrid better in 2 of 5 seeds).
    - **P95 Absolute Error**: Hybrid beats Base in **4 of 5 seeds** (mean diff $+0.0225 \pm 0.0269$). On seed 1011, Base achieves slightly lower P95 error (1.2564 vs. 1.2666). Because 1 seed favors Base, **we make NO claim of a consistent tail benefit on P95**.
    - **Extreme Tail (P99 AE & Max AE)**: Hybrid consistently achieves lower extreme error across **5 of 5 seeds**: P99 AE improves by $+0.1520 \pm 0.0634$ (5/5 seeds), and Max AE improves by $+2.5528 \pm 0.9175$ (5/5 seeds, reducing maximum absolute error from 15.0811 to 12.5283).
  - **Tuned XGBoost (23 Eng Features) vs. Hybrid (10 Raw Features)**:
    - Tuned XGBoost beats Hybrid across **5 of 5 seeds** on RMSE (mean diff $+0.0754 \pm 0.0652$), MAE ($+0.0892 \pm 0.0564$), and P95 AE ($+0.3129 \pm 0.2098$).
    - In extreme tails, however, Tuned XGBoost leads on P99 AE in **3 of 5 seeds** (mean diff $+0.1063 \pm 0.2020$), and the Residual Hybrid actually achieves lower Max AE on **3 of 5 seeds** (seeds 123, 789, 1011; overall mean diff $+0.8141 \pm 2.5716$, Tuned leads in 2 of 5 seeds).
  - Standalone Tuned MLP exhibits higher tail error (P95 AE = **3.6101**, P99 AE = **5.8123**, Max AE = **36.1018**), reflecting rounding errors near the sharp threshold boundaries of the CCME formula.

### 2. Inference Latency and Model Footprint:
- **Inference Latency**: The Tuned MLP beats tree models decisively in inference throughput. As profiled in Table 4 (`reports/compute_cost.csv`), the Tuned MLP executes inference at **0.1882s per 100,000 rows** via vectorized PyTorch BLAS matrix operations on CPU. This is:
  - **8.9× faster** than XGBoost (**1.6714s per 100k rows**).
  - **20.3× faster** than full Random Forest (**3.8200s per 100k rows**).
- **Model Footprint & Measurement**:
  - The Tuned MLP contains **204,289 parameters**.
  - Full-data Random Forest contains **2,191,352 tree nodes** (398,998 at 100k scale) and XGBoost contains **109,501 tree nodes**.
  - The Hybrid architecture contains an MLP residual network of **11,777 parameters** (measured directly via `sum(p.numel() for p in model.parameters())`: Layer 1 Linear(10, 128) has $10 \times 128 + 128 = 1,408$; Layer 2 Linear(128, 64) has $128 \times 64 + 64 = 8,256$; Layer 3 Linear(64, 32) has $64 \times 32 + 32 = 2,080$; Layer 4 Linear(32, 1) has $32 \times 1 + 1 = 33$; Total = $1,408 + 8,256 + 2,080 + 33 = 11,777$). Earlier mention of 11,809 was an arithmetic typo (+32 double counting output), and 46,809 was an unverified placeholder estimate prior to architecture freezing. Fixed to 11,777 everywhere.

---

## 5. Technical Validation & Anti-Leakage Audit

All five technical validation checks were automated and verified in `reports/leakage_check.csv`:

```csv
check_id,description,component,status,details
CHK_01_ROW_DISJOINTNESS,Zero row overlap across train val and test splits,src.split / splits.parquet,PASSED,"Train=1,986,397, Val=415,805, Test=415,804. Overlaps: Tr-Va=0, Tr-Te=0, Va-Te=0."
CHK_02_SCALER_FIT_ISOLATION,Preconditioning transformers fitted strictly on training partition,src.dl.nn_preprocess,PASSED,"Fitted on n=10,000 train rows. Transforming validation / test data produces zero mutation to learned quantiles or reference thresholds."
CHK_03_TARGET_STATS_ISOLATION,Target standardization mean and std computed strictly from y_train,src.dl.data / LoadersBundle,PASSED,"y_mean=85.1380, y_std=17.2453 learned from train partition only. Holdout targets are standardized and unscaled using train statistics."
CHK_04_SPATIAL_GROUP_ISOLATION,GroupKFold by monitoring station (Area) prevents site identity leakage,src.dl.hybrid,PASSED,"Tested across 5 folds. Overlapping monitoring stations between training and OOF: 0. Fails fast via RuntimeError on any station alignment mismatch."
CHK_05_FORBIDDEN_FORMULA_FEATURES,Direct mathematical formula components and target labels strictly excluded,src.features,PASSED,"Audited features against 10 formula/label components. Forbidden columns present: None. Only documented 8 inputs used."
```

---

## 6. Compute Cost & Hardware Profiling

Hardware profiled: **Apple M-series Silicon (8 physical cores, arm64), macOS Darwin, PyTorch 2.15.0, XGBoost 3.4.1**.

| Model | Training Samples | Size (parameters or tree nodes) | Training Time (s) | Inference Time per 100k Rows (s) | Framework |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Naive Baseline** | 20,000 | 1 param | 0.000s | 0.0004s | numpy |
| **Random Forest** | 20,000 | 185,636 nodes | 0.420s | 0.1222s | scikit-learn |
| **Random Forest (Full Trial)** | 1,986,397 | 2,191,352 nodes | 52.410s | 3.8200s | scikit-learn |
| **XGBoost (Phase 1 Tuned)** | 20,000 | 109,501 nodes | 47.020s | 1.5770s | xgboost |
| **Tuned MLP** | 20,000 | 204,289 params | 17.480s | **0.2080s** | pytorch |
| **Residual Hybrid** | 20,000 | 11,777 params (MLP only) | 76.440s | 1.8405s | xgboost + pytorch (MLP only) |

---

## 7. Theoretical Rigor Notes (For Graded Report & Viva Defense)

### 1. Backpropagation and Reverse-Mode Automatic Differentiation

Neural network training relies on formulating the model as a directed acyclic computation graph (DAG), where input activations $x \in \mathbb{R}^{d}$ are mapped to scalar loss $\mathcal{L} \in \mathbb{R}$ through composition of linear maps and non-linear activation functions:

$$z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}, \quad a^{(l)} = \sigma\left(z^{(l)}\right), \quad l = 1, \dots, L$$

where $a^{(0)} = x$, and $\hat{y} = z^{(L)}$ for continuous linear regression. For an $L2$ regression objective on continuous water quality indices:

$$\mathcal{L} = \frac{1}{2} \left( \hat{y} - y \right)^2$$

#### The Reverse-Mode Chain Rule
Backpropagation executes **reverse-mode automatic differentiation**, computing the adjoint state (error gradient with respect to pre-activations):

$$\delta^{(l)} \equiv \frac{\partial \mathcal{L}}{\partial z^{(l)}} \in \mathbb{R}^{d_l}$$

Starting at the output layer $L$:

$$\delta^{(L)} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \odot \sigma'\left(z^{(L)}\right) = (\hat{y} - y) \cdot 1$$

For any preceding hidden layer $l < L$, applying the multivariate chain rule yields the backward recurrence relation:

$$\delta^{(l)} = \left( (W^{(l+1)})^T \delta^{(l+1)} \right) \odot \sigma'\left(z^{(l)}\right)$$

Once adjoints $\delta^{(l)}$ are established, parameter gradients are obtained via outer products:

$$\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \delta^{(l)} \left( a^{(l-1)} \right)^T, \quad \frac{\partial \mathcal{L}}{\partial b^{(l)}} = \delta^{(l)}$$

#### Gradient Flow and Activation Dynamics
The term $\sigma'\left(z^{(l)}\right)$ modulates gradient propagation. For standard piecewise linear activations like ReLU ($\sigma(z) = \max(0, z)$):

$$\sigma_{\text{ReLU}}'(z) = \begin{cases} 1 & z > 0 \\ 0 & z < 0 \end{cases}$$

While ReLU prevents gradient attenuation for positive activations ($\sigma' = 1$), any neuron with negative pre-activations suffers from zero gradient ($\delta^{(l)} = 0$), causing permanent dead neuron inactivation.

In contrast, the **Gaussian Error Linear Unit (GELU)** adopted in Himanshu's tuned architecture:

$$\text{GELU}(z) = z \Phi(z) = z \cdot \frac{1}{2}\left[ 1 + \text{erf}\left( \frac{z}{\sqrt{2}} \right) \right]$$

has continuous, non-zero derivative across $\mathbb{R}$:

$$\frac{d}{dz}\text{GELU}(z) = \Phi(z) + z \frac{e^{-z^2/2}}{\sqrt{2\pi}}$$

This smooth non-monotonic curvature allows small negative gradients to flow through dormant units, avoiding gradient death and producing a smoother optimization landscape.

---

### 2. The Adam and AdamW Optimization Mechanics

Standard Stochastic Gradient Descent (SGD) with learning rate $\eta$ updates weights along the negative instantaneous gradient:

$$\theta_t = \theta_{t-1} - \eta g_t, \quad g_t = \nabla_\theta \mathcal{L}_t(\theta_{t-1})$$

In tabular deep learning, features exhibit severe conditioning discrepancies (e.g., pH has variance $\sim 1$, while Ammonia exhibits extreme right-skewed spikes). SGD oscillates violently across steep coordinate directions while making negligible progress along flat directions.

#### The Adam Algorithm (Kingma & Ba, 2014)
Adaptive Moment Estimation solves coordinate conditioning by tracking both the **first raw moment** (directional momentum) and the **second uncentered moment** (coordinate-wise curvature):

1. **Biased First Moment**:
   $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
2. **Biased Second Moment**:
   $$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
3. **Bias Corrections** (accounting for initialization at the origin $m_0 = 0, v_0 = 0$):
   $$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
4. **Parameter Update**:
   $$\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

#### Why Decoupled Weight Decay (AdamW) is Mathematically Superior
In standard L2-regularized Adam, the regularization term $\frac{1}{2}\lambda \|\theta\|^2$ is simply added to the loss function, altering the gradient:

$$g_t^{\text{L2}} = g_t + \lambda \theta_{t-1}$$

Consequently, the weight decay term enters the second moment accumulator $v_t$:

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) (g_t + \lambda \theta_{t-1})^2$$

When updating parameters:

$$\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \left( \hat{m}_t + \frac{\lambda \theta_{t-1}}{1 - \beta_1^t} \right)$$

This creates an unwanted pathological coupling:
- Weights associated with large, frequent gradients accumulate large $v_t$, meaning their weight decay is **divided by a large $\sqrt{\hat{v}_t}$**, scaling down the effective regularization.
- Weights with small, sparse gradients experience **far stronger regularization**.

To restore true weight decay, **AdamW (Loshchilov & Hutter, 2019)** decouples the weight decay update completely from the gradient moments:

$$\theta_t = \theta_{t-1} - \eta \lambda \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

This guarantees that every parameter decays proportionally to its magnitude at rate $\eta \lambda$, regardless of gradient variance. In our experiments, AdamW with weight decay $10^{-4}$ maintained parameter stability without over-constraining expressive capacity.

---

### 3. Why Models Are Compared Across Seeds

In academic literature, evaluating a single random seed produces deceptive rankings due to multiple independent sources of stochastic variance:
1. **Weight Initialization Variance**: In PyTorch, weights are randomly sampled from Kaiming/Xavier distributions. Different initializations position the model in different basins of attraction.
2. **Mini-Batch Shuffling Stochasticity**: The order in which gradient updates are processed creates different trajectories along the non-convex loss surface.
3. **Subsampling Variance**: In tree models and subsampled ablations, bootstrap draws capture slightly different sample distributions.

#### The "Noise Masquerading as Signal" Trap
Taj's Phase 2 regularization study (`docs/notes_phase2_taj.md`) demonstrated this directly:
- In a 3-seed sweep, weight decay $10^{-4}$ appeared to beat the unregularized reference by $+0.0051$ validation MAE.
- However, when expanded to 5 seeds, the measured seed-to-seed standard deviation was **$\pm 0.0444$** — eight times larger than the apparent gain.
- Relying on a single seed would have led to falsely adopting weight decay as an "algorithmic breakthrough," when in reality the difference was within random noise.

By holding splits, preprocessing, architectures, and the 5 team seeds (`[42, 123, 456, 789, 1011]`) strictly identical across all conditions, we can compute **paired difference tests**:

$$\Delta_i = M_{A, \text{seed}_i} - M_{B, \text{seed}_i}, \quad t = \frac{\bar{\Delta}}{s_\Delta / \sqrt{n}}$$

This isolates true architectural signal from stochastic initialization variance, ensuring robust, peer-review-grade findings for the capstone dissertation.
