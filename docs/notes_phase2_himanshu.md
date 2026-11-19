# What I built and why — Himanshu Pandey (Phase 2)

| | |
| :--- | :--- |
| **GitHub** | [@14-himanshu](https://github.com/14-himanshu) |
| **Role** | DL Literature Review, Base MLP, Architecture Ablation |
| **Branch prefix** | `feat/himanshu-` |
| **Theory note for report** | Multi-Layer Perceptrons: Universal approximation, depth vs. width, activations, weight initialization |

*Maintained incrementally by Himanshu Pandey throughout Phase 2.*

---

## 1. Files I Own

### Delivered & Merged into `main`:
- `docs/refs_dl.bib`: BibTeX database of 12 landmark tabular deep learning papers with verified DOIs and proceedings URLs (Merged in PR #22).
- `docs/literature_review_dl.md`: Comprehensive DL literature review covering tree vs. deep learning trade-offs, regularization, and uncertainty/calibration (Merged in PR #22).
- `src/dl/model.py`: Modular, configurable PyTorch `WaterQualityMLP` architecture with dynamic activation-derived Kaiming/Xavier gain scaling, linear regression output head, and full unit test coverage in `tests/test_dl_model.py` (Merged in PR #24).
- `notebooks/12_base_mlp.ipynb`: Fully executed baseline neural network notebook proving convergence on Area-grouped splits and decisively beating the naive baseline (Merged in PR #24).

### Delivered on this Branch (`feat/himanshu-architecture`):
- `src/dl/architecture.py`: Systematic architecture ablation and validation-only Optuna hyperparameter optimization engine.
- `tests/test_architecture.py`: Fast unit test suite (10 tests) covering grid integrity, metric aggregation, Optuna execution, and results compilation.
- `notebooks/14_architecture.ipynb`: Fully executed and illustrated architecture analysis notebook with depth sweeps, width variations, activation comparisons, Optuna trajectory, and architecture justification.
- `reports/mlp_results.csv`: Comparative multi-seed benchmark table against Naive, Random Forest, XGBoost, and Base MLP.
- `reports/mlp_ablation.csv` & `reports/mlp_optuna_trials.csv`: Persisted raw and aggregated ablation and search logs.
- `reports/figures/architecture_*.png`: High-resolution diagnostic figures (depth sweep, width sweep, activation comparison, Optuna trajectory, and tuned MLP learning curves).
- `docs/theory_note_himanshu.md`: Comprehensive theoretical rigor note on Universal Approximation, Depth vs. Width representation efficiency, activation function gradients, and weight initialization mechanics.
- `docs/notes_phase2_himanshu.md`: Individual viva defense and handover record.

---

## 2. What I Built & Key Decisions

### Step 1: DL Literature Review & BibTeX Database (Delivered on this branch)
- Curated and synthesized **12 seminal papers** across 4 critical pillars:
  1. *Trees vs. Tabular Deep Learning:* Grinsztajn et al. (NeurIPS 2022), Shwartz-Ziv & Armon (Information Fusion 2022), Borisov et al. (IEEE TNNLS 2022).
  2. *Tabular Architectures & Universal Representation:* Gorishniy et al. (NeurIPS 2021), Arik & Pfister (AAAI 2021), Hornik et al. (1989).
  3. *Tabular Regularization Cocktails:* Kadra et al. (NeurIPS 2021), Srivastava et al. (JMLR 2014), Ioffe & Szegedy (ICML 2015), Loshchilov & Hutter (ICLR 2019).
  4. *Uncertainty & Calibration:* Gal & Ghahramani (ICML 2016), Guo et al. (ICML 2017).
- Added verified DOIs and proceedings URLs across all 12 entries for IEEE citation compliance.
- Formulated the central Phase 2 experimental question comparing regularized MLPs against tuned XGBoost/Random Forest and evaluating out-of-fold residual ensembling.
- Clarified that because our target is continuous regression (`CCME_Values`), regression prediction interval metrics (PICP, MPIW) are primary, while Expected Calibration Error (ECE) serves as a secondary post-hoc analysis on binned CCME categories.
- Structured citations for direct integration into Satya's IEEE LaTeX report on Overleaf.

### Step 2: Base MLP Architecture Design & Verification (Delivered in PR #24)
- **Configurable Modular Class (`src/dl/model.py`):** Designed a PyTorch `WaterQualityMLP` parameterized by:
  - `input_dim`: Dynamic feature dimension matching preprocessing output.
  - `hidden_dims`: Sequence specifying layer widths (e.g., `(128, 64)`).
  - `activation`: Modular activation (`relu`, `gelu`, `silu`, `leaky_relu`, `tanh`).
  - `dropout_rate`: Continuous dropout probability for hidden activations.
  - `use_batch_norm` / `use_layer_norm`: Normalization flags.
  - `output_dim`: Defaults to 1 for continuous `CCME_Values` regression.
- **Dynamic Activation-Specific Initialization:** Implemented `_resolve_nonlinearity_and_gain()` which dynamically maps activation non-linearities (`relu`, `leaky_relu`, `gelu`, `silu`, `tanh`) to their exact mathematical gain factors, ensuring initial variance preservation across the forward and backward passes.
- **Linear Output Head:** Continuous regression on $y \in [0, 100]$ demands an unconstrained linear projection without sigmoid squashing, avoiding artificial gradient saturation near index boundaries (0 and 100).
- **Unit Test Coverage:** Authored 10 comprehensive unit tests in `tests/test_dl_model.py` verifying forward passes, output shapes, gradient flow, and parameter validation.

### Step 3: Base MLP Training & Benchmark (Delivered in PR #24)
- Built `notebooks/12_base_mlp.ipynb` on the shared Area-grouped train/val/test split.
- Demonstrated decisive victory over the naive statistical baseline:
  - **Naive Mean Baseline:** Val MAE = 14.8069, Val RMSE = 18.5284, Val R² = -0.0006
  - **Base MLP (128-64 ReLU):** Val MAE = 1.2385, Val RMSE = 1.7581, Val R² = 0.9910
- Saved model checkpoint to `checkpoints/base_mlp.pt` and generated diagnostic plots.
- Rebased branch onto `origin/main`, resolved merge conflicts, safely serialized `Path` objects as strings for PyTorch 2.6+ `weights_only=True` security, and secured merge into `main`.

### Step 4: Architecture Ablation & Tuning (Delivered on `feat/himanshu-architecture`)
- **Systematic Ablation:** Swept depth (1 to 5 layers), width capacity (narrow, uniform, wide, very wide), and activation non-linearities (ReLU, GELU, SiLU, LeakyReLU, Tanh) across multi-seed evaluations.
- **Key Finding on Depth:** Depth is non-negotiable for tabular water quality modeling. Single-layer networks fail to resolve multi-chemical interactions (Val MAE $\approx 1.50$), whereas Depth 3–4 drops MAE by ~50% to $\approx 0.76 - 0.83$. Beyond 4 layers, gains plateau due to gradient attenuation without residual connections.
- **Key Finding on Activations:** Smooth activations (GELU, SiLU) outperform or match ReLU, eliminating the dead neuron pathology and smoothing the loss landscape.
- **Validation-Only Hyperparameter Optimization:** Implemented Optuna `TPESampler` search evaluating strictly on validation MAE, optimizing depth, widths, shape, learning rate, and weight decay with zero test leakage.
- **Comparative Benchmarking:** Compiled final multi-seed benchmark table (`reports/mlp_results.csv`) and generated full diagnostic plots in `reports/figures/`.
- **Theory Note:** Completed extensive theoretical rigor documentation (`docs/theory_note_himanshu.md`) covering Universal Approximation, Depth vs. Width representation efficiency, activation function dynamics, and weight initialization mechanics.

---

## 3. Why I Built It That Way

1. **Why an MLP rather than TabNet or complex transformers:**
   - As proven by Gorishniy et al. (2021) and Kadra et al. (2021), complex tabular transformers rarely beat well-tuned MLPs on medium-scale tabular data and incur massive hyperparameter sensitivity. A clean, modular MLP provides reliable reproducibility and clear ablation attribution.
2. **Why modular activation functions (ReLU vs. GELU vs. SiLU):**
   - Grinsztajn et al. (2022) highlighted that smooth activations can struggle on tabular functions with threshold steps. Comparing piecewise linear ReLU against smooth GELU provides direct empirical evidence on whether smoothness helps or hinders water quality index estimation.
3. **Why early base MLP delivery was prioritized:**
   - In the Phase 2 workflow, Taj, Jivit, and Satya all branch from the base MLP. Releasing a working, verified model early unblocks the entire team.

---

## 4. Theory Note Summary (Multi-Layer Perceptrons)

- **Universal Approximation Theorem (Hornik et al., 1989):** A feedforward network with a single hidden layer and non-linear activation can approximate any continuous function on a compact subset of $\mathbb{R}^n$ given sufficient width. However, depth exponentially reduces the number of units required to represent compositional functions.
- **Depth vs. Width Trade-off:** Depth introduces hierarchical feature abstraction (combining individual chemical concentrations into composite impairment indicators), while excessive depth risks vanishing gradients and degradation without residual skip-connections.
- **Activation Functions:** ReLU ($\max(0, x)$) mitigates vanishing gradients for positive inputs but suffers from dead neurons for negative pre-activations; GELU ($x \Phi(x)$) and SiLU ($x \sigma(x)$) provide smooth non-monotonic landscapes enabling finer gradient flow.
- **Weight Initialization:** He (Kaiming) initialization ($\text{Var}(W) = \frac{2}{n_{\text{in}}}$) preserves activation and gradient variance across deep ReLU/GELU layers, preventing initial forward signal explosion or backward vanishing.
