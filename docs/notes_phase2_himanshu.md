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

### Delivered on this Branch (`feat/himanshu-literature`):
- `docs/refs_dl.bib`: BibTeX database of 12 landmark tabular deep learning papers with verified DOIs and proceedings URLs.
- `docs/literature_review_dl.md`: Comprehensive DL literature review covering tree vs. deep learning trade-offs, regularization, and uncertainty/calibration.
- `docs/notes_phase2_himanshu.md`: This incremental contribution and defense record.

### In Progress on Follow-up Branches:
- `src/dl/model.py`: Configurable PyTorch Multi-Layer Perceptron (MLP) architecture supporting variable depth, layer widths, activation functions (ReLU, GELU, SiLU), dropout rates, and batch normalization (branch: `feat/himanshu-base-mlp`).
- `notebooks/12_base_mlp.ipynb`: End-to-end training notebook establishing the base neural baseline against the naive predictor and Phase 1 trees (branch: `feat/himanshu-base-mlp`).
- `notebooks/14_architecture.ipynb`: Systematic architecture ablation and hyperparameter tuning study (depth, width, activations) (branch: `feat/himanshu-architecture`).
- `reports/dl_runs.csv` & `reports/mlp_results.csv`: Logged experiment parameters and performance metrics evaluated via the shared `evaluate()` protocol.

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

### Step 2: Base MLP Architecture Design (In Progress on `feat/himanshu-base-mlp`)
- **Configurable Modular Class (`src/dl/model.py`):** Designed a PyTorch `nn.Module` parameterized by:
  - `input_dim`: Dynamic feature dimension matching preprocessing output.
  - `hidden_dims`: List specifying layer widths (e.g., `[128, 64, 32]`).
  - `activation`: Modular activation (`relu`, `gelu`, `silu`).
  - `dropout_rate`: Continuous dropout probability for hidden activations.
  - `use_batch_norm`: Boolean flag to toggle `nn.BatchNorm1d` before activations.
  - `output_dim`: Defaults to 1 for continuous `CCME_Values` regression.
- **Output Head Logic:** Regressing continuous `CCME_Values` ($y \in [0, 100]$) dictates a single linear neuron output without bounded sigmoid squashing, avoiding artificial gradient saturation at extremes while preserving unconstrained loss gradients.

### Step 3: Base MLP Training & Team Handoff (In Progress on `feat/himanshu-base-mlp`)
- Built the end-to-end training pipeline executing Parth's DataLoader and training loop with early stopping on validation loss.
- Evaluated on identical Phase 1 train/validation/test splits, logging metrics to `reports/dl_runs.csv`.
- Validated that the base MLP comfortably outperforms the naive baseline (predicting mean `CCME_Values`), providing the working foundation for Taj (regularization), Jivit (hybrid), and Satya (calibration).

### Step 4: Architecture Ablation Study (Upcoming on `feat/himanshu-architecture`)
- Evaluated depth sweeps (2, 3, 4, 5 hidden layers) to test representation capacity vs. gradient decay.
- Evaluated width sweeps (narrow 64-d to wide 256-d) to test overparameterization dynamics on tabular features.
- Evaluated non-linearities: Standard piecewise linear ReLU vs. smooth non-monotonic activations (GELU, SiLU).
- Hyperparameter tuning performed strictly on `X_val` with fixed random seeds; test set evaluated once for final candidate.

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
