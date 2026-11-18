# Deep Learning Literature Review: Tabular Architectures, Regularization, and Uncertainty

**Author:** Himanshu Pandey  
**Role:** DL Literature Review, Base MLP, Architecture Ablation  
**Project:** Water Quality Prediction — Phase 2 (Deep Learning Extension)  
**BibTeX Source:** `docs/refs_dl.bib`  

---

## 1. Executive Summary and Scope

Phase 2 extends the machine learning water quality estimation benchmark from tree ensembles (Random Forest, XGBoost) to deep feedforward neural architectures (Multi-Layer Perceptrons; MLPs) and hybrid formulations. 

This literature review synthesizes 12 seminal and state-of-the-art works to answer the foundational questions governing tabular deep learning:
1. **Why do gradient-boosted decision trees (GBDTs) historically outperform deep architectures on tabular datasets?**
2. **What architectural choices (depth, width, activations, embeddings) enable MLPs to compete effectively?**
3. **Which regularization mechanisms prevent severe overfitting on continuous, correlated tabular measurements?**
4. **How can epistemic model uncertainty and predictive calibration be rigorously quantified in deep regression models?**

---

## 2. Thematic Paper Reviews

### Theme A: Tree Ensembles vs. Deep Neural Networks on Tabular Data

#### 1. Grinsztajn, Oyallon, & Varoquaux (NeurIPS 2022)
* **Title:** *Why do tree-based models still outperform deep learning on tabular data?* \cite{grinsztajn2022why}
* **Findings:**
  - Evaluated on a diverse benchmark of 45 tabular datasets across numerical, categorical, regression, and classification tasks.
  - Demonstrates that tree-based models (XGBoost, CatBoost, Random Forest) consistently outperform deep learning models (ResNet, MLP, Saint, FT-Transformer) with substantially less computational tuning.
  - Identifies three fundamental inductive properties responsible for the performance gap:
    1. **Indifference to uninformative features:** Trees greedily select informative split dimensions, ignoring noisy or redundant inputs, whereas neural networks assign non-zero weights across all input dimensions, diluting signal.
    2. **Preservation of non-smooth functions:** Real-world tabular data often contains discrete jumps, boundary cutoffs, and irregular decision surfaces. Smooth neural activations (e.g., Sigmoid, GELU) struggle to fit discontinuous boundaries without excessive depth.
    3. **Rotation invariance bias:** Standard neural networks treat arbitrary orthogonal rotations of feature space as equivalent ($f(Rx)$ has identical capacity to $f(x)$). Tabular features, unlike image pixels or audio waveforms, possess individual semantic identities (e.g., `pH` vs. `Temperature`), rendering rotation invariance an unfavorable inductive bias.
* **Relevance to Water Quality:** Water quality data features heavy-tailed, non-Gaussian distributions and strict guideline threshold boundaries (e.g., CCME ceiling at 100.0, discrete chemical limit infractions). Grinsztajn et al. explain why an off-the-shelf MLP underperforms XGBoost without explicit feature scaling, regularization, and specialized architectures.

#### 2. Shwartz-Ziv & Armon (Information Fusion 2022)
* **Title:** *Tabular Data: Deep Learning is Not All You Need* \cite{shwartzziv2022tabular}
* **Findings:**
  - Conducted a comprehensive benchmark comparing XGBoost against recent deep tabular models (including TabNet, NODE, DNF-Net, and 1D-CNNs).
  - Showed that XGBoost decisively outperformed all deep models across majority datasets. When deep models did perform well, the computational cost of hyperparameter tuning was orders of magnitude higher.
  - Demonstrated that ensembling XGBoost with deep tabular models yielded superior accuracy over either model alone, suggesting complementary error profiles.
* **Relevance to Water Quality:** Directly justifies Jivit's hybrid track and the team's Phase 2 hypothesis: an MLP alone may not unilaterally surpass tuned XGBoost, but an ensemble/hybrid leveraging out-of-fold residuals can capture orthogonal signal.

#### 3. Borisov et al. (IEEE TNNLS 2022)
* **Title:** *Deep Neural Networks and Tabular Data: A Survey* \cite{borisov2022deep}
* **Findings:**
  - Establishes a comprehensive taxonomy of tabular deep learning architectures: data-transformation models, specialized tabular networks, and hybrid architectures.
  - Identifies key tabular challenges: mixed feature types, low sample-to-feature ratios, extreme value skewness, lack of spatial/temporal local correlation, and absence of standardized pre-training pipelines.
* **Relevance to Water Quality:** Guides data preprocessing for neural networks (e.g., standard vs. robust scaling, handling extreme skew via logarithmic compression, categorical embeddings for spatial identifiers like `Country` and `Waterbody Type`).

---

### Theme B: Tabular Architectures and Multi-Layer Perceptrons

#### 4. Gorishniy, Rubachev, Khrulkov, & Babenko (NeurIPS 2021)
* **Title:** *Revisiting Deep Learning Models for Tabular Data* \cite{gorishniy2021revisiting}
* **Findings:**
  - Demonstrated that classical architectures—specifically a carefully tuned Multi-Layer Perceptron (MLP) and a simple ResNet-like feedforward network—consistently match or outperform specialized, complex tabular architectures such as TabNet.
  - Proved that standard MLPs, when combined with proper normalization (LayerNorm or BatchNorm), modern activations, and disciplined hyperparameter search (learning rate, weight decay, dropout), establish a formidable tabular baseline.
* **Relevance to Water Quality:** Validates our architectural choice: instead of relying on fragile complex architectures, we implement a modular, well-regularized MLP (`src/dl/model.py`) optimized with systematic depth/width sweeps.

#### 5. Arik & Pfister (AAAI 2021)
* **Title:** *TabNet: Attentive Interpretable Tabular Learning* \cite{arik2021tabnet}
* **Findings:**
  - Proposed sequential multi-step attention to select salient features at each decision step, mimicking decision tree split selection.
  - Introduced sparsemax masks to enforce sparse feature selection across tabular inputs.
* **Relevance to Water Quality:** Emphasizes that tabular feature sparsity matters; in our MLP, this motivates evaluating dropout and L1/L2 weight penalties to prevent over-reliance on collinear nutrient features (`Nitrogen` vs. `Nitrate`).

#### 6. Hornik, Stinchcombe, & White (Neural Networks 1989)
* **Title:** *Multilayer feedforward networks are universal approximators* \cite{hornik1989multilayer}
* **Findings:**
  - Proved the Universal Approximation Theorem: standard multilayer feedforward networks with as few as a single hidden layer and continuous sigmoidal or non-linear activations can approximate any Borel measurable function to arbitrary precision on compact subsets of $\mathbb{R}^n$.
* **Relevance to Water Quality:** Grounding for our Phase 2 Theory Note. The CCME Water Quality Index formula is a continuous, closed-form mapping $\mathcal{F}: \mathbb{R}^8 \to [0, 100]$; hence, an MLP theoretically possesses sufficient representation capacity to emulate the deterministic index surface.

---

### Theme C: Regularization Paradigms for Tabular MLPs

#### 7. Kadra, Lindauer, Hutter, & Grabocka (NeurIPS 2021)
* **Title:** *Well-tuned Simple Nets Excel on Tabular Datasets* \cite{kadra2021well}
* **Findings:**
  - Introduced the concept of "Regularization Cocktails" for simple MLPs.
  - Showed that an unregularized or poorly regularized MLP fails on tabular data, but searching over a joint space of regularizers (dropout, weight decay, label smoothing, mixup, stochastic depth) closes the performance gap with tree ensembles across 40 datasets.
* **Relevance to Water Quality:** Directly informs Taj's regularization ablation (`notebooks/13_regularization.ipynb`) and our architectural parameterization (`src/dl/model.py`), confirming that regularization is the decisive factor separating failing MLPs from competitive MLPs on tabular data.

#### 8. Srivastava et al. (JMLR 2014)
* **Title:** *Dropout: A Simple Way to Prevent Neural Networks from Overfitting* \cite{srivastava2014dropout}
* **Findings:**
  - Formulates dropout: randomly setting hidden unit activations to zero with probability $p$ during training.
  - Interpreted mathematically as an implicit geometric ensemble of $2^N$ shared-weight subnetworks, preventing co-adaptation of feature detectors.
* **Relevance to Water Quality:** Essential for Himanshu's theory note and Taj's regularization sweep. Collinear chemical measurements in water quality (e.g., `BOD` and `DO`, or `Nitrogen` and `Nitrate`) encourage strong co-adaptation; dropout forces alternative pathways.

#### 9. Ioffe & Szegedy (ICML 2015)
* **Title:** *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift* \cite{ioffe2015batch}
* **Findings:**
  - Normalizes layer inputs across mini-batches to zero mean and unit variance, followed by learnable affine transformations ($\gamma, \beta$).
  - Stabilizes gradient propagation, permits higher learning rates, and acts as a mild regularizer.
* **Relevance to Water Quality:** Water quality analytes vary across multiple orders of magnitude (e.g., `Ammonia` 0.001–200 mg/l vs. `pH` 6–9); batch normalization inside hidden layers prevents internal covariate drift during training.

#### 10. Loshchilov & Hutter (ICLR 2019)
* **Title:** *Decoupled Weight Decay Regularization (AdamW)* \cite{loshchilov2018decoupled}
* **Findings:**
  - Showed that standard L2 regularization in adaptive gradient algorithms like Adam does not act as true weight decay due to gradient scaling.
  - Decoupling weight decay directly from gradient updates ($w_{t+1} = (1 - \eta \lambda) w_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$) restores effective weight decay regularization.
* **Relevance to Water Quality:** Mandates using AdamW or decoupled weight decay in `src/dl/train.py` rather than naive SGD or unregularized Adam.

---

### Theme D: Uncertainty, Calibration, and Interpretability

#### 11. Gal & Ghahramani (ICML 2016)
* **Title:** *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning* \cite{gal2016dropout}
* **Findings:**
  - Proved that deep networks with dropout applied at test time (Monte Carlo Dropout / MC Dropout) are mathematically equivalent to approximate variational inference in deep Gaussian processes.
  - By performing $T$ stochastic forward passes at inference, one extracts both predictive mean $\hat{y} = \frac{1}{T}\sum \hat{y}_t$ and epistemic variance $\sigma^2 = \frac{1}{T}\sum (\hat{y}_t - \hat{y})^2 + \tau^{-1} I$.
* **Relevance to Water Quality:** Powers Satya's uncertainty quantification (`src/dl/calibration.py`, `notebooks/16_calibration_explain.ipynb`). In safety-critical environmental monitoring, confidence intervals are vital to flag when water quality predictions are uncertain.

#### 12. Guo, Pleiss, Sun, & Weinberger (ICML 2017)
* **Title:** *On Calibration of Modern Neural Networks* \cite{guo2017calibration}
* **Findings:**
  - Modern deep networks, while highly accurate, are poorly calibrated (overconfident in their predictions).
  - Explores Expected Calibration Error (ECE), reliability diagrams, and post-hoc temperature scaling to align predicted probabilities with empirical observation frequencies.
* **Relevance to Water Quality:** Informs calibration assessment when continuous indices are discretized into regulatory categories. Because our primary modeling task is continuous index regression (`TARGET_KIND = "regression"`), uncertainty is fundamentally quantified through continuous prediction intervals (PICP/MPIW). ECE applies strictly as a secondary post-hoc diagnostic when continuous WQI scores are binned into the five standard CCME classes (Poor, Marginal, Fair, Good, Excellent).

---

## 3. Synthesis: The Core Research Question for Phase 2

### What others found across literature:
1. GBDTs remain the standard baseline for raw tabular data due to robustness to unscaled features, feature selection capability, and step-function boundaries \cite{grinsztajn2022why, shwartzziv2022tabular}.
2. Properly regularized MLPs (leveraging AdamW, dropout, batch normalization, and tuned depth/width) perform competitively with complex tabular networks \cite{gorishniy2021revisiting, kadra2021well}.
3. Hybrids combining tree predictions and neural representations exploit orthogonal structural inductive biases \cite{shwartzziv2022tabular, borisov2022deep}.
4. Deep models provide natural Bayesian uncertainty quantification via test-time dropout that tree models lack \cite{gal2016dropout}.

### What question this project tests:
> **Primary Phase 2 Research Question:**  
> *When applied to multi-country surface water monitoring data under strictly identical spatial train-validation-test splits, does a regularized Multi-Layer Perceptron (MLP) achieve predictive parity with tuned tree ensembles (Random Forest, XGBoost) in estimating the continuous CCME Water Quality Index? Furthermore, does an out-of-fold residual hybrid of XGBoost and MLP improve overall accuracy, calibration, and resilience under missing sensor measurements?*

### Evaluation Metrics Used:
- **Accuracy / Fit:** Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), Coefficient of Determination ($R^2$).
- **Multi-Seed Stability:** Mean and standard deviation across 5 fixed random seeds ($seeds = [42, 123, 456, 789, 1011]$).
- **Uncertainty & Regression Calibration (Primary):** Prediction Interval Coverage Probability (PICP) and Mean Prediction Interval Width (MPIW) evaluated for 90% and 95% intervals via Monte Carlo Dropout ($T=50$ forward passes).
- **Categorical Calibration (Secondary Post-Hoc):** Expected Calibration Error (ECE) and reliability diagrams, evaluated strictly as a secondary analysis if continuous CCME predictions are categorized into the five discrete regulatory quality bands (Poor, Marginal, Fair, Good, Excellent).
- **Computational Cost:** Parameter count, wall-clock training duration, and per-sample inference latency.
