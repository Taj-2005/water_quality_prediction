# What I built and why — Satya Yadav (Phase 2)

| | |
| :--- | :--- |
| **GitHub** | [@ACE-pwd](https://github.com/ACE-pwd) |
| **Role** | Calibration, Uncertainty Estimation, Explainability, Technical Validation |
| **Branch prefix** | `feat/satya-calibration` |
| **Theory note for report** | Calibration (Reliability Diagrams, ECE), Prediction Intervals (PICP, Conformal), Monte Carlo Dropout, and Feature Attribution |

*Maintained by Satya Yadav for Phase 2 assessment and viva defense.*

---

## 1. Files I Own in Phase 2

### Code & Modules:
- `src/dl/calibration.py`: Complete calibration and uncertainty module providing:
  - Prediction Interval Coverage Probability (`prediction_interval_coverage`: PICP & MPIW)
  - Split Conformal Prediction intervals (`calibrate_conformal_quantile`, `conformal_prediction_intervals`)
  - Monte Carlo Dropout uncertainty sampling (`monte_carlo_dropout_predict`, `enable_dropout_only`)
  - Permutation Feature Importance for neural networks (`permutation_feature_importance_mlp`)
  - Re-exports and integrates reliability diagram and ECE helpers from `src/metrics.py`.
- `src/metrics.py`: Extended with `reliability_diagram_data()`, `expected_calibration_error()`, `category_calibration_summary()`, and `calibration_summary()`.
- `tests/test_calibration.py`: Comprehensive test suite (43 unit tests) testing reliability diagrams, ECE, MCE, and category calibration on synthetic/dummy data.
- `tests/test_dl_calibration.py`: Unit test suite testing prediction interval coverage, conformal quantiles, Monte Carlo dropout stochasticity, and permutation importance.

### Notebooks & Reports:
- `notebooks/16_calibration_explain.ipynb`: End-to-end technical validation notebook comparing XGBoost and the regularized MLP across MC dropout uncertainty, 95% conformal intervals, reliability diagrams, and feature attribution.
- `notebooks/16_calibration.ipynb`: Multi-model category calibration and reliability diagram suite.
- `reports/calibration_results.csv`: Persisted technical validation benchmark table reporting nominal coverage, empirical PICP, MPIW, ECE, and MCE.
- `reports/figures/16_*.png`: Diagnostic figures (MC dropout uncertainty vs error, conformal interval comparisons, reliability curves, and feature importance rankings).

---

## 2. What I Built & Key Technical Decisions

### Decision 1: Calibration for Continuous Regression vs Classification
- *The Problem:* Our target `CCME_Values` is a continuous index in [31.3, 100.0], not discrete classes. In classification, calibration is well-defined as $P(Y = y \mid \hat{P} = p) = p$. For regression, calibration must be formulated in terms of **prediction intervals** and **binned quantile reliability**.
- *Solution:* I implemented a two-tier calibration architecture:
  1. **Prediction Interval Coverage Probability (PICP):** Evaluates whether nominal $95\%$ intervals contain ground truth $95\%$ of the time on unseen test sites.
  2. **Reliability Diagrams & ECE:** Discretizes the continuous prediction range into equal-width intervals, measuring the gap between mean predicted value $\bar{\hat{y}}_b$ and mean observed value $\bar{y}_b$.

### Decision 2: Conformal Prediction for Distribution-Free Guarantees
- *Alternative Rejected:* Gaussian parametric intervals ($\hat{y} \pm 1.96 \cdot \text{RMSE}$).
- *Why Rejected:* Target errors are heavily non-Gaussian and heteroscedastic near the index ceiling ($y = 100$). Assuming normality yields severe under-coverage.
- *Chosen Approach:* Split Conformal Prediction (Vovk et al., 2005). We calibrate non-conformity scores $|y_i - \hat{y}_i|$ on the validation split:
  $$\hat{q} = \text{Quantile}\left( |r_{\text{val}}|, \frac{\lceil (n+1)(1-\alpha) \rceil}{n} \right)$$
  This guarantees exact finite-sample coverage $P(Y_{\text{test}} \in [\hat{y} - \hat{q}, \hat{y} + \hat{q}]) \ge 1 - \alpha$ with zero distributional assumptions.

### Decision 3: Monte Carlo Dropout for Epistemic Uncertainty
- *Implementation:* During test-time inference, the model is set to `eval()` mode but `nn.Dropout` modules are forced into `train()` mode (`enable_dropout_only()`).
- *Why:* Keeping BatchNorm layers in eval mode prevents mini-batch statistics drift while allowing weight masking across $K = 25$ forward passes. The empirical variance across passes estimates epistemic (model) uncertainty.

### Decision 4: Permutation Importance vs Tree SHAP
- *Finding:* When permuting features on the MLP, **Orthophosphate (mg/l)** causes the largest MAE degradation, followed by **Ammonia (mg/l)** and **Biochemical Oxygen Demand (mg/l)**.
- *Physics Alignment:* This perfectly matches Phase 1 XGBoost Tree SHAP rankings (where Orthophosphate had top SHAP of 8.813). Both trees and neural networks independently converge on the same physical drivers of water degradation.

---

## 3. Theory Note for the Final Report

### 3.1 Reliability Diagrams and Expected Calibration Error (ECE)
For continuous index regression binned into $M$ disjoint intervals $B_1, \dots, B_M$:
$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \frac{1}{|B_m|} \sum_{i \in B_m} \hat{y}_i - \frac{1}{|B_m|} \sum_{i \in B_m} y_i \right|$$
Normalizing by the index span $(y_{\max} - y_{\min} = 100)$ produces a dimensionless score in $[0, 1]$. Maximum Calibration Error (MCE) records the supremum over bin discrepancies:
$$\text{MCE} = \max_{m \in \{1,\dots,M\}} \left| \bar{\hat{y}}_m - \bar{y}_m \right|$$

### 3.2 Prediction Interval Coverage Probability (PICP) & Sharpness (MPIW)
An interval estimator $[\hat{L}(x), \hat{U}(x)]$ is evaluated on two dual criteria:
1. **Empirical Coverage (PICP):**
   $$\text{PICP} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(y_i \in [\hat{L}(x_i), \hat{U}(x_i)])$$
2. **Sharpness (MPIW):**
   $$\text{MPIW} = \frac{1}{N} \sum_{i=1}^N (\hat{U}(x_i) - \hat{L}(x_i))$$
A model with $\text{PICP} \ge 1 - \alpha$ and minimal MPIW provides the sharpest valid predictions.

### 3.3 Monte Carlo Dropout as Approximate Bayesian Inference
Gal & Ghahramani (2016) demonstrated that an $L$-layer neural network with dropout applied before every weight layer is mathematically equivalent to variational inference in a deep Gaussian process. Minimizing cross-entropy or MSE loss minimizes the Kullback-Leibler divergence $D_{\mathrm{KL}}(q(W) \parallel p(W))$ between the dropout distribution $q(W)$ and the true weight posterior $p(W \mid \mathcal{D})$. The predictive mean and epistemic variance are approximated via Monte Carlo integration:
$$\mathbb{E}[y^*] \approx \frac{1}{K} \sum_{k=1}^K \hat{f}_{\hat{W}_k}(x^*), \quad \operatorname{Var}[y^*] \approx \sigma_{\text{noise}}^2 + \frac{1}{K} \sum_{k=1}^K \left( \hat{f}_{\hat{W}_k}(x^*) - \mathbb{E}[y^*] \right)^2$$

### 3.4 Feature Attribution via Permutation Importance
Fisher, Rudin & Dominici (2019) define Model Reliance (Permutation Importance) as the ratio or difference between test error on corrupted inputs $X^{\text{perm}(j)}$ and baseline test error $e(f, X, y)$:
$$\operatorname{PFI}(j) = \mathcal{L}(f(X^{\text{perm}(j)}), y) - \mathcal{L}(f(X), y)$$
Unlike gradient-based saliency methods that suffer from saturation, permutation importance directly reflects the global dependency of predictions on feature $j$.

---

## 4. How My Work Connects to the Team Pipeline

- **From Himanshu:** Consumed `WaterQualityMLP` from `src/dl/model.py` and best architecture parameters to run MC dropout and permutation importance.
- **From Taj:** Consumed preconditioned neural inputs from `src/dl/nn_preprocess.py`.
- **From Jivit:** Reused Phase 1 XGBoost predictions to produce side-by-side conformal interval benchmarks.
- **From Parth:** Enforced strict Area-grouped test isolation with zero leakage.
