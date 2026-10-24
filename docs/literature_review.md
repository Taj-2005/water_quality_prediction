# Literature Review: Water Quality Prediction Using Machine Learning

**Author:** Himanshu Pandey  
**Role:** Literature Review, Feature Engineering, Random Forest  
**Project:** CSAI Project 18 — Water Quality Prediction (Phase 1)  
**BibTeX Source:** `docs/refs.bib`  

---

## 1. Water Quality Assessment and Water Quality Indices (WQI)

### 1.1 Rationale for Systematic Water Quality Assessment
Surface and freshwater bodies are dynamic ecological systems subjected to compounding stresses from agricultural non-point runoff, municipal wastewater discharge, industrial effluents, and climate-induced hydrologic fluctuations. Routine monitoring of aquatic systems requires measuring a wide variety of physicochemical and biological variables, including dissolved oxygen (DO), biochemical oxygen demand (BOD), ammoniacal nitrogen, nitrates, orthophosphates, pH, electrical conductivity, and water temperature. Each individual parameter reflects distinct chemical equilibria and biological pathways. For instance, DO governs aquatic respiration and fish survival; excessive BOD indicates high concentrations of biodegradable organic waste; elevated nitrates and orthophosphates trigger cultural eutrophication and toxic algal blooms; and deviations in pH alter chemical speciation and heavy metal bioavailability (Abbasi & Abbasi, 2012).

However, reporting disjointed matrices of individual chemical concentrations poses severe communication challenges for environmental resource managers, regulatory agencies, and policymakers. A single water sample might meet standards for DO and pH while simultaneously exceeding toxicity thresholds for ammonia and orthophosphate. To bridge the divide between complex environmental analytical chemistry and operational governance, researchers developed aggregated numerical indices.

### 1.2 Genesis and Structure of Water Quality Indices
The concept of a composite Water Quality Index was introduced by Horton (1965) and formalized by Brown et al. (1970) under the auspices of the National Sanitation Foundation (NSF WQI). A WQI aggregates disparate physical and chemical measurements into a single, dimensionless numerical rating (typically scaling from 0 to 100) reflecting overall ecosystem health. Global environmental agencies have since adapted this framework, giving rise to diverse index formulations including the Oregon WQI, the British Columbia WQI, the weighted arithmetic WQI, and the Canadian Council of Ministers of the Environment Water Quality Index (CCME WQI) (Lumb et al., 2011).

Despite their utility, classical index formulations suffer from structural aggregation ambiguities:
- **Eclipsing:** Occurs when an overall index score indicates acceptable water quality despite a single critical toxicant violating lethal environmental thresholds.
- **Ambiguity:** Occurs when multiple divergent parameter combinations produce identical aggregated scores.
- **Rigid Thresholding:** Fixed guideline cutoffs do not account for natural geogenic background variation across catchments.

### 1.3 The CCME Water Quality Index Methodology
The CCME WQI methodology (CCME, 2001) is one of the most widely adopted international standards due to its flexibility across varying aquatic ecosystem objectives and jurisdictional guideline suites. The CCME framework derives a final dimensionless index score by evaluating guideline compliance along three orthogonal geometric dimensions:

1. **Scope ($F_1$):** Measures the percentage of water quality variables that fail to meet their respective environmental guidelines at least once during the evaluation window relative to the total number of variables monitored:
   $$F_1 = \left( rac{	ext{Number of failed variables}}{	ext{Total number of variables}} ight) 	imes 100$$

2. **Frequency ($F_2$):** Represents the percentage of individual analytical tests that violate guideline standards across all samples and sampling times:
   $$F_2 = \left( rac{	ext{Number of failed tests}}{	ext{Total number of tests}} ight) 	imes 100$$
   *Note on instant/point observations:* When calculated on discrete, single-observation records (as in instantaneous sensor measurements where each row represents a single sampling timestamp across $P$ parameters), the number of failed tests equals the number of failed variables, meaning $F_2 \equiv F_1$.

3. **Amplitude ($F_3$):** Quantifies the magnitude by which non-compliant test values exceed their permissible standards. For parameters where higher values exceed the guideline (e.g., nutrients, BOD):
   $$	ext{excursion}_i = \left( rac{	ext{value}_i}{	ext{standard}_i} ight) - 1$$
   For parameters where values below standard represent guideline failure (e.g., dissolved oxygen):
   $$	ext{excursion}_i = \left( rac{	ext{standard}_i}{	ext{value}_i} ight) - 1$$
   The normalized sum of excursions ($nse$) is calculated across all tests:
   $$nse = rac{\sum_{i=1}^{	ext{failed}} 	ext{excursion}_i}{	ext{Total number of tests}}$$
   An asymptotic hyperbolic transformation scales the excursion into an amplitude factor bounded strictly between 0 and 100:
   $$F_3 = rac{nse}{0.01 \cdot nse + 0.01}$$

The final index is computed as the Euclidean distance from the worst-case impaired point $(F_1=100, F_2=100, F_3=100)$ to the observed coordinate in three-dimensional factor space, normalized by $\sqrt{100^2 + 100^2 + 100^2} = \sqrt{30000} pprox 173.2$:
$$	ext{CCME\_Values} = 100 - \left( rac{\sqrt{F_1^2 + F_2^2 + F_3^2}}{1.732} ight)$$

The resulting continuous score $[0, 100]$ maps to five qualitative water quality tiers:
- **Excellent (95–100):** Water quality is protected with virtual absence of impairment; conditions close to pristine natural state.
- **Good (80–94):** Minor degree of threat or impairment; conditions rarely depart from desirable levels.
- **Fair (65–79):** Water quality is usually protected but occasionally threatened or impaired.
- **Marginal (45–64):** Water quality is frequently threatened or impaired; conditions often depart from desirable levels.
- **Poor (0–44):** Water quality is almost always threatened or impaired; conditions usually depart from desirable levels.

### 1.4 Index Calculation vs. Predictive Modeling
A fundamental methodological distinction must be drawn between **index calculation** and **predictive machine learning**:
- **Index Calculation:** An exact, deterministic, closed-form mapping $\mathcal{F}: \mathbb{R}^P 	o [0, 100]$. Given the complete suite of analytical measurements for a water parcel, computing the CCME WQI requires no estimation or model fitting; it is an arithmetic aggregation of contemporaneous test measurements against legal thresholds.
- **Predictive Modeling:** Approximates an unknown, stochastic, or unmeasured data-generating process $\hat{f}(X) pprox y$. Legitimate machine learning applications arise when:
  1. Inferring expensive, labor-intensive lab measurements (e.g., 5-day BOD, heavy metal mass spectrometry) from rapid, low-cost optical or electrochemical sensor measurements (e.g., pH, conductivity, turbidity, water temperature).
  2. Forecasting future water quality states $y_{t+\Delta t}$ from historical time series $\{X_t, X_{t-1}, \dots\}$.
  3. Spatial interpolation of water quality across unmonitored river reaches or ungauged catchments.
  4. Building computational surrogate emulators to bypass costly numerical simulations.

When a machine learning model is trained using the exact chemical features that constitute the inputs to the CCME mathematical formula in order to predict `CCME_Values`, the model is performing **function approximation of an explicit mathematical equation**, not environmental forecasting. Acknowledging this distinction is critical to avoid misinterpreting high empirical accuracy as physical insight.

---

## 2. Machine Learning for Water Quality Prediction

### 2.1 Scope of Supervised Learning in Water Resources
Over the past decade, supervised machine learning has emerged as a dominant paradigm in environmental informatics, addressing non-linearities, threshold responses, and complex biogeochemical interactions that traditional linear regression and mechanistic hydrodynamic models struggle to capture (Ahmed et al., 2019). Machine learning applications broadly fall into two task formulations:

1. **Constituent Prediction (Target = Individual Analyte):** Predicting unmeasured or difficult-to-monitor water parameters from readily accessible surrogates (e.g., predicting Dissolved Oxygen or BOD from temperature, pH, and electrical conductivity).
2. **Index Prediction (Target = Aggregated WQI):** Modeling composite indices directly to provide continuous watershed health scores or rapid classification for early-warning alarm systems (Bui et al., 2020; Aldhyani et al., 2020).

### 2.2 Regression vs. Classification Formulations

| Dimension | Continuous Regression ($y \in [0, 100]$) | Categorical Classification ($y \in \{	ext{Poor}, \dots, 	ext{Excellent}\}$) |
| :--- | :--- | :--- |
| **Information Retention** | Preserves all variance, fine-grained gradients, and trajectory dynamics. | Discretizes continuous space into discrete bins, discarding within-class variance. |
| **Threshold Sensitivity** | Smooth, continuous error penalization across all ranges. | Highly sensitive to arbitrary boundaries; a shift from 79.9 to 80.1 is penalized identically to a shift from 50 to 95 under 0/1 loss. |
| **Class Imbalance** | Handled naturally as skewed target distributions; modeled via continuous loss functions (MSE, MAE, Huber). | Severe minority class penalties; poor water quality samples typically comprise $<5\%$ of monitoring records, distorting unweighted cross-entropy. |
| **Decision Support** | Provides precise numerical values required for regulatory deficit and trend analysis. | Directly provides operational classification suitable for public alert dashboards. |
| **Primary Metrics** | MAE, RMSE, Mean Absolute Percentage Error (MAPE), Coefficient of Determination ($R^2$). | Macro F1-score, Balanced Accuracy, Cohen's Kappa, ROC-AUC, Brier score / calibration curves. |

In academic research, authors frequently convert continuous WQI scores into discrete classes to report inflated classification accuracy ($>95\%$). However, discretizing an intrinsically continuous physical score introduces artificial label noise at class boundaries and conceals systematic model bias. Following our project baseline, `CCME_Values` is formulated strictly as a continuous regression task.

### 2.3 Environmental Data Challenges: Missingness, Spatial Variability, and Temporal Non-Stationarity
Environmental datasets collected across real-world hydrological networks present unique statistical challenges that violate standard independent and identically distributed (i.i.d.) machine learning assumptions:

1. **Missing Data Mechanisms and Upstream Imputation:** Environmental monitoring stations frequently experience sensor biofouling, battery depletion, transmission dropouts, and intermittent laboratory sampling intervals. Missingness rarely conforms to Missing Completely at Random (MCAR). More commonly, data is Missing at Random (MAR)—dependent on hydrological conditions (e.g., sensors submerged during floods)—or Missing Not at Random (MNAR), where instruments clip out-of-range concentrations. A pervasive flaw in published studies is the uncritical application of global mean or median imputation across the full dataset prior to model partitioning, which introduces cross-split data leakage and distorts joint covariance structures.
2. **Spatial Autocorrelation and Regional Heterogeneity:** Water monitoring stations are organized hierarchically within dendritic stream networks, river basins, and national administrative jurisdictions. Observations collected within the same sub-catchment share unmeasured geological, soil, and land-use covariates. Models validated using random row-level splitting suffer from spatial autocorrelation leakage, learning to memorize specific station coordinates rather than generalizable chemical relationships (Roberts et al., 2017).
3. **Temporal Non-Stationarity and Seasonality:** Water parameters exhibit strong seasonal cyclicity (e.g., temperature-dependent oxygen solubility, winter de-icing salt runoff, spring agricultural fertilizer flushes). Long-term multi-decadal records are subject to structural regime shifts driven by climate change and upgraded wastewater treatment standards, meaning historical data distributions may diverge substantially from contemporary conditions.

---

## 3. Random Forest for Water Quality Prediction

### 3.1 Algorithm Mechanics: Bagging, Random Subspaces, and Variance Reduction
The Random Forest (RF) algorithm, introduced by Breiman (2001), is a non-parametric ensemble learning method constructed from decorrelated decision trees. Individual classification and regression trees (CART) are low-bias, high-variance estimators: a single unconstrained tree grows deep to fit intricate local patterns in training data, making it highly sensitive to training sample perturbations.

Random Forest achieves substantial variance reduction without inflating bias through two distinct sources of randomness:

1. **Bootstrap Aggregation (Bagging):** For an ensemble of $B$ trees, each tree $T_b$ ($b = 1, \dots, B$) is trained on an independent bootstrap sample $\mathcal{D}_b$ of size $N$ drawn uniformly with replacement from the training dataset $\mathcal{D}$. Statistically, each bootstrap sample contains approximately $1 - e^{-1} pprox 63.2\%$ of the original observations, leaving the remaining $pprox 36.8\%$ as Out-of-Bag (OOB) observations.
2. **Random Subspace Method (Feature Subsampling):** At every candidate split node within each tree, rather than evaluating the optimal split across all $p$ available input features, the algorithm restricts the search to a randomly chosen feature subset of size $m \ll p$ (conventionally $m = \lfloor p/3 floor$ for regression).

The mathematical justification for ensemble variance reduction is demonstrated by considering $B$ identically distributed, unbiased tree predictors, each having variance $\sigma^2$ and positive pairwise correlation $ho = 	ext{Corr}(T_i(X), T_j(X))$ for $i 
e j$. The variance of the ensemble average $ar{T}(X) = rac{1}{B} \sum_{b=1}^B T_b(X)$ is:
$$	ext{Var}(ar{T}(X)) = ho \sigma^2 + rac{1 - ho}{B} \sigma^2$$
As the number of ensemble trees grows ($B 	o \infty$):
$$\lim_{B 	o \infty} 	ext{Var}(ar{T}(X)) = ho \sigma^2$$

While standard bagging reduces the second term $rac{1-ho}{B}\sigma^2$, it cannot reduce the asymptotic limit $ho \sigma^2$. The random feature subsampling mechanism forcibly decorrelates individual trees by preventing dominant predictors (such as Orthophosphate in water quality indices) from being selected at the root of every tree, thereby driving $ho$ downward and lowering the irreducible ensemble variance floor.

### 3.2 Out-of-Bag (OOB) Generalization and Feature Importance
- **OOB Error Estimation:** Because each sample $i$ is omitted from roughly $36.8\%$ of the bootstrap training sets, evaluating each sample only across trees where it was out-of-bag provides an unbiased estimate of generalization performance:
  $$	ext{MSE}_{	ext{OOB}} = rac{1}{N} \sum_{i=1}^N \left( y_i - rac{1}{|B_{(i)}|} \sum_{b \in B_{(i)}} T_b(x_i) ight)^2$$
  where $B_{(i)}$ denotes the set of trees where sample $i$ was not selected during bootstrapping.
- **Variable Importance Measures:**
  1. *Mean Decrease in Impurity (MDI / Gini Importance):* Computes the total sum of impurity reductions (variance reduction in regression) brought by splits on feature $j$ across all trees. MDI is computationally efficient but notoriously biased toward continuous variables and features with high cardinality (Strobl et al., 2007).
  2. *Mean Decrease in Accuracy (MDA / Permutation Importance):* Evaluates the degradation in OOB prediction accuracy when the values of feature $j$ are randomly permuted across OOB samples, breaking its true relationship with the target. MDA provides a much more robust, uncorrupted metric of true feature utility.

### 3.3 Empirical Studies in Water Resources
Numerous empirical studies validate Random Forest as an exceptionally strong baseline for tabular water quality modeling:
- **Ahmed et al. (2019)** benchmarked RF against Artificial Neural Networks (ANN), Support Vector Machines (SVM), and Decision Trees across diverse Malaysian river basins to predict DO and WQI. RF achieved superior stability (lower RMSE and higher $R^2 pprox 0.88$) and exhibited resilience against sensor noise.
- **Bui et al. (2020)** investigated tree-based ensembles and hybrid neuro-fuzzy systems for predicting WQI across 12 sampling stations in the Han River basin, South Korea, reporting that tree ensembles achieved $R^2 = 0.92$ and consistently outperformed unregularized multi-layer perceptrons on small-to-medium sample sizes.
- **Aldhyani et al. (2020)** evaluated RF and Naive Bayes for water quality classification across Indian river systems, noting that RF handled mixed categorical and continuous tabular distributions without requiring feature scaling or distributional normality.

---

## 4. Gradient Boosting and XGBoost

### 4.1 Gradient Boosting Mechanics: Functional Gradient Descent
Gradient Boosting, established by Friedman (2001), departs fundamentally from the independent averaging of bagging. Instead of building independent trees in parallel, gradient boosting constructs an additive ensemble of weak learners (typically shallow regression trees) sequentially, minimizing a differentiable loss function $\mathcal{L}(y, \hat{y})$ via gradient descent in function space:
$$\hat{y}_i^{(M)} = \sum_{m=1}^M f_m(x_i), \quad f_m \in \mathcal{F}$$
At stage $m$, given the existing ensemble prediction $\hat{y}_i^{(m-1)}$, the algorithm computes the negative gradient of the loss function with respect to the current predictions—the pseudo-residuals:
$$r_{im} = -\left[ rac{\partial \mathcal{L}(y_i, \hat{y})}{\partial \hat{y}} ight]_{\hat{y} = \hat{y}_i^{(m-1)}}$$
A new regression tree $f_m(x)$ is fitted to predict these pseudo-residuals $r_{im}$. To prevent greedy, non-generalizing updates and control variance, shrinkage (learning rate $\eta \in (0, 1]$) scales the contribution of each newly added tree:
$$\hat{y}_i^{(m)} = \hat{y}_i^{(m-1)} + \eta f_m(x_i)$$

### 4.2 XGBoost: Algorithmic Innovations and Regularization
Chen and Guestrin (2016) introduced XGBoost (Extreme Gradient Boosting), resolving computational bottlenecks and structural overfitting in classical gradient boosting through key mathematical and architectural innovations:

1. **Second-Order Taylor Approximation:** XGBoost approximates the objective function at step $m$ using both first-order gradients $g_i$ and second-order Hessians $h_i$:
   $$\mathcal{L}^{(m)} pprox \sum_{i=1}^N \left[ \mathcal{L}(y_i, \hat{y}_i^{(m-1)}) + g_i f_m(x_i) + rac{1}{2} h_i f_m^2(x_i) ight] + \Omega(f_m)$$
   where $g_i = \partial_{\hat{y}^{(m-1)}} \mathcal{L}(y_i, \hat{y}_i^{(m-1)})$ and $h_i = \partial^2_{\hat{y}^{(m-1)}} \mathcal{L}(y_i, \hat{y}_i^{(m-1)})$. For standard squared-error loss $\mathcal{L} = rac{1}{2}(y - \hat{y})^2$, $g_i = (\hat{y}_i^{(m-1)} - y_i)$ and $h_i = 1$.
2. **Explicit Tree Complexity Regularization:** XGBoost penalizes tree complexity directly in the objective function via:
   $$\Omega(f) = \gamma T + rac{1}{2}\lambda \sum_{j=1}^T w_j^2 + lpha \sum_{j=1}^T |w_j|$$
   where $T$ is the number of terminal leaf nodes, $w_j$ represents the continuous output weight assigned to leaf $j$, $\gamma$ acts as a pruning threshold (minimum loss reduction required to split a node), and $\lambda, lpha$ represent $L_2$ and $L_1$ weight penalties respectively.
3. **Exact and Approximate Split Finding:** XGBoost introduces an exact greedy algorithm for smaller data and a quantile-sketch approximate algorithm for massive datasets, evaluating split gain via:
   $$	ext{Gain} = rac{1}{2} \left[ rac{G_L^2}{H_L + \lambda} + rac{G_R^2}{H_R + \lambda} - rac{(G_L + G_R)^2}{H_L + H_R + \lambda} ight] - \gamma$$
4. **Column Subsampling and Missing Value Handling:** Implements random feature subsampling at the tree and node levels (analogous to Random Forest) and includes a unified default split direction for missing values, routing missing observations into the leaf that maximizes training loss reduction.

### 4.3 Controlled Comparisons: Random Forest vs. XGBoost
Empirical studies in tabular machine learning frequently debate whether XGBoost universally outperforms Random Forest:
- **Bias-Variance Profile:** Random Forest is fundamentally a variance-reduction technique that rarely overfits when tree count $B$ increases, making it exceptionally robust to default hyperparameters. XGBoost is primarily a bias-reduction technique that fits intricate higher-order feature interactions, but it can easily overfit noisy tabular data if hyperparameters ($\eta$, max depth, subsampling, $\gamma$) are poorly tuned.
- **The Necessity of Controlled Benchmarking:** Much of the existing environmental literature claims gradient boosting superiority based on flawed methodology: researchers tune XGBoost through hundreds of Bayesian optimization iterations while evaluating Random Forest with default scikit-learn settings, or they compare models across disparate train/validation splits. A fair, reproducible comparison requires:
  1. Identical, immutable cross-validation split indices.
  2. Unified preprocessing and missingness pipelines.
  3. Equivalent hyperparameter search budgets.
  4. Scoring via identical, multi-faceted evaluation metrics on untouched holdout test data.

---

## 5. Feature Engineering and Data Leakage

### 5.1 Hydrochemical Domain Feature Engineering
Raw environmental measurements rarely enter machine learning pipelines in optimal mathematical forms. Incorporating hydrochemical domain principles enhances tree partition efficiency:
1. **Stoichiometric and Chemical Ratios:** Ecological water quality is governed by nutrient balances and chemical reactions rather than absolute concentrations alone:
   - *Total Nitrogen to Total Phosphorus ($N:P$) Ratio:* Dictates aquatic algal biomass and phosphorus-limiting vs. nitrogen-limiting eutrophication states.
   - *Biochemical Oxygen Demand to Chemical Oxygen Demand ($BOD:COD$) Ratio:* Measures the biodegradability fraction of organic pollutants in wastewater discharge.
   - *Ammoniacal Nitrogen to Total Nitrogen ($NH_4:TN$):* Distinguishes raw sewage discharge from oxidized agricultural runoff.
2. **Non-Linear Transformations:** Environmental pollutant concentrations frequently exhibit heavy right-skewed, log-normal distributions. While decision trees are invariant to monotonic univariate scaling, extreme outliers compress node split searches. Applying $\log(1 + x)$ or Box-Cox transformations reduces extreme skewness and kurtosis, accelerating gradient convergence and benefiting downstream linear and neural models.
3. **Spatiotemporal Encodings:** Transforming string dates into cyclic harmonic features ($\sin(2\pi \cdot 	ext{day}/365)$, $\cos(2\pi \cdot 	ext{day}/365)$) captures seasonal hydrological cycles without introducing artificial step boundaries between December 31 and January 1. Catchment-level aggregations (e.g., historical station median pollutant levels) capture unmeasured geogenic baselines.

### 5.2 The Taxonomy of Data Leakage in Machine Learning
Data leakage occurs when information from outside the training dataset is used, directly or indirectly, to create the model, leading to overly optimistic evaluation and catastrophic real-world failure (Kaufman et al., 2012). Environmental pipelines are prone to four primary leakage modes:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        DATA LEAKAGE TAXONOMY                           │
├──────────────────────────┬─────────────────────────────────────────────┤
│ 1. Direct Label Leakage  │ Downstream derivative of target fed as input │
│                          │ Example: CCME_WQI categorical bins in X     │
├──────────────────────────┼─────────────────────────────────────────────┤
│ 2. Target Formula        │ Input features are direct mathematical      │
│    Inversion Leakage     │ determinants of a calculated target index   │
│                          │ Example: 8 Ireland standard chemicals in WQI│
├──────────────────────────┼─────────────────────────────────────────────┤
│ 3. Preprocessing Leakage │ Scalers/imputers fitted across combined     │
│                          │ train + val + test dataset                  │
│                          │ Example: Global median fill before split    │
├──────────────────────────┼─────────────────────────────────────────────┤
│ 4. Spatiotemporal        │ Random k-fold splitting on autocorrelated   │
│    Validation Leakage    │ sensor stations and time-series records     │
│                          │ Example: Same station in train & test       │
└──────────────────────────┴─────────────────────────────────────────────┘
```

1. **Direct Label Leakage:** Feeding features that are deterministic post-hoc transformations of the target variable. In water quality prediction, passing `CCME_WQI` (which categorizes `CCME_Values` into five discrete bins) directly leaks the target's numerical intervals into $X$.
2. **Target Formula Inversion Leakage:** When predicting an index $y = \mathcal{F}(x_1, \dots, x_k)$, feeding the exact contributing inputs $\{x_1, \dots, x_k\}$ turns the machine learning problem into mathematical equation emulation. High empirical performance ($R^2 > 0.98$) reflects only the capacity of tree splits to reconstruct the piecewise CCME geometry, not a learned environmental mapping.
3. **Preprocessing Leakage:** Computing summary statistics (imputation medians, standardization means/variances, categorical target encodings) across the entire dataset prior to splitting. Preprocessing must be treated as a fitted component of the model pipeline, computed strictly on training folds and applied out-of-sample to validation and test folds.
4. **Spatiotemporal Validation Leakage:** Partitioning time-series or spatially grouped records using random IID cross-validation. When measurements from station $A$ on day $t$ appear in the training set and station $A$ on day $t+1$ appears in the test set, models exploit high autocorrelation to interpolate values, masking an inability to generalize to unseen geographic catchments (Roberts et al., 2017).

---

## 6. Research Gap and Project Motivation

### 6.1 Critical Synthesis of Limitations in Published Studies
A comprehensive review of the contemporary literature on machine learning for water quality index prediction reveals five pervasive methodological deficiencies:

1. **Unacknowledged Formula Leakage:** The vast majority of published studies train models using the identical chemical parameters that constitute the index formula, reporting high predictive scores ($R^2 > 0.95$) without explicitly acknowledging that the model is functioning purely as an emulator of a closed-form formula.
2. **Pervasive Preprocessing Contamination:** Studies routinely apply global imputation strategies (such as full-table median imputation) to resolve sensor dropouts, silently leaking test set distributions into training routines and obscuring missingness mechanics.
3. **Absence of Shared, Controlled Benchmarks:** Comparison between models (e.g., Random Forest vs. XGBoost vs. Neural Networks) is rarely conducted on strictly identical, persisted train/val/test splits with equal hyperparameter search budgets. Most claims of boosting superiority stem from uneven tuning effort.
4. **Ignoring Target Distribution Censoring:** Environmental indices frequently exhibit severe ceiling effects. In the global CCME dataset, **$35.92\%$ of all samples sit exactly at the ceiling value of $100.0$**. Standard regression models evaluated purely on aggregate RMSE or $R^2$ hide significant boundary distortions, failing to report stratified performance between at-ceiling and below-ceiling observations.
5. **Lack of Explainability and Physical Verification:** High-performing models are rarely interrogated through modern game-theoretic interpretability frameworks (such as SHAP), leaving regulatory authorities with unverified black-box predictions that cannot be explained in environmental terms.

### 6.2 Project Motivation and Phase 1 Architecture
Our project is designed specifically to resolve these methodological gaps through a rigorous, reproducible Phase 1 architecture:

1. **Explicit Problem Formulation and Boundary Analysis:**
   - Formal recognition of `CCME_Values` as a continuous regression target, avoiding artificial discretization noise.
   - Comprehensive documentation of target ceiling censoring ($35.92\%$ at $100.0$) and stratified evaluation.
2. **Strict Leakage Prevention Framework:**
   - Immediate, hard exclusion of `CCME_WQI` to eliminate direct label leakage.
   - Formal ablation experiments comparing base chemical inputs (formula emulation regime) against spatiotemporal, metadata, and engineered chemical ratios (generalization regime).
   - Upstream median-imputation detection: flagging synthetic fill values via `<col>__imputed_upstream` rather than inheriting them silently.
3. **Fair, Byte-Identical Experimental Benchmark:**
   - Centralized random seed (`SEED = 42`) and persisted index partitions (`data/splits/`) ensuring that the Naive Baseline, Random Forest, and XGBoost evaluate on identical training, validation, and untouched holdout test observations.
   - Preprocessing pipelines (`src/preprocess.py`) fitted strictly on training data.
   - Unified evaluation contract (`src/evaluate_api.py`) standardizing metric calculation across all team members.
4. **Interpretable Model Explainability:**
   - Global and local SHAP analysis (`notebooks/07_shap.ipynb`) to verify feature contribution profiles against known chemical and ecological principles.

---

## 7. Bibliographic Citations (`docs/refs.bib`)

Every scientific statement and theoretical claim in this review is mapped to verified bibliographic records in `docs/refs.bib`:

- **Abbasi, T., & Abbasi, S. A. (2012).** *Water Quality Indices*. Elsevier.
- **Ahmed, A. N., et al. (2019).** Machine learning methods for better water quality prediction. *Journal of Hydrology*, 578, 124084.
- **Aldhyani, T. H. H., et al. (2020).** Water quality prediction using artificial intelligence algorithms. *Applied Water Science*, 10(10), 228.
- **Breiman, L. (2001).** Random Forests. *Machine Learning*, 45(1), 5–32.
- **Brown, R. M., et al. (1970).** A Water Quality Index—Do We Dare? *Water & Sewage Works*, 117(10), 339–343.
- **Bui, D. T., et al. (2020).** Improving prediction of water quality indices using novel hybrid machine-learning algorithms. *Science of The Total Environment*, 721, 137612.
- **Canadian Council of Ministers of the Environment (CCME). (2001).** *Canadian Water Quality Guidelines for the Protection of Aquatic Life: CCME Water Quality Index 1.0, User's Manual*. CCME, Winnipeg.
- **Chen, T., & Guestrin, C. (2016).** XGBoost: A Scalable Tree Boosting System. *ACM SIGKDD*, 785–794.
- **Friedman, J. H. (2001).** Greedy function approximation: A gradient boosting machine. *The Annals of Statistics*, 29(5), 1189–1232.
- **Goodfellow, I., Bengio, Y., & Courville, A. (2016).** *Deep Learning*. MIT Press.
- **Kaufman, S., et al. (2012).** Leakage in Data Mining: Formulation, Detection, and Avoidance. *ACM TKDD*, 6(4), 15:1–15:21.
- **Lumb, A., et al. (2011).** A Review of Genesis and Evolution of Water Quality Index (WQI) and Its Applications. *Water Quality, Exposure and Health*, 3(1), 11–24.
- **Lundberg, S. M., & Lee, S.-I. (2017).** A Unified Approach to Interpreting Model Predictions. *NeurIPS*, 30, 4765–4774.
- **Roberts, D. R., et al. (2017).** Cross-validation strategies for data with temporal, spatial, hierarchical or phylogenetic structure. *Ecography*, 40(8), 913–929.
