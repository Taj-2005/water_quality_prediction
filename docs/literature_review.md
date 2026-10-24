# Literature Review: Water Quality Prediction Using Machine Learning

**Author:** Himanshu Pandey  
**Role:** Literature Review, Feature Engineering, Random Forest  
**Project:** CSAI Project 18 — Water Quality Prediction (Phase 1)  
**BibTeX Source:** `docs/refs.bib`  

---

## 1. Water Quality Assessment and Water Quality Indices (WQI)

### 1.1 Rationale for Systematic Water Quality Assessment
Surface freshwater ecosystems are subject to compounding pressures from industrial effluent discharge, agricultural nutrient runoff, municipal wastewater, and hydrologic variability. Routine ecological monitoring requires measuring a wide spectrum of physical, chemical, and biological analytes:
- **Dissolved Oxygen (DO):** Governs aquatic respiration and fish survival; deficits indicate severe organic pollution.
- **Biochemical Oxygen Demand (BOD):** Measures biodegradable organic matter consuming dissolved oxygen during microbial decomposition.
- **Ammoniacal Nitrogen ($NH_3/NH_4^+$), Nitrate ($NO_3^-$), and Orthophosphate ($PO_4^{3-}$):** Drive eutrophication, toxic algal blooms, and secondary hypoxia.
- **pH and Water Temperature:** Dictate chemical equilibria, gas solubility (e.g., Henry's law for oxygen), and heavy metal bioavailability (Abbasi & Abbasi, 2012).

Because individual water samples often satisfy standards for some analytes while simultaneously violating ecological thresholds for others, reporting raw chemical matrices creates communication barriers for environmental managers and regulatory bodies.

### 1.2 Genesis and Structure of Water Quality Indices
To transform multidimensional water analyses into actionable metrics, composite **Water Quality Indices (WQIs)** were formulated, originating with Horton (1965) and formalized by Brown et al. (1970) for the National Sanitation Foundation (NSF WQI). A WQI aggregates disparate physical and chemical measurements into a single, dimensionless score (typically on a $0	ext{--}100$ scale). Subsequent global adaptations include the Oregon WQI, the British Columbia WQI, weighted arithmetic WQIs, and the Canadian Council of Ministers of the Environment Water Quality Index (CCME WQI) (Lumb et al., 2011).

Classical index formulations historically suffer from structural aggregation pitfalls:
- **Eclipsing:** An aggregated index registers an acceptable score despite a single critical toxicant violating lethal environmental thresholds (Abbasi & Abbasi, 2012).
- **Ambiguity:** Divergent constituent combinations yield identical composite scores (Lumb et al., 2011).

### 1.3 The CCME Water Quality Index Methodology: Standard Protocol vs. Row-Level Formulation
The CCME WQI methodology (CCME, 2001) is designed to evaluate guideline non-compliance across three orthogonal geometric dimensions:

1. **Scope ($F_1$):** Percentage of water quality variables failing to meet guidelines at least once during the sampling window relative to the total number of variables evaluated:
   $$F_1 = \left( rac{	ext{Number of failed variables}}{	ext{Total number of variables}} ight) 	imes 100$$

2. **Frequency ($F_2$):** Percentage of individual analytical tests that violate guideline standards across all samples and sampling times:
   $$F_2 = \left( rac{	ext{Number of failed tests}}{	ext{Total number of tests}} ight) 	imes 100$$

3. **Amplitude ($F_3$):** Amount by which non-compliant measurements exceed permissible standards.
   - For parameters where higher values exceed the standard (nutrients, BOD):
     $$	ext{excursion}_i = \left( rac{	ext{value}_i}{	ext{standard}_i} ight) - 1$$
   - For parameters where values below standard represent failure (dissolved oxygen):
     $$	ext{excursion}_i = \left( rac{	ext{standard}_i}{	ext{value}_i} ight) - 1$$
   - Normalized sum of excursions ($nse$):
     $$nse = rac{\sum_{i=1}^{	ext{failed}} 	ext{excursion}_i}{	ext{Total number of tests}}$$
   - Asymptotic hyperbolic transformation bounding $F_3$ strictly in $[0, 100]$:
     $$F_3 = rac{nse}{0.01 \cdot nse + 0.01}$$

The composite index is calculated as the Euclidean vector distance from the theoretical worst-case impaired point $(F_1=100, F_2=100, F_3=100)$, normalized by $\sqrt{100^2 + 100^2 + 100^2} = \sqrt{30000} pprox 173.2$:
$$	ext{CCME\_Values} = 100 - \left( rac{\sqrt{F_1^2 + F_2^2 + F_3^2}}{1.732} ight)$$

This maps to five categorical quality bands: **Excellent (95–100)**, **Good (80–94)**, **Fair (65–79)**, **Marginal (45–64)**, and **Poor (0–44)**.

#### Critical Distinction: Official Protocol vs. Dataset-Specific Row-Level Calculation
An essential distinction must be maintained between official regulatory methodology and dataset-specific calculation:
- **Official CCME Protocol (CCME, 2001):** The official standard explicitly mandates an aggregative temporal monitoring window requiring **at least four variables sampled on at least four separate monitoring events** (minimum 16 analytical tests). The CCME index is conceptually designed to assess a river reach or lake over a season or year, capturing temporal failure frequency ($F_2$).
- **Dataset-Specific Row-Level Adaptation:** In the project's harmonized surface-water monitoring dataset (Figshare DOI: 10.6084/m9.figshare.27800394), the CCME WQI is reported on a per-row, instantaneous basis ($N=1$ per timestamp). When computed on a single measurement event across $P$ variables, the total number of tests equals $P$, and the number of failed tests equals the number of failed variables ($k$). Consequently, **Frequency ($F_2$) collapses algebraically to Scope ($F_1$)**, leaving only two independent factors ($F_1$ and $F_3$). While this adaptation allows continuous tabular regression on individual sampling rows, it represents an empirical modification rather than the standard multi-period CCME protocol.
- **Provisional Status:** Whether this dataset strictly followed this exact single-row CCME algebraic implementation is documented in `docs/target_definition.md` (citing upstream notebook `Data_Processing_Harmonization.ipynb`), but remains **provisional** until the source harmonization scripts and raw 305 MB CSV can be executed and verified locally.

### 1.4 Index Calculation vs. Predictive Modeling
- **Index Calculation:** A closed-form, deterministic mathematical mapping $\mathcal{F}: \mathbb{R}^P 	o [0, 100]$. Given contemporaneous laboratory measurements, calculating CCME WQI requires no estimation or model training.
- **Predictive Modeling:** Approximates an unknown or unobserved physical data-generating process $\hat{f}(X) pprox y$. Machine learning is applied when:
  1. Inferring expensive, delayed lab measurements (e.g., 5-day BOD) from rapid, in-situ optical/electrochemical sensors (pH, conductivity).
  2. Forecasting future water quality states $y_{t+\Delta t}$ from historical time series.
  3. Spatial interpolation across ungauged river catchments.
  4. Constructing rapid computational emulators of complex numerical simulations.

When a model is trained using the exact 8 chemical/physical parameters that define the CCME formula to predict `CCME_Values`, it acts as a **function emulator of an algebraic equation**, not a physical environmental predictor. Acknowledging this distinction is critical to prevent conflating formula fitting with environmental generalization.

---

## 2. Machine Learning for Water Quality Prediction

### 2.1 Scope of Supervised Learning in Water Resources
Supervised machine learning captures high-dimensional, non-linear hydrochemical interactions that conventional linear models and physical hydrodynamic equations struggle to scale across large watersheds (Ahmed et al., 2019; Tyralis et al., 2019). Modeling objectives are divided into:
1. **Constituent Prediction:** Predicting unmeasured or delayed parameters from sensor surrogates.
2. **Index Prediction:** Directly modeling composite indices (e.g., WQI) to provide automated watershed health scoring (Bui et al., 2020).

### 2.2 Regression vs. Classification Objectives

| Criterion | Continuous Regression ($y \in [0, 100]$) | Categorical Classification ($y \in \{	ext{Poor}, \dots, 	ext{Excellent}\}$) |
| :--- | :--- | :--- |
| **Information Retention** | Retains all variance, fine-grained gradients, and trajectory dynamics. | Discretizes continuous space into discrete bins, discarding within-class variance. |
| **Threshold Sensitivity** | Smooth, continuous penalization across all ranges. | Discontinuous 0/1 penalty; a shift across 79.9 to 80.1 is penalized identically to 50 to 95. |
| **Class Imbalance** | Handled naturally as continuous target skewness (MSE, MAE, Huber loss). | Severe minority class penalty; Poor water quality typically comprises $<5\%$ of monitoring records. |
| **Primary Metrics** | MAE, RMSE, Mean Absolute Percentage Error (MAPE), Coefficient of Determination ($R^2$). | Macro F1-score, Balanced Accuracy, Cohen's Kappa, ROC-AUC, Brier score / calibration curves. |

Discretizing intrinsically continuous water quality indices into qualitative classes discards variance near regulatory transition thresholds. In accordance with the project baseline, `CCME_Values` is formulated strictly as a continuous regression task. In Phase 2, this continuous formulation will provide the foundation for deep neural network architectures (MLPs) as explored in tabular deep learning literature (Goodfellow et al., 2016).

### 2.3 Environmental Data Challenges: Missingness, Spatial Variability, and Non-Stationarity
- **Missing Data and Upstream Imputation:** Sensors frequently experience biofouling, battery drain, and intermittent laboratory testing schedules. In real datasets, missingness rarely follows Missing Completely at Random (MCAR). Applying global mean or median imputation across the full dataset prior to splitting leaks test set statistics into training sets (Kaufman et al., 2012).
- **Spatial Autocorrelation:** Stations nested within river networks share unmeasured catchment, soil, and upstream discharge covariates. Standard random row-level train/test splits result in spatial leakage, where models memorize station identities rather than generalizable chemical relationships (Roberts et al., 2017).
- **Temporal Non-Stationarity:** Hydrological time series exhibit strong seasonality (e.g., temperature-dependent oxygen saturation) and multi-decadal shifts from land-use changes, meaning historical patterns may not represent contemporary aquatic conditions.

---

## 3. Random Forest for Water Quality Prediction

### 3.1 Algorithm Mechanics: Bagging, Random Subspaces, and Variance Reduction
Random Forest (Breiman, 2001) is an ensemble of decorrelated decision trees (CART). Individual decision trees are low-bias, high-variance estimators that overfit local training noise. Random Forest achieves substantial variance reduction without inflating bias through two mechanics:

1. **Bootstrap Aggregation (Bagging):** For an ensemble of $B$ trees, each tree $T_b$ ($b = 1, \dots, B$) is trained on an independent bootstrap sample $\mathcal{D}_b$ of size $N$ drawn uniformly with replacement from training data $\mathcal{D}$. Each bootstrap sample contains $pprox 63.2\%$ unique observations, leaving $pprox 36.8\%$ as Out-of-Bag (OOB) samples.
2. **Random Subspace Method:** At every split node within each tree, split optimization is restricted to a randomly selected subset of $m \ll p$ candidate features (typically $m = \lfloor p/3 floor$ for regression).

The mathematical basis for ensemble variance reduction is demonstrated by considering $B$ identically distributed tree estimators, each with variance $\sigma^2$ and positive pairwise correlation $ho = 	ext{Corr}(T_i(X), T_j(X))$ for $i 
e j$. The ensemble average variance $ar{T}(X) = rac{1}{B} \sum_{b=1}^B T_b(X)$ is:
$$	ext{Var}(ar{T}(X)) = ho \sigma^2 + rac{1 - ho}{B} \sigma^2$$
As tree count increases ($B 	o \infty$):
$$\lim_{B 	o \infty} 	ext{Var}(ar{T}(X)) = ho \sigma^2$$
Standard bagging reduces the second term $rac{1-ho}{B}\sigma^2$, but cannot reduce $ho \sigma^2$. The random feature subsampling mechanism decorrelates the trees by preventing dominant predictors (such as Orthophosphate in CCME WQI) from appearing at the root of every tree, thereby lowering $ho$ and reducing the ensemble variance floor.

### 3.2 Out-of-Bag (OOB) Error and Variable Importance
- **OOB Generalization:** Evaluating each sample $i$ using only the subset of trees where sample $i$ was omitted from bootstrapping provides an internal, unbiased estimate of generalization error without requiring a separate validation set (Breiman, 2001).
- **Feature Importance:**
  - *Mean Decrease in Impurity (MDI / Gini Importance):* Sum of variance reductions across all splits on feature $j$. MDI is computationally fast but notoriously biased toward continuous variables and features with high cardinality (Strobl et al., 2007).
  - *Mean Decrease in Accuracy (MDA / Permutation Importance):* Measures the drop in OOB accuracy when values of feature $j$ are randomly shuffled across OOB samples, breaking its true relationship with the target. MDA provides an unbiased metric of true predictive value.

### 3.3 Empirical Studies in Water Resources
- **Ahmed et al. (2019)** investigated machine learning methods for water quality constituent prediction in the Johor River Basin, Malaysia. They evaluated Multi-Layer Perceptrons (MLP), Radial Basis Function (RBF) networks, and Adaptive Neuro-Fuzzy Inference Systems (ANFIS). Their findings demonstrated that integrating a wavelet de-noising technique with ANFIS (WDT-ANFIS) improved the modeling of non-linear hydrological series by filtering high-frequency measurement noise prior to inference.
- **Bui et al. (2020)** evaluated four tree-based algorithms (Random Forest, M5P, Random Tree, and Reduced Error Pruning Tree) combined with three meta-learning ensemble frameworks (Bagging, Cross-Validation Parameter Selection, and Randomizable Filtered Classification) for predicting the Iran Water Quality Index ($	ext{IRWQI}_{sc}$) in the Talar catchment, Iran. The authors reported that the hybrid **Bagging-Random Tree (BA-RT)** model achieved the highest predictive accuracy among the evaluated configurations ($R^2 = 0.941$, $	ext{RMSE} = 2.71$), demonstrating that bagging ensembles effectively reduce prediction variance across multi-year hydrological data.
- **Tyralis et al. (2019)** conducted a comprehensive review of Random Forests in hydrology and water resources, synthesizing findings across over 200 studies. They concluded that Random Forest is widely favored across water science due to its low parameter-tuning burden, resistance to overfitting when tree count grows large, and intrinsic out-of-bag validation capabilities.

---

## 4. Gradient Boosting and XGBoost

### 4.1 Gradient Boosting Mechanics: Functional Gradient Descent
Gradient Boosting (Friedman, 2001) builds an additive model sequentially. Rather than averaging independent trees, it optimizes a differentiable loss function $\mathcal{L}(y, \hat{y})$ via gradient descent in function space:
$$\hat{y}_i^{(M)} = \sum_{m=1}^M f_m(x_i)$$
At step $m$, given previous predictions $\hat{y}_i^{(m-1)}$, the algorithm computes the negative gradient (pseudo-residuals):
$$r_{im} = -\left[ rac{\partial \mathcal{L}(y_i, \hat{y})}{\partial \hat{y}} ight]_{\hat{y} = \hat{y}_i^{(m-1)}}$$
A new regression tree $f_m(x)$ is fitted to predict these residuals, scaled by a shrinkage learning rate $\eta \in (0, 1]$ to control variance:
$$\hat{y}_i^{(m)} = \hat{y}_i^{(m-1)} + \eta f_m(x_i)$$

### 4.2 XGBoost Algorithmic Enhancements
Chen and Guestrin (2016) introduced XGBoost, addressing computational bottlenecks and overfitting through key algorithmic features:
1. **Second-Order Taylor Expansion:** Uses both first-order gradients $g_i$ and second-order Hessians $h_i$ to approximate the objective function.
2. **Explicit Regularization:** Penalizes tree complexity directly in the loss function:
   $$\Omega(f) = \gamma T + rac{1}{2}\lambda \sum_{j=1}^T w_j^2 + lpha \sum_{j=1}^T |w_j|$$
   where $T$ is terminal leaf count, $w_j$ is leaf weight, $\gamma$ is the minimum gain required to split, and $\lambda, lpha$ provide $L_2$ and $L_1$ shrinkage.
3. **Sparsity-Aware Split Finding:** Automatically learns default split pathways for missing values.

### 4.3 Controlled Comparisons: Random Forest vs. XGBoost
- **Bias-Variance Profile:** Random Forest is primarily a variance-reduction technique that rarely overfits as tree count grows, making it robust to default hyperparameters. XGBoost is a bias-reduction technique capable of capturing higher-order interactions, but can overfit if hyperparameters ($\eta$, max depth, subsample, $\gamma$) are improperly tuned.
- **Controlled Benchmarking Necessity:** In environmental literature, comparative claims between tree ensembles are often confounded by unequal experimental conditions (e.g., extensive hyperparameter search on boosting versus default settings on bagging, or inconsistent train/validation partitions). Valid comparison requires:
  1. Identical, immutable cross-validation split indices.
  2. Unified preprocessing and missingness pipelines.
  3. Equivalent hyperparameter search budgets.
  4. Scoring via identical multi-faceted evaluation metrics on untouched holdout test data.

---

## 5. Feature Engineering and Data Leakage

### 5.1 Hydrochemical Domain Feature Engineering and Schema Verification
Feature engineering proposals must be evaluated against the actual columns in the project's 14-column dataset (`Combined_dataset.csv`, Figshare DOI: 10.6084/m9.figshare.27800394).

#### Verification of Available Dataset Columns:
The documented schema consists of:
- **Metadata & Spatiotemporal:** `Country`, `Area`, `Waterbody Type`, `Date`
- **Contributing Chemical/Physical Inputs (8 Ireland Standard Variables):** `Ammonia (mg/l)`, `Biochemical Oxygen Demand (mg/l)`, `Dissolved Oxygen (mg/l)`, `Orthophosphate (mg/l)`, `pH (ph units)`, `Temperature (cel)`, `Nitrogen (mg/l)`, `Nitrate (mg/l)`
- **Target and Derived Label:** `CCME_Values` (target), `CCME_WQI` (post-hoc categorical binning)

*(Note: Chemical Oxygen Demand (COD) is absent from the dataset schema. The previously proposed $BOD:COD$ ratio is unfeasible and has been excluded).*

#### Candidate Chemical Features and Numerical Safeguards:
1. **Candidate Nutrient Stoichiometric Ratio ($N:P$ Ratio):**
   - *Candidate Formula:* $[	ext{Nitrogen}] / ([	ext{Orthophosphate}] + \epsilon)$
   - *Scientific Interpretation:* Proposes to capture relative nutrient limitation (nitrogen- vs. phosphorus-limited algal biomass).
   - *Data Validation & Caveats:* In the raw dataset, `Orthophosphate (mg/l)` contains values at or near zero (minimum $-0.004$, 10 negative readings). An arithmetic smoothing term $\epsilon = 0.001	ext{ mg/l}$ prevents division-by-zero, but **smoothing alone does not guarantee scientific validity**. Clipping negative values ($\max(0, x)$) changes the statistical distribution of instrument sensor noise near detection limits; such adjustments require explicit validation against data provenance.
2. **Candidate Inorganic Nitrogen Speciation Ratio:**
   - *Candidate Formula:* $[	ext{Ammonia}] / ([	ext{Nitrate}] + \epsilon)$
   - *Scientific Interpretation:* Evaluates the proportion of reduced (ammonia) to oxidized (nitrate) nitrogen, reflecting nitrification and oxygen consumption.
   - *Data Validation & Caveats:* `Nitrate (mg/l)` reaches $0.000$; requires denominator smoothing and handling of near-zero values.
3. **Candidate Organic Nitrogen Fraction:**
   - *Candidate Formula:* $[	ext{Nitrogen}] - ([	ext{Nitrate}] + [	ext{Ammonia}])$
   - *Scientific Interpretation:* Attempts to approximate the organic nitrogen fraction by subtracting inorganic forms from total nitrogen.
   - *Data Validation & Caveats:* While units are consistent ($	ext{mg/l}$), independent laboratory assay errors can cause the sum of inorganic components to exceed total nitrogen, producing negative values. Truncating or clipping these values alters sensor error structures and must be validated empirically during feature selection.
4. **Non-Linear Skewness Transformations:**
   - *Evidence from EDA:* `Ammonia (mg/l)` exhibits skewness of $11.98$ (kurtosis $221.7$), and `Biochemical Oxygen Demand (mg/l)` exhibits skewness of $10.21$.
   - *Transform:* $\log(1 + \max(0, x))$. Truncating at zero is computationally necessary to prevent `NaN` values under $\log(1+x)$ caused by isolated negative sensor records.
5. **Spatiotemporal Features:**
   - *Cyclic Date Encoding:* Transforming `Date` into $\sin(2\pi \cdot 	ext{day}/365)$ and $\cos(2\pi \cdot 	ext{day}/365)$ models seasonal hydrological cycles.
   - *Station Aggregations:* Per-site historical averages (`Area`) must be computed **strictly on training observations** to prevent data leakage.

### 5.2 Taxonomy of Data Leakage

```
┌────────────────────────────────────────────────────────────────────────┐
│                        DATA LEAKAGE TAXONOMY                           │
├──────────────────────────┬─────────────────────────────────────────────┤
│ 1. Direct Label Leakage  │ Downstream derivative of target fed as input │
│                          │ Example: CCME_WQI categorical bins in X     │
├──────────────────────────┼─────────────────────────────────────────────┤
│ 2. Target Formula        │ Input features are direct mathematical      │
│    Inversion Leakage     │ determinants of a calculated target index   │
│                          │ Example: 8 Ireland standard parameters in X │
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

- **Direct Label Leakage:** Feeding variables that are direct post-hoc transformations of the target (e.g., passing `CCME_WQI` into $X$, exposing target intervals).
- **Target Formula Inversion Leakage:** Feeding the exact 8 chemical/physical parameters that calculate `CCME_Values`.
  - *Temperature Verification:* `Temperature (cel)` **is an active contributing input** to the Ireland CCME formula (threshold = 25.0 °C) as documented in `docs/target_definition.md`. It cannot be classified as a non-contributing feature.
  - *Non-contributing features* are restricted strictly to metadata and spatiotemporal covariates (`Country`, `Area`, `Waterbody Type`, `Date`).
- **Preprocessing Leakage:** Computing summary statistics (medians, scalers) over combined train and test sets (Kaufman et al., 2012). Preprocessing must be fitted strictly on $X_{train}$.
- **Spatiotemporal Validation Leakage:** Random cross-validation on autocorrelated time-series or station records, causing models to interpolate known local readings rather than generalizing (Roberts et al., 2017).

---

## 6. Research Gap and Proposed Project Framework

### 6.1 Critical Synthesis of Limitations and Methodological Risks
The literature on machine learning for water quality assessment reveals three distinct categories of methodological concerns:

1. **Literature-Demonstrated Limitations:**
   - Aggregated indices suffer from intrinsic eclipsing (masking single-parameter toxicity) and ambiguity (multiple parameter states yielding identical scores) as established by Abbasi & Abbasi (2012) and Lumb et al. (2011).
   - In tree ensembles, impurity-based feature importance (Gini / MDI) exhibits well-documented bias toward continuous variables and high-cardinality features, requiring permutation-based importance or SHAP values for valid interpretation (Strobl et al., 2007; Lundberg & Lee, 2017).
2. **General Methodological Risks Supported by Authoritative Sources:**
   - **Data Leakage in Preprocessing:** Fitting missing-value imputations or normalizations across full datasets prior to splitting artificially inflates test performance (Kaufman et al., 2012).
   - **Spatial Autocorrelation Overoptimism:** Random k-fold cross-validation on spatially clustered water monitoring networks allows models to memorize localized catchment features rather than learning transport dynamics, requiring grouped or spatial block cross-validation (Roberts et al., 2017).
3. **Project-Specific Hypotheses Requiring Empirical Investigation:**
   - **Target Ceiling Censoring:** In the project's dataset, **$35.92\%$ of target values are exactly $100.0$**. Standard regression models evaluated purely on aggregate RMSE or $R^2$ may suffer boundary distortions, hypothesizing the need for stratified metrics (at-ceiling vs. below-ceiling).
   - **Upstream Median Imputation Confound:** Exploratory data analysis revealed that median imputation was applied upstream across five of the eight chemical variables, predominantly in England records. Whether models exploit these constant fill values as geographical proxies requires empirical ablation.
   - **Formula Emulation vs. Physical Generalization:** Whether high predictive accuracy on `CCME_Values` reflects equation reconstruction rather than hydrological prediction requires controlled comparison against non-contributing features.

### 6.2 Proposed Experimental Regimes
To address these concerns transparently, our Phase 1 experimental architecture is structured into two proposed comparative evaluation regimes (with no empirical results claimed prior to model execution):

- **Proposed Regime 1 (Formula Emulation Benchmark):**
  - *Design:* Train Random Forest and XGBoost using the 8 chemical/physical variables documented as formula contributors (`Ammonia`, `BOD`, `DO`, `Orthophosphate`, `pH`, `Temperature`, `Nitrogen`, `Nitrate`).
  - *Prerequisite:* Execution is conditional on confirming the upstream target-generation formula and verifying that the dataset reproduces the CCME calculations.
  - *Objective:* Benchmark the capacity of tree ensembles to approximate the non-linear piecewise CCME geometric distance equation.
- **Proposed Regime 2 (Independent Environmental Prediction Benchmark):**
  - *Design:* Exclude all 8 formula-contributing chemical inputs. Train models using strictly non-contributing metadata and spatiotemporal covariates (`Country`, `Area`, `Waterbody Type`, and seasonal date transformations).
  - *Objective:* Evaluate how accurately water quality index categories can be inferred from environmental context alone, while transparently acknowledging the severe physical limitations of predicting chemical indices without contemporaneous chemical assays.

Both regimes will utilize identical, persisted index partitions (`data/splits/`), shared preprocessing fitted strictly on training data (`src/preprocess.py`), and a standardized evaluation contract (`src/evaluate_api.py`).

---

## 7. Bibliographic Records (`docs/refs.bib`)

Every scientific citation in this review is mapped to verified bibliographic records in `docs/refs.bib`:

- **Abbasi, T., & Abbasi, S. A. (2012).** *Water Quality Indices*. Elsevier, Amsterdam. ISBN: 978-0-444-54304-2.
- **Ahmed, A. N., Othman, F. B., Afan, H. A., Ibrahim, R. K., Fai, C. M., Hossain, M. S., Ehteram, M., & Elshafie, A. (2019).** Machine learning methods for better water quality prediction. *Journal of Hydrology*, 578, 124084. [DOI: 10.1016/j.jhydrol.2019.124084](https://doi.org/10.1016/j.jhydrol.2019.124084).
- **Breiman, L. (2001).** Random Forests. *Machine Learning*, 45(1), 5–32. [DOI: 10.1023/A:1010933404324](https://doi.org/10.1023/A:1010933404324).
- **Brown, R. M., McClelland, N. I., Deininger, R. A., & Tozer, R. G. (1970).** A Water Quality Index—Do We Dare? *Water & Sewage Works*, 117(10), 339–343.
- **Bui, D. T., Khosravi, K., Tiefenbacher, J. P., Nguyen, H., & Kazakis, N. (2020).** Improving prediction of water quality indices using novel hybrid machine-learning algorithms. *Science of The Total Environment*, 721, 137612. [DOI: 10.1016/j.scitotenv.2020.137612](https://doi.org/10.1016/j.scitotenv.2020.137612).
- **Canadian Council of Ministers of the Environment (CCME). (2001).** *Canadian Water Quality Guidelines for the Protection of Aquatic Life: CCME Water Quality Index 1.0, User's Manual*. CCME, Winnipeg.
- **Chen, T., & Guestrin, C. (2016).** XGBoost: A Scalable Tree Boosting System. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, pp. 785–794. [DOI: 10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785).
- **Friedman, J. H. (2001).** Greedy function approximation: A gradient boosting machine. *The Annals of Statistics*, 29(5), 1189–1232. [DOI: 10.1214/aos/1013203451](https://doi.org/10.1214/aos/1013203451).
- **Goodfellow, I., Bengio, Y., & Courville, A. (2016).** *Deep Learning*. MIT Press, Cambridge, MA.
- **Horton, R. K. (1965).** An index number system for rating water quality. *Journal of Water Pollution Control Federation*, 37(3), 300–306.
- **Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012).** Leakage in Data Mining: Formulation, Detection, and Avoidance. *ACM Transactions on Knowledge Discovery from Data*, 6(4), 15:1–15:21. [DOI: 10.1145/2382577.2382579](https://doi.org/10.1145/2382577.2382579).
- **Lumb, A., Sharma, T. C., & Bibeault, J.-F. (2011).** A Review of Genesis and Evolution of Water Quality Index (WQI) and Its Applications. *Water Quality, Exposure and Health*, 3(1), 11–24. [DOI: 10.1007/s12403-011-0040-0](https://doi.org/10.1007/s12403-011-0040-0).
- **Lundberg, S. M., & Lee, S.-I. (2017).** A Unified Approach to Interpreting Model Predictions. In *Advances in Neural Information Processing Systems (NeurIPS)*, 30, 4765–4774.
- **Roberts, D. R., Bahn, V., Ciuti, S., Boyce, M. S., Elith, J., Guillera-Arroita, G., Hauenstein, S., Lahoz-Monfort, J. J., Schröder, B., Thuiller, W., & Warton, D. I. (2017).** Cross-validation strategies for data with temporal, spatial, hierarchical or phylogenetic structure. *Ecography*, 40(8), 913–929. [DOI: 10.1111/ecog.02881](https://doi.org/10.1111/ecog.02881).
- **Strobl, C., Boulesteix, A.-L., Zeileis, A., & Hothorn, T. (2007).** Bias in random forest variable importance measures: Illustrations, sources and a solution. *BMC Bioinformatics*, 8(1), 25. [DOI: 10.1186/1471-2105-8-25](https://doi.org/10.1186/1471-2105-8-25).
- **Tyralis, H., Papacharalampous, G., & Langousis, A. (2019).** A Brief Review of Random Forests for Water Scientists and Practitioners and Their Recent History in Water Resources. *Water*, 11(5), 910. [DOI: 10.3390/w11050910](https://doi.org/10.3390/w11050910).
- **Water Quality Research Group. (2024).** *Combined Harmonized Surface-Water Quality Dataset with CCME WQI*. Figshare. [DOI: 10.6084/m9.figshare.27800394](https://doi.org/10.6084/m9.figshare.27800394).
