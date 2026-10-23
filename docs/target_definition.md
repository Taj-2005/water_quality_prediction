# Target Definition

## 1. Target Column

The confirmed target column is:

`CCME_Values`

## 2. Problem Type

**Regression**

The target `CCME_Values` is a continuous numerical index ranging from 0.0 to 100.0 (observed values in the dataset range from 31.304 to 100.0). It is not a discrete class label. Per project guidelines, a continuous index is treated as a regression task and must not be artificially discretized into classes unless explicitly specified by faculty requirements.

## 3. Target Meaning

`CCME_Values` represents the Canadian Council of Ministers of the Environment (CCME) Water Quality Index (WQI). 

It provides a single dimensionless number from 0 to 100 summarizing overall surface water quality based on multiple physical and chemical parameters:
- **95 – 100**: Excellent (water quality is protected with a virtual absence of threat or impairment)
- **80 – 94**: Good (minor degree of threat or impairment)
- **65 – 79**: Fair (water quality is usually protected but occasionally threatened or impaired)
- **45 – 64**: Marginal (water quality is frequently threatened or impaired)
- **0 – 44**: Poor (water quality is almost always threatened or impaired)

In the raw dataset (`Combined_dataset.csv`, 2,827,977 rows), the empirical distribution of `CCME_Values` is:
- Count: 2,827,977
- Minimum: 31.304
- 25th Percentile: 77.153
- Median (50th Percentile): 90.596
- Mean: 85.047
- Standard Deviation: 17.647
- 75th Percentile: 100.000
- Maximum: 100.000

## 4. Target Definition

The CCME Water Quality Index combines three factors of guideline non-compliance:

1. **Scope ($F_1$)**: The percentage of water quality variables that do not meet their respective guidelines at least once during the sampling period:
   $$F_1 = \left( \frac{\text{Number of failed variables}}{\text{Total number of variables}} \right) \times 100$$

2. **Frequency ($F_2$)**: The percentage of individual tests that do not meet guidelines:
   $$F_2 = \left( \frac{\text{Number of failed tests}}{\text{Total number of tests}} \right) \times 100$$
   *Note from dataset source pipeline (`Data_Processing_Harmonization.ipynb`, Cell 30):* Because each observation in the table is an individual measurement row rather than an aggregated time-series window, $F_2$ is set equal to $F_1$.

3. **Amplitude ($F_3$)**: The amount by which failed test values exceed their guidelines. For parameters where higher values exceed the standard:
   $$\text{excursion}_i = \left( \frac{\text{value}_i}{\text{standard}_i} \right) - 1$$
   $$\text{nse} = \frac{\sum_{i=1}^{\text{failed}} \text{excursion}_i}{\text{Total number of variables}}$$
   $$F_3 = \frac{\text{nse}}{0.01 \cdot \text{nse} + 0.01}$$

The final index is computed as the vector distance from the ideal state (100 in three-dimensional factor space):
$$\text{CCME\_Values} = 100 - \left( \frac{\sqrt{F_1^2 + F_2^2 + F_3^2}}{1.732} \right)$$
*(where $\sqrt{3} \approx 1.732$ normalizes the maximum possible distance of $\sqrt{100^2 + 100^2 + 100^2} = 173.2$ to 100).*

In the harmonized dataset pipeline, the 8 evaluated parameter standards (based on Ireland WQI standards) are:
- `Ammonia (mg/l)`: Standard threshold = 0.5 mg/l
- `Biochemical Oxygen Demand (mg/l)`: Standard threshold = 6.0 mg/l
- `Dissolved Oxygen (mg/l)`: Standard threshold = 14.0 mg/l
- `Orthophosphate (mg/l)`: Standard threshold = 0.05 mg/l
- `pH (ph units)`: Standard threshold = 8.5
- `Temperature (cel)`: Standard threshold = 25.0 °C
- `Nitrogen (mg/l)`: Standard threshold = 20.0 mg/l
- `Nitrate (mg/l)`: Standard threshold = 45.0 mg/l

## 5. Feature Candidates

The raw dataset contains 14 columns across 5 countries (England, USA, Ireland, China, Canada):

### Metadata and Spatiotemporal Features
1. `Country` (categorical: 5 unique values — England: 2,129,198, USA: 413,814, Ireland: 235,019, China: 45,997, Canada: 3,949)
2. `Area` (categorical identifier: 61,689 unique monitoring stations/sampling sites)
3. `Waterbody Type` (categorical: 12 unique classes — River, Effluent, Lake, Estuarine, Bay, Sea Water, Canal, Sewage, Marine, Drainage, Transitional, Coastal)
4. `Date` (temporal string: 23,876 unique dates spanning 1953 to 2020)

### Physical and Chemical Features
5. `Ammonia (mg/l)` (continuous float: min -0.005, mean 1.171, median 0.055, max 200.0)
6. `Biochemical Oxygen Demand (mg/l)` (continuous float: min -2.0, mean 4.887, median 2.700, max 255.0)
7. `Dissolved Oxygen (mg/l)` (continuous float: min 0.0, mean 10.008, median 10.200, max 20.0)
8. `Orthophosphate (mg/l)` (continuous float: min -0.004, mean 0.710, median 0.107, max 100.0)
9. `pH (ph units)` (continuous float: min -1.0, mean 7.736, median 7.780, max 30.0)
10. `Temperature (cel)` (continuous float: min -5.26, mean 11.839, median 11.460, max 98.0)
11. `Nitrogen (mg/l)` (continuous float: min 0.0, mean 5.210, median 4.000, max 46.0)
12. `Nitrate (mg/l)` (continuous float: min 0.0, mean 4.767, median 4.500, max 155.0)

### Target and Label Columns
13. `CCME_Values` (continuous float: target column)
14. `CCME_WQI` (categorical: discrete class representation)

## 6. Leakage Analysis

### Direct Label Leakage
- **`CCME_WQI`**: This column was generated directly by thresholding `CCME_Values` (`Excellent` for $\ge 95$, `Good` for $80 \le \text{WQI} < 95$, `Fair` for $65 \le \text{WQI} < 80$, `Marginal` for $45 \le \text{WQI} < 65$, `Poor` for $< 45$). Supplying `CCME_WQI` as an input feature is direct target leakage because it exposes the target's value range to the model.

### Mathematical Inversion / Contributing Formula Leakage
- **The 8 Chemical/Physical Features** (`Ammonia (mg/l)`, `Biochemical Oxygen Demand (mg/l)`, `Dissolved Oxygen (mg/l)`, `Orthophosphate (mg/l)`, `pH (ph units)`, `Temperature (cel)`, `Nitrogen (mg/l)`, `Nitrate (mg/l)`):
  The target `CCME_Values` is a deterministic closed-form mathematical function of these exact 8 features (verified with $0.000000$ reconstruction error against the source formula in `Data_Processing_Harmonization.ipynb`).
  - In academic machine learning, when an index target is computed directly from input measurements, regressing the index on those same measurements constitutes **target formula leakage**. The model does not learn water quality physics; rather, it fits a polynomial / tree-based approximation of the known CCME formula.
  - The project work plan explicitly warns of this:
    > *"If the target is an index computed from the input columns (e.g. a WQI derived from pH, DO and turbidity), then regressing it on those same columns is target leakage. The allowed resolutions are: (1) drop the contributing input columns, or (2) choose a different target, and whichever is chosen is stated explicitly in the report."*
  - The original benchmark notebook (`Technical_Validation.ipynb`) retained these 8 features to predict `CCME_Values` as a regression task, treating the model as a machine learning surrogate / emulator of the CCME index.
  - In our project, per phase requirements, this leakage risk is formally documented here. In Himanshu's feature engineering and Jivit's ablation experiments, the models will be evaluated with clear ablation between base chemical inputs (emulator regime) and external/spatiotemporal/engineered features.

## 7. Feature Exclusions

| Column | Decision | Reason |
| :--- | :--- | :--- |
| `CCME_Values` | **Target** | Ground-truth continuous target for regression. Excluded from input feature matrix $X$. |
| `CCME_WQI` | **Exclude** | Direct label leakage. Deterministic post-hoc categorization of the target index. |
| `Ammonia (mg/l)` | **Documented Leakage Contributor** | Mathematical input into the CCME formula. Retained only under the documented surrogate/emulator framing. |
| `Biochemical Oxygen Demand (mg/l)` | **Documented Leakage Contributor** | Mathematical input into the CCME formula. Retained only under the documented surrogate/emulator framing. |
| `Dissolved Oxygen (mg/l)` | **Documented Leakage Contributor** | Mathematical input into the CCME formula. Retained only under the documented surrogate/emulator framing. |
| `Orthophosphate (mg/l)` | **Documented Leakage Contributor** | Mathematical input into the CCME formula. Retained only under the documented surrogate/emulator framing. |
| `pH (ph units)` | **Documented Leakage Contributor** | Mathematical input into the CCME formula. Retained only under the documented surrogate/emulator framing. |
| `Temperature (cel)` | **Documented Leakage Contributor** | Mathematical input into the CCME formula. Retained only under the documented surrogate/emulator framing. |
| `Nitrogen (mg/l)` | **Documented Leakage Contributor** | Mathematical input into the CCME formula. Retained only under the documented surrogate/emulator framing. |
| `Nitrate (mg/l)` | **Documented Leakage Contributor** | Mathematical input into the CCME formula. Retained only under the documented surrogate/emulator framing. |

*Note: Per Parth's Phase 1 rules, no columns are deleted from the raw dataset at this stage. Final feature filtering and pipeline construction will occur during Parth's preprocessing stage after Taj's cleaning stage produces `data/processed/clean.csv`.*

## 8. Final Decision

- **Target Column**: `CCME_Values`
- **Problem Type**: Regression
- **Known Leakage Columns**:
  - `CCME_WQI`: Definitive exclusion (direct label leakage).
  - 8 chemical measurements (`Ammonia`, `BOD`, `DO`, `Orthophosphate`, `pH`, `Temperature`, `Nitrogen`, `Nitrate`): Contributing formula inputs, documented for pipeline handling and ablation.
- **Remaining Candidate Input Features**:
  - Metadata & Spatiotemporal: `Country`, `Area`, `Waterbody Type`, `Date` (and date-derived temporal components).
  - Chemical parameters: usable under the documented emulator framing and for feature engineering (ratios, non-linear interactions).
