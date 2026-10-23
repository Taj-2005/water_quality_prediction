# Data dictionary

Owner: Shaik Tajuddin. Every column in the raw dataset, documented before cleaning.

All figures below were measured on the full raw file (not a sample) with
`notebooks/01_cleaning.ipynb`. Reproduce them with `python -m src.clean --profile`.

## Source

| Field | Value |
| :--- | :--- |
| Dataset name | Combined harmonized surface-water quality dataset with CCME WQI |
| Provider / link | _TBD — pending from Parth. His analysis cites the upstream notebooks `Data_Processing_Harmonization.ipynb` and `Technical_Validation.ipynb`; we need the publication, DOI or repository URL for the report citation._ |
| License and terms of use | _TBD — pending from Parth (same source)._ |
| File placed at | `data/raw/Combined_dataset.csv` (305 MiB, tracked with Git LFS) |
| Rows × columns | 2,827,977 × 14 |
| Date coverage | 2 January 1940 → 17 December 2023 |
| Date retrieved | 24 October 2026 |

## Columns

Missing is **0.00 % for all 14 columns**. See [Upstream imputation](#upstream-imputation)
before treating that as good news.

### Metadata and spatiotemporal

| # | Column | Unit | Meaning | Type | Unique | Notes |
| :-- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `Country` | — | Country the monitoring record comes from | categorical | 5 | England 75.29 %, USA 14.63 %, Ireland 8.31 %, China 1.63 %, Canada 0.14 % |
| 2 | `Area` | — | Monitoring station / sampling site identifier | categorical | 61,689 | Free text, inconsistent across countries. Candidate grouping key for a site-separated split |
| 3 | `Waterbody Type` | — | Class of water body sampled | categorical | 12 | River 65.51 %, Effluent 21.27 %, Lake 5.43 %, Estuarine 1.75 %, Bay 1.63 %, Sea Water 1.13 %, Canal 1.01 %, Sewage 0.84 %, Marine 0.82 %, Drainage 0.36 %, Transitional 0.15 %, Coastal 0.10 % |
| 4 | `Date` | — | Sampling date | string `DD-MM-YYYY` | 23,876 | All 2,827,977 values parse with `%d-%m-%Y`; no malformed dates |

### Physical and chemical measurements

| # | Column | Unit | Meaning | Min | Max | Mean | SD | Negative values |
| :-- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 5 | `Ammonia (mg/l)` | mg/l | Ammoniacal nitrogen concentration | −0.005 | 200.000 | 1.171 | 5.669 | 2 |
| 6 | `Biochemical Oxygen Demand (mg/l)` | mg/l | Oxygen consumed by microbial breakdown of organic matter | −2.000 | 255.000 | 4.887 | 16.414 | 2 |
| 7 | `Dissolved Oxygen (mg/l)` | mg/l | Free oxygen available in the water | 0.000 | 20.000 | 10.008 | 1.851 | 0 |
| 8 | `Orthophosphate (mg/l)` | mg/l | Reactive phosphate concentration | −0.004 | 100.000 | 0.710 | 2.089 | 10 |
| 9 | `pH (ph units)` | pH units | Acidity / alkalinity | −1.000 | 30.000 | 7.736 | 0.495 | 1 |
| 10 | `Temperature (cel)` | °C | Water temperature at sampling | −5.260 | 98.000 | 11.839 | 5.010 | 32 |
| 11 | `Nitrogen (mg/l)` | mg/l | Total nitrogen concentration | 0.000 | 46.000 | 5.210 | 6.200 | 0 |
| 12 | `Nitrate (mg/l)` | mg/l | Nitrate concentration | 0.000 | 155.000 | 4.767 | 6.074 | 0 |

### Target and derived label

| # | Column | Unit | Meaning | Range | Notes |
| :-- | :--- | :--- | :--- | :--- | :--- |
| 13 | `CCME_Values` | dimensionless index | **Target.** CCME Water Quality Index | 31.304 – 100.000 (mean 85.047, SD 17.647) | Regression target. See [target_definition.md](target_definition.md) |
| 14 | `CCME_WQI` | — | Five-class binning of column 13 | Good 36.55 %, Excellent 36.09 %, Marginal 11.62 %, Fair 11.24 %, Poor 4.49 % | **Direct label leakage — excluded as an input feature** |

## Data quality findings

### Upstream imputation

The file reports zero missing values, but five of the eight measurement columns carry
their own **exact median** as the single most frequent value, at a frequency no natural
measurement distribution produces:

| Column | Modal value | Rows | Share | Equals column median |
| :--- | :--- | :--- | :--- | :--- |
| `Dissolved Oxygen (mg/l)` | 10.2 | 893,535 | **31.60 %** | yes |
| `Nitrate (mg/l)` | 4.5 | 716,254 | **25.33 %** | yes |
| `Biochemical Oxygen Demand (mg/l)` | 2.7 | 686,753 | **24.28 %** | yes |
| `pH (ph units)` | 7.78 | 367,035 | **12.98 %** | yes |
| `Temperature (cel)` | 11.46 | 311,639 | **11.02 %** | yes |
| `Ammonia (mg/l)` | 0.03 | 526,716 | 18.63 % | no (median 0.055) |
| `Nitrogen (mg/l)` | 5.0 | 408,683 | 14.45 % | no (median 4.0) |
| `Orthophosphate (mg/l)` | 0.144 | 444,269 | 15.71 % | no (median 0.107) |

**Conclusion:** missing values were median-imputed before the dataset reached us. Up to
31.6 % of `Dissolved Oxygen` is therefore synthetic rather than measured.

Two consequences the report must state:

1. The project rule *"imputation is fitted on the training split only"* is already
   violated by the data as delivered — the upstream fill used medians computed across all
   rows, including those that will become our test set.
2. Any model fed these columns is partly fitting a constant, not a measurement.

`src/clean.py` therefore adds a `*_imputed_upstream` flag per affected column so Parth's
pipeline and Jivit's ablation can measure the effect instead of inheriting it silently.

### Detection limits

**None present.** Checked explicitly: 5,508 rows contain a `<` character, and every one
is in the `Area` site name (for example `BURNHAM MARKET STW F/E <=2028M3/D`), never in a
measurement column. All eight measurement columns parse as pure floats. The planned
detection-limit rule is therefore a documented negative result, not a cleaning step — the
upstream harmonization already resolved censored values, by means we cannot see.

### Impossible values

Flagged, never deleted, per the work plan:

| Flag column | Rows affected | Rule |
| :--- | :--- | :--- |
| `flag_ph_out_of_range` | 505 | pH outside 0–14; observed min −1.0, max 30.0 |
| `flag_negative_measurement` | 15 | negative value in any non-temperature measurement (ammonia 2, BOD 2, orthophosphate 10, pH 1) |
| `flag_temp_implausible` | 7,284 | temperature above 40 °C; observed max 98 °C, impossible for surface water |
| `flag_temp_below_zero` | 32 | temperature below 0 °C — retained and *not* counted as impossible, since saline water can be sub-zero |
| `flag_any_impossible` | 7,676 | union of the three impossible-value flags above |

After duplicate removal the cleaned table is **2,818,006 rows × 20 columns** (14 original
+ `Date_parsed` + 5 flags), written to `data/processed/clean.csv` (434 MB, git-ignored).

### Duplicates

**9,971 exact duplicate rows (0.35 %)**, removed in `src/clean.py`. Duplicates are
compared across all 14 columns.

### Composition caveats

- **England dominates** at 75.29 % of rows; Canada contributes 3,949 rows (0.14 %). Any
  per-country comparison is unbalanced by two orders of magnitude.
- **22.11 % of rows are `Effluent` or `Sewage`** — treatment-works outflow rather than
  natural surface water. Whether these belong in a "water quality prediction" dataset is a
  modelling decision, not a cleaning one; raised for the team in `02_eda.ipynb`.
- **Date coverage is uneven**: 82.2 % of rows fall in 2000–2019, and England's records
  only begin in 2000. A time-separated split would therefore split mostly on England.

## Cleaning decisions

Each decision with its reason. Mirrors the log in `notebooks/01_cleaning.ipynb`.

| Decision | Reason |
| :--- | :--- |
| Column names kept verbatim, not snake_cased | `src/config.py` binds `TARGET_COLUMN = "CCME_Values"`; renaming would break Parth's interface and every downstream module |
| `Date` parsed to `datetime64`, original string retained | Parth's split may key on time; parsing once here avoids five people reparsing inconsistently |
| Detection-limit rule: none applied | No censored values exist in the measurement columns (verified above) |
| Unit conversions: none applied | All measurement columns already share one unit per column (mg/l, pH units, °C); no mixed-unit rows found |
| Impossible values flagged, rows retained | Work plan rule — deleting them would hide a data-quality problem the report must discuss |
| Upstream-imputed values flagged, rows retained | Lets the team quantify the effect in ablation rather than inherit it invisibly |
| Exact duplicates removed | 9,971 rows; identical across all 14 columns, so they add no information and would inflate any split |
| No imputation performed | Belongs in `src/preprocess.py`, fitted on the training split only |
