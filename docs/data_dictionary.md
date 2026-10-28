# Data dictionary

Owner: Shaik Tajuddin. Every column in the raw dataset, documented before cleaning.

All figures below were measured on the full raw file (not a sample) with
`notebooks/01_cleaning.ipynb`. Reproduce them with `python -m src.clean --profile`.

## Source

| Field | Value |
| :--- | :--- |
| Dataset name | A Comprehensive Surface Water Quality Monitoring Dataset (1940–2023): 2.82 Million Record Resource for Empirical and ML-Based Research |
| Authors | Karim, Syeed, Rahman, Rabbani, Fatema, Khan, Hossain & Uddin (Figshare, February 2025) |
| Provider / link | Figshare — DOI [10.6084/m9.figshare.27800394.v2](https://doi.org/10.6084/m9.figshare.27800394) (verified 28 Oct 2026) |
| License and terms of use | **CC BY 4.0** — reuse permitted with attribution. The IEEE report must cite the DOI above |
| Upstream processing | Harmonization and CCME computation were performed by the dataset authors in `Data_Processing_Harmonization.ipynb`; benchmark models in `Technical_Validation.ipynb`. We did not run either — see [Upstream imputation](#upstream-imputation) for what that cost us |
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

**What is observed:** five of eight measurement columns place 11–32 % of their mass on a
single value that is also the column median.

**What is inferred:** that missing values were median-imputed before the dataset reached
us. This is the most parsimonious explanation of the pattern, but it is an inference — we
cannot identify which individual records were originally absent, and the distributed data
cannot confirm the upstream mechanism.

Two consequences the report must state:

1. The project rule *"imputation is fitted on the training split only"* is already
   violated by the data as delivered — the upstream fill used medians computed across all
   rows, including those that will become our test set.
2. Any model fed these columns is partly fitting a constant, not a measurement.

`src/clean.py` therefore adds, per measurement column whose modal value exceeds 5 % of
rows, a `<column>__imputed_upstream` boolean, plus `n_imputed_upstream` (how many of a
row's 8 measurements are suspect, 0–8) and `flag_any_imputed_upstream`. All 8 columns
cross the threshold; 5 carry the median signature.

#### How much of the table is affected

| Suspect measurements in a row | Rows | Share |
| :--- | :--- | :--- |
| 0 — fully measured | 1,102,041 | **39.11 %** |
| 1 | 602,467 | 21.38 % |
| 2 | 408,139 | 14.48 % |
| 3 | 190,526 | 6.76 % |
| 4 or more | 514,833 | **18.27 %** |

Only 39 % of rows are free of fill values, and 18 % have at least half their chemistry
filled in.

#### The imputation is almost entirely England

| Country | Mean suspect measurements per row | Rows |
| :--- | :--- | :--- |
| England | **2.037** | 2,127,897 |
| Ireland | 0.040 | 234,993 |
| China | 0.016 | 45,977 |
| USA | 0.014 | 405,245 |
| Canada | 0.008 | 3,894 |

This is the finding with the most consequences for modelling, and it is **not** a
cleaning problem:

1. `n_imputed_upstream` is a near-perfect proxy for `Country == England`, which is
   75.3 % of the dataset. Any model given these flags can learn country membership from
   them.
2. The target moves with the flag: mean `CCME_Values` is **91.31** for fully measured
   rows versus **75.89** for rows with 3 suspect measurements. Fill status and water
   quality are confounded, so a model can score well by detecting *which country filled
   its gaps* rather than by learning chemistry.
3. Parth's site-separated split cannot fix this, because the confound follows the sites.

Raised for the team in `02_eda.ipynb`; the report's Limitations section must state it.

### Detection limits

**None present.** Checked explicitly: 5,508 rows contain a `<` character, and every one
is in the `Area` site name (for example `BURNHAM MARKET STW F/E <=2028M3/D`), never in a
measurement column. All eight measurement columns parse as pure floats. The planned
detection-limit rule is therefore a documented negative result, not a cleaning step — the
upstream harmonization already resolved censored values, by means we cannot see.

### Anomaly screening

Observations outside the thresholds below are **flagged and retained**, never deleted.
The flags mark records as *potentially anomalous and warranting review*; apart from the
duplicate rule, none of them establishes that a reading is erroneous. Rows are kept so the
decision to exclude them stays with the modelling pipeline, where it is visible.

| Flag column | Rows | Screening rule | Status of the threshold |
| :--- | ---: | :--- | :--- |
| `flag_ph_out_of_range` | 505 | pH outside 0–14 (504 above 14, max 30.0; 1 below 0, min −1.0) | **Convention, not a physical bound** — see below |
| `flag_negative_measurement` | 15 | negative value in any non-temperature measurement: orthophosphate 10, ammonia 2, BOD 2, pH 1 | Physically non-negative quantities; cause not established |
| `flag_temp_implausible` | 7,284 | temperature above 40 °C (observed max 98 °C) | **Project-specific screening heuristic** — no standard supports this exact value |
| `flag_temp_below_zero` | 32 | temperature below 0 °C (min −5.26 °C) | Review only — **deliberately excluded** from `flag_any_impossible` |
| `flag_any_impossible` | 7,676 | union of the first three flags | Aggregate indicator |

The cleaned table is **2,818,006 rows × 30 columns** (14 original + `Date_parsed` + 5
anomaly flags + 10 upstream-imputation columns), written to
`data/processed/clean.parquet` (45 MB, committed) and optionally
`data/processed/clean.csv` (559 MiB, git-ignored).

#### pH: a convention rather than a limit

0–14 is the conventional scale for natural water
([USGS, *pH and Water*](https://www.usgs.gov/special-topics/water-science-school/science/ph-and-water)),
but pH is defined as −log₁₀(a_H⁺) and is not bounded by it. Environmental waters as acidic
as pH −3.6 have been measured in acid mine drainage
([Nordstrom et al. 2000](https://doi.org/10.1021/es990646v)). These 505 records are
therefore implausible for routine ambient monitoring, not impossible.

#### Temperature: why 40 °C, and what it is not

**No published standard supports a 40 °C cutoff, and it is not a physical limit for
surface water.** The USGS documents thermal springs discharging at 50 °C and boiling at
93 °C in a California creek
([Farrar 2007, USGS Fact Sheet 2007-3045](https://pubs.usgs.gov/fs/2007/3045/fs2007-3045.pdf)).
The threshold is justified only by the character of this corpus — routine ambient
monitoring, not geothermal survey.

Two pieces of internal evidence suggest these records are a *recording artefact* rather
than extreme measurements. Reproduce both with `python -m src.sensitivity_temperature`.

| Country | Rows > 40 °C | % of that country's rows | Median as reported | Median if read as °F | % in 0–35 °C band if °F |
| :--- | ---: | ---: | ---: | ---: | ---: |
| USA | 6,970 | 1.6843 % | 63.00 °C | 17.22 °C | **99.99 %** |
| Ireland | 180 | 0.0766 % | 46.40 °C | 8.00 °C | 97.22 % |
| England | 132 | 0.0062 % | 48.45 °C | 9.14 °C | 96.21 % |
| Canada | 2 | 0.0506 % | 65.33 °C | 18.52 °C | 100.00 % |

95.7 % of flagged records are from the USA, and converting them as Fahrenheit yields a
median of 17.2 °C against 11.1 °C for unflagged US records. A unit-coding error confined
largely to one national programme is the most parsimonious explanation. **It remains a
hypothesis** — the distributed data cannot confirm it — and no value is corrected.

**Sensitivity to the cutoff** (`results/temperature_sensitivity.csv`):

| Cutoff | Rows flagged | % of dataset | vs default |
| ---: | ---: | ---: | ---: |
| 30 °C | 8,918 | 0.3153 % | +1,634 |
| 35 °C | 7,576 | 0.2679 % | +292 |
| **40 °C** | **7,284** | **0.2576 %** | — |
| 45 °C | 6,818 | 0.2411 % | −466 |
| 50 °C | 5,979 | 0.2114 % | −1,305 |

35 °C and 45 °C differ by 758 records — 0.027 % of the corpus. **No conclusion in this
project depends on the choice within that range.**

#### Negative values: flagged, cause not established

A concentration cannot be negative physically. A negative *reported* result is not
necessarily a transcription error, however, since blank-corrected analytical results near
the detection limit can take small negative values. We could not establish the reporting
conventions of the contributing programmes from the distributed data, so these 15 records
are flagged for review rather than attributed to a cause, and none is deleted.

*Known imprecision, documented rather than changed:* the screened column set
(`NON_NEGATIVE` in `src/clean.py`) includes pH, which is a logarithmic activity rather
than a concentration. The single negative pH value is therefore counted by both
`flag_negative_measurement` and `flag_ph_out_of_range`. The flag semantics are consumed by
downstream preprocessing and the ablation study, so they are left unchanged.

#### Sub-zero temperature: separated because we are uncertain

Sub-zero water temperature is physically possible — water at 35 ppt salinity freezes near
−2 °C ([NOAA, *Sea Water*](https://www.noaa.gov/jetstream/ocean/sea-water)). That is a
statement about saline water in general and **not** about these records: 30 of the 32 are
English and 23 are rivers, whose salinity is unknown to us. The separate flag records our
uncertainty; it does not assert the readings are valid.

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
| Anomalous values flagged, rows retained | Deleting them would hide a data-quality problem the report must discuss, and would move a modelling decision into the cleaning step. The flags mark records for review; they do not establish error |
| 40 °C temperature cutoff | **Project-specific heuristic, not a standard.** Insensitive to the exact value (35/40/45 °C → 7,576/7,284/6,818 rows) |
| Suspected upstream imputation flagged, rows retained | The repeated-value pattern is observed; median imputation is the inference drawn from it. Flagging lets the team quantify the effect in ablation rather than inherit it invisibly |
| Exact duplicates removed | 9,971 rows; identical across all 14 columns, so they add no information and would inflate any split |
| No imputation performed | Belongs in `src/preprocess.py`, fitted on the training split only |
