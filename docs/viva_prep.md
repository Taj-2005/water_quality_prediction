# Viva preparation

The viva is **individually assessed**, so every member must be able to explain any part of
the pipeline — including a teammate's notebook. This document is the shared question bank.

Questions are grouped by area, with the owner who should lead the answer. **Everyone is
expected to answer anything**; the owner column says who to ask when you are stuck, not
who is responsible for knowing it.

Answer sketches are deliberately short. They are prompts to check you can reconstruct the
reasoning, not scripts to memorise — an examiner will follow up, and a memorised answer
collapses on the second question.

---

## 0. The five questions most likely to decide the viva

If you prepare nothing else, prepare these.

**1. Your R² is 0.9997. Why is that bad news?**
Because the target is computed from the inputs by a fixed formula. With all 8
measurements the model reconstructs the CCME equation, so the score measures function
approximation, not prediction. We report it as an emulation benchmark and a pipeline
check — if it were materially lower, something upstream would be broken.

**2. Then what is your actual contribution?**
Estimating the index when measurements are missing, where the formula cannot be
evaluated at all. Withholding orthophosphate alone drops XGBoost to R² = 0.834 and
multiplies MAE by 48 (0.100 → 4.776). Withholding four leaves R² = 0.573. That gap is
the quantitative statement of how much apparent accuracy was formula reconstruction.

**3. How did you know the data had been imputed if it reports no missing values?**
By checking modal values. Five of eight measurement columns carry their own *exact
median* as the single most frequent value — dissolved oxygen at 31.60 % of rows, nitrate
25.33 %, BOD 24.28 %, pH 12.98 %, temperature 11.02 %. No natural measurement
distribution concentrates that much mass on its median. It is a median fill applied
before publication.

**4. Why is that imputation a modelling problem rather than a data-quality one?**
Because it is not uniformly distributed. England averages 2.04 filled measurements per
record against ≤0.04 for every other country, while supplying 75.3 % of the data. So
fill status is a proxy for country, and mean CCME is 91.31 for fully measured records
against 75.89 at three filled. Fill status, country and target are mutually confounded,
and no split design removes it.

**5. Why should we believe your test scores are not inflated by leakage across the split?**
Records are grouped by monitoring station and whole stations are allocated to one
partition only — 61,689 sites, 70/15/15, stratified by country. The same site sampled on
consecutive dates has near-identical chemistry, so a record-level random split would put
near-duplicates on both sides. Every test record comes from a site the model has never
seen.

---

## 1. Dataset, cleaning and EDA — *Taj*

| Question | Answer sketch |
| :--- | :--- |
| What is the dataset and where is it from? | Harmonised surface-water monitoring, 2,827,977 × 14 before cleaning, five countries, 1940-01-02 to 2023-12-17. Figshare DOI 10.6084/m9.figshare.27800394, CC BY 4.0. |
| How many records after cleaning, and what changed? | 2,818,006 × 30. Removed 9,971 exact duplicates (0.35 %); added a parsed date, 5 impossible-value flags and 10 imputation-related columns. Nothing else was deleted. |
| Why flag impossible values instead of dropping them? | Deleting hides a data-quality problem the report must discuss, and it turns cleaning into the place where modelling decisions get made silently. Flags move that choice to the pipeline, where it is visible. 505 pH outside 0–14, 15 negative concentrations, 7,284 temperatures above 40 °C. |
| Why is −5.26 °C flagged separately from 98 °C? | Saline water can be sub-zero, so negative temperature is physically possible and is flagged for review only. 98 °C is not possible for surface water. Likely a Fahrenheit/Celsius mix upstream. |
| Where did the 40 °C threshold come from? | A judgement call, not a standard. It flags 7,284 records. Be honest about this — say it is a defensible choice that the team agreed, and that 35 or 45 would also be defensible. |
| What was the detection-limit rule? | There was none to apply. 5,508 rows contain `<`, every one inside an `Area` site name such as `BURNHAM MARKET STW F/E <=2028M3/D`. All 8 measurement columns parse as pure floats. The check stays in the code so a future refresh reports censored values rather than coercing them to NaN. |
| Why no imputation during cleaning? | It must be fitted on the training split only, and the split does not exist yet. Imputing here would fit statistics across rows destined for test. |
| Why parquet rather than CSV? | 45 MB against 559 MiB. GitHub rejects files over 100 MB, so the CSV could not be committed at all; parquet also loads in 0.3 s against 8.5 s and preserves category and bool dtypes that a CSV round trip destroys. Everyone then models identical bytes. |
| What does the target distribution look like? | Bounded at 100 with 35.92 % of records exactly at the bound. 75th, 90th and 99th percentiles are all 100 — no upper tail, just a spike. Ceiling share ranges 20.9 % (England) to 84.5 % (USA). |
| Which features are skewed, and does it matter? | Ammonia skew 11.98 (kurtosis 222), BOD 10.21, orthophosphate 8.94. Trees are invariant to monotone transforms so it barely matters for Phase 1; it matters for the Phase 2 MLP. |
| What is in the dataset that is not natural surface water? | 22.1 % is Effluent or Sewage — treatment-works outflow. Mean index 66.6 against 90.0 for rivers. A team decision to keep, stated in the report. |

---

## 2. Target definition and leakage — *Parth*

| Question | Answer sketch |
| :--- | :--- |
| What exactly is the target? | `CCME_Values`, a continuous index in [0, 100]. Regression, not classification. |
| Write the CCME formula. | $y = 100 - \sqrt{F_1^2+F_2^2+F_3^2}/1.732$, where $F_1$ is scope (share of variables failing guidelines), $F_2$ frequency (share of failing tests), $F_3$ amplitude (how far failures exceed guidelines). $\sqrt{3}\approx1.732$ normalises the maximum distance to 100. |
| Why is $F_2 \equiv F_1$ here? | Each record is a single measurement rather than an aggregated time window, so frequency and scope coincide. This is an upstream adaptation, not the published protocol — be ready to say so. |
| How did you prove the leakage? | Recomputed the index from the 8 measurements and compared to the published column: zero reconstruction error. |
| Why not just drop the 8 measurements? | Then almost nothing predictive remains — only country, water body type and date. We kept them and changed the question instead, reporting 8/8 as emulation and 7/8, 6/8, 4/8 as the real task. |
| Why is `CCME_WQI` excluded? | It is a 5-class binning of the target itself. Using it as a feature is direct label leakage. |
| Is the target censored? | **No** — bounded with a point mass. Censoring implies a latent value beyond a threshold that recording conceals. Here a fully compliant record gives $F_1=F_2=F_3=0$ and therefore exactly 100; there is no hidden value above it. So Tobit-style censored regression is the wrong remedy; stratified reporting is the right one. |
| Why does dissolved oxygen barely correlate with the target? | Pearson −0.031, Spearman −0.021. DO is a more-is-better parameter, but the upstream implementation appears to treat its 14 mg/l guideline as an upper bound, inverting its contribution. We report the index as published and flag this as a probable upstream defect. |
| Why does orthophosphate dominate? | Spearman −0.917, top SHAP at 8.813. Its guideline threshold (0.05 mg/l) is the tightest of the eight, so its excursion term dominates the amplitude factor. |

---

## 3. Split and preprocessing — *Parth*

| Question | Answer sketch |
| :--- | :--- |
| Describe the split. | Grouped by `Area` (61,689 sites), whole sites allocated 70/15/15, stratified by country, fixed seed 42, indices persisted to disk. Train 1,986,397 / val 415,805 / test 415,804. |
| Why group by site rather than split rows at random? | Repeated measures. The same station on consecutive dates is near-identical, so random splitting places near-duplicates on both sides and reports site memorisation as generalisation. |
| How are sites allocated to hit 70/15/15? | Shuffle under a fixed seed, sort descending by size, then greedy bin-packing — each site goes to whichever partition has the largest remaining deficit. Integer site sizes mean the proportions are approximate. |
| Why is England 76.8 % of test but 75.3 % of the corpus? | Whole sites are allocated, so proportions match only as closely as site sizes permit. |
| What happened to China? | One site (Hou Bay). It cannot be divided without breaking the grouping guarantee, so it went entirely to training. **China has zero test records** and is absent from the country table. Nothing in the report evidences performance on Chinese waters. |
| What does your split establish, and what does it not? | Establishes generalisation to unseen monitoring stations. Does **not** establish anything temporal — the split is not chronological, so nothing here speaks to future years. |
| How is train-only fitting enforced? | All fitted statistics come from the training partition and are applied unchanged to val and test. Programmatic checks assert the target is not in the feature matrix and no group spans partitions. |
| When you mask a measurement, what else has to go? | Every derived feature computed from it — otherwise a masked measurement re-enters through a log transform or ratio. |
| Why exclude site ID as a feature? | 61,689 levels, and under a grouped split every test site is unseen by construction, so it can only add noise. |

---

## 4. Feature engineering — *Himanshu*

| Question | Answer sketch |
| :--- | :--- |
| What did you engineer and why? | Log transforms of the skewed measurements (ammonia, BOD, nitrate, nitrogen, orthophosphate) and ratios between chemical parameters. |
| Do log transforms help tree models? | In principle no — trees split on order, so they are invariant to monotone transforms. The ablation nevertheless shows a gain, which comes from the *ratios* and from interaction terms, not the logs alone. Be ready to make this distinction. |
| Quantify the gain. | XGBoost test MAE 0.1517 → 0.1000 with engineered features, a 34 % relative improvement; RMSE 0.4005 → 0.3282. |
| How did you avoid leakage in the features? | No feature is a function of the target, and any aggregate is computed from training rows only. |
| Why are impurity importances not reported? | They are biased under correlated and high-cardinality predictors. We report SHAP instead. |

---

## 5. Models — *Himanshu (RF), Jivit (XGBoost)*

| Question | Answer sketch |
| :--- | :--- |
| How does Random Forest reduce variance? | Bagging — bootstrap samples plus random feature subsets at each split decorrelate the trees; averaging decorrelated predictors reduces variance without raising bias much. |
| What is out-of-bag error? | Each tree omits ~37 % of records; predicting those with only the trees that did not see them gives a validation estimate without a holdout. |
| How does gradient boosting differ? | Sequential, each tree fitted to the gradient of the loss of the current ensemble. Bias reduction rather than variance reduction. |
| What does XGBoost add over plain boosting? | Regularised objective ($L_1$/$L_2$ on leaf weights), shrinkage, column and row subsampling, sparsity-aware split finding, second-order gradients. |
| What did you tune, and on what? | RF: trees, depth, min leaf, features per split. XGBoost: learning rate, depth, subsample, colsample, min child weight, $L_1$/$L_2$, with early stopping. **Validation only** — the test set was used once. |
| **Why is Random Forest trained on only 100,000 records?** | Computational budget. RF grows unpruned trees whose cost scales with records retained at every node; the full 1,986,397 was not feasible for us, while boosting's depth-limited trees are far cheaper per record. **Do not pretend this is a clean comparison** — the XGBoost margin confounds algorithm with training-set size and bounds rather than measures the algorithmic gap. |
| So is XGBoost genuinely better? | Probably, but we cannot prove the magnitude. The margin holds at 2–4 % of R² across a twentyfold training-size difference, which suggests the ordering is not purely a budget effect. Equalising budgets is the first experiment we would add. |

---

## 6. Metrics and evaluation — *Satya*

| Question | Answer sketch |
| :--- | :--- |
| Why report MAE *and* RMSE? | They respond differently to the heavy tails. RMSE is dominated by a small number of large excursions; MAE describes the typical record. Reporting one alone hides the shape of the error. |
| What is the baseline, and what does it score? | Median predictor: test MAE 13.6047, RMSE 19.4916, R² −0.0923. A mean predictor gives MAE 14.8996, RMSE 18.6558, R² −0.0006. |
| Why is the mean baseline's R² ≈ 0? | Because R² is *defined* against the mean predictor — that is its reference. The median scores worse on R² only because it minimises absolute, not squared, error. Each baseline wins on the metric it optimises. |
| Why bootstrap confidence intervals? | So the comparison between models is a statement about sampling variability rather than a single point estimate. RF and XGBoost intervals do not overlap on any metric. |
| What is wrong with the metrics as reported? | They are pooled across the ceiling. With 35.92 % of records at exactly 100, a pooled number conflates two tasks: identifying fully compliant records, and estimating how far the rest fall short. Stratified reporting would be more honest — this is stated as a limitation. |
| What does SHAP actually compute? | A Shapley value per feature per prediction: the average marginal contribution of that feature across all orderings of the feature set. Unlike impurity importance it is local, signed, and additive to the prediction. |

---

## 7. Results interpretation — everyone

| Question | Answer sketch |
| :--- | :--- |
| Walk me down the degradation curve. | 8/8: R² 0.9997, MAE 0.100 (emulation). 7/8: 0.834 / 4.776. 6/8: 0.814 / 5.209. 4/8: 0.573 / 8.596 — roughly one CCME quality class of error. |
| Why does removing *one* feature cost so much? | Orthophosphate is the dominant term in the amplitude factor. Once it is gone the formula can no longer be reconstructed, and the model falls back on statistical structure among the remaining parameters. |
| Why is 6/8 barely worse than 7/8? | Most of the loss happens at the first removal, because that is what breaks reconstruction. BOD carries less unique information. |
| Do the imputation flags help? | Marginally — MAE 0.1517 → 0.1474. Given they proxy for country, that gain does not justify including them, so they are excluded from the primary model and kept as diagnostics. |
| Which country is hardest? | On the full-measurement regime all are high. Canada's figures rest on 584 records and should not be read as a country comparison. China has no test records at all. |

---

## 8. Process and reproducibility — everyone

| Question | Answer sketch |
| :--- | :--- |
| How would I reproduce your results? | `pip install -r requirements.txt`, `python -m src.clean`, `python -m src.split`, then the model modules. Seed 42 fixed in `src/config.py` and imported everywhere; split indices persisted to disk. |
| How do you know everyone trained on the same data? | One split, generated once, written to `data/splits/`, and the cleaned table committed as parquet so nobody depends on their own rerun. |
| What does your CI check? | Lint (ruff), format, 124 unit tests, and that no notebook was committed carrying an error traceback. It checks out with `lfs: false` so runs do not consume the repository's LFS bandwidth quota. |
| What do the tests actually test? | The cleaning rules on small synthetic frames — that impossible values are flagged and never dropped, that sub-zero temperature is review-only, that duplicate matching ignores derived flags, that constant fill is detected. One of them caught a real bug: the detection-limit check tested `dtype == object`, which silently never fires on pandas 3 where strings get the `str` dtype. |

---

## 9. Phase 2 — everyone

| Question | Answer sketch |
| :--- | :--- |
| What changes in Phase 2? | A regularised MLP on the identical splits and preprocessing, so the neural model and the trees are comparable without the confounds that make such comparisons unreliable. |
| Why might the MLP behave differently on these features? | It is not invariant to monotone transforms, so the skewness (up to 11.98) and the log features should matter in a way they do not for trees. |
| What would a hybrid look like, and would it be justified? | An MLP on XGBoost residuals, compared fairly against each model alone. Only worth claiming if it beats both. |

---

## Questions we currently answer badly

Be honest about these rather than bluffing. Examiners reward a known limitation over a confident evasion.

1. **"Why 100k for Random Forest?"** — budget, not science. We cannot separate the algorithmic gap from the data gap.
2. **"Which 7 of 8, and why those?"** — the subsets are interpretable, not exhaustive. There are 8 ways to drop one and 70 ways to drop four; we report one informative choice per level.
3. **"What would a model with no formula contributors score?"** — we did not run it. It would bound the residual leakage at 7/8 and 6/8.
4. **"Is the 40 °C threshold standard?"** — no, it is our judgement.
5. **"Can you trust the index itself?"** — partly. The dissolved-oxygen handling looks inverted and $F_2 \equiv F_1$ departs from the published protocol.

---

## Mock viva protocol

The work plan requires **at least two** mock vivas. Run them as follows:

1. **Round 1 — cross-examination.** Each member is assigned a *teammate's* area and must present it for five minutes, then take questions from the rest. The owner stays silent until the end, then corrects.
2. **Round 2 — the hard list.** Everyone answers the five questions in Section 0 and the five in "Questions we answer badly", cold.

Record who struggled with what and re-run those areas. The failure mode to avoid is five people who can each explain their own file and none who can explain the project.
