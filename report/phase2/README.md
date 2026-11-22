# Phase 2 report

The IEEE conference-format report is built from `main.tex` and summarizes Phase 1
context plus saved Phase 2 evidence for neural input conditioning, regularization,
architecture selection, residual hybrid evaluation, a five-seed model comparison,
leakage checks, subgroup error summaries, and compute cost. The companion presentation
is `water_quality_prediction_phase2_presentation.pdf`.

The report uses the checked-in CSVs and notebooks, distinguishing the controlled
20,000-row five-seed comparison from earlier operating points with different training
scales. It records empirical calibration/interval-coverage results, MLP feature
attribution, and temporal error analysis as outstanding because no numerical outputs
for those analyses are saved. Area and country analyses, cost measurements, and the
five technical leakage checks are included from the latest project evidence.

## Evidence references

- Architecture and hybrid experiments: `notebooks/14_architecture.ipynb`,
  `notebooks/15_hybrid.ipynb`, and the associated `reports/mlp_*.csv` and
  `reports/hybrid_*.csv` files.
- Controlled final comparison, paired differences, leakage checks, subgroup errors,
  and compute cost: `notebooks/10_final_comparison.ipynb` and `reports/final_comparison*.csv`,
  `reports/paired_differences.csv`, `reports/leakage_check.csv`, and
  `reports/compute_cost.csv`.
- Calibration and explainability implementation: `notebooks/16_calibration*.ipynb`
  and `src/dl/calibration.py`.

## Build

From this directory:

```bash
tectonic -X compile main.tex --outdir build
cp build/main.pdf water_quality_prediction_phase2_report.pdf
```
