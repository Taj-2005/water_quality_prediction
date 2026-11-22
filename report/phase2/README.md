# Phase 2 report

The IEEE conference-format Phase 2 report is built from `main.tex` and summarizes
saved project evidence for neural input conditioning, regularization, architecture
selection, and the leakage-safe residual hybrid. It links the Phase 1 findings to the
Phase 2 motivation and distinguishes saved results from evaluations that remain
outstanding.

The companion Phase 2 presentation is included as
`water_quality_prediction_phase2_presentation.pdf`.

## Evidence included

- Phase 1 benchmark and partial-measurement context, attributed to the companion IEEE PDF.
- Phase 2 preprocessing and regularization results recorded in the project report sources.
- Five-seed architecture and hybrid ablations, with notebooks and CSVs at
  `notebooks/14_architecture.ipynb`, `notebooks/15_hybrid.ipynb`, and `reports/`.
- Architecture figures at `reports/figures/architecture_*.png`.

The report is complete as an account of the artifacts available in this branch. It
explicitly identifies calibration/interval coverage, MLP explainability, Phase 2
site/time error analysis, and complete MLP compute timings as not yet empirically
reported. It also explains that the hybrid test CSV repeats seed 42 and therefore is
not a multi-seed test comparison. No results are inferred for missing evaluations.

Some saved notebooks and artifacts were recovered from locally available feature
branch refs because the local `origin/main` snapshot did not contain them. A network
fetch of GitHub was unavailable during this work, so the online main branch could not
be checked independently.

## Build

From this directory:

```bash
tectonic -X compile main.tex --outdir build
cp build/main.pdf water_quality_prediction_phase2_report.pdf
```
