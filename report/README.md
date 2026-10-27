# IEEE Phase 1 report

LaTeX source for the Phase 1 report. Owner: **Satya Yadav**; scaffolded by Taj so that
the results, tables and figures were wired in before drafting began.

## Compiling

No TeX engine is required locally — the project is built to compile on Overleaf.

**Overleaf (recommended):** create the project from this GitHub repository, set
*Menu → Compiler → pdfLaTeX*, and *Main document → `report/main.tex`*. Pulling from
GitHub keeps the report in step with `results/` and `reports/figures/`.

**Locally with Tectonic** (self-contained, no sudo, fetches packages on demand):

```bash
brew install tectonic
mkdir -p report/build
tectonic -X compile report/main.tex --outdir report/build
```

**Locally with a full TeX distribution:**

```bash
cd report
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

The compiled PDF is committed at `report/water_quality_prediction_phase1_report.pdf`
so the report can be read without a TeX install. Re-copy it from `build/` after any
change: `cp report/build/main.pdf report/water_quality_prediction_phase1_report.pdf`.

## Structure

```
report/
├── main.tex              # IEEEtran conference class, author block, section includes
├── make_tables.py        # GENERATES tables/ from results/*.csv
├── sections/             # one file per section, per the work plan
│   ├── 00_abstract.tex           07_conclusion.tex
│   ├── 01_introduction.tex       …
└── tables/               # generated — do not edit by hand
```

Figures are pulled directly from `../reports/figures/` via `\graphicspath`, and the
bibliography from `../docs/refs.bib`. Neither is duplicated, so updating a figure or a
citation updates the report.

## Regenerating the tables

**Never type a number into a table by hand.** After any experiment re-runs:

```bash
python report/make_tables.py
```

This rebuilds all six tables from `results/*.csv`. Hand-edited numbers silently go stale
the moment an experiment is re-run, and a mismatch between the report and the results
files is the kind of thing a viva finds.

## Status

**All seven sections are written** — abstract, introduction, related work, dataset and
EDA, methodology, experiments and results, limitations, conclusion. No `TODO` markers
remain in the source. Approximately 4{,}300 words across the sections, plus six tables
and four figures.

Before submission:

1. **Compiles clean.** 8 pages, no overfull or underfull boxes except one minor
   underfull line in the bibliography. All 16 `\cite` keys resolve, no dangling `\ref`,
   all figure paths present.
2. **Check 8 pages** against whatever limit applies. If it must shrink, cut Related
   Work first — it is the most compressible section.
3. **Re-run `python report/make_tables.py`** if any experiment changes.

## Known inconsistency elsewhere in the repo

`docs/literature_review.md` §6.2 proposes a "Regime 2" using metadata-only features,
which was never run. The report does **not** claim that regime: its Related Work
describes only what was executed, and Limitations states explicitly that a fully
independent regime would bound the residual leakage and that we did not run it. The
report is internally consistent; the standalone literature review document is the thing
that still disagrees with the experiments, and should be reconciled before it is
submitted alongside.
