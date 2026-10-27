# Phase 1 presentation

Owner: **Satya Yadav**. Each member presents their own slides, so Satya keeps the style
consistent and sets the order.

| File | Contents |
| :--- | :--- |
| `phase1_presentation.pdf` | The current deck, 7 slides. Previously `water_quality_prediction_presentation_v2.pdf` in the repository root |

The source lives outside the repository; only the exported PDF is committed. Re-export
to this same filename so the path stays stable — version numbers in filenames mean every
revision breaks the links that point at it.

## Keeping the deck and the report in step

The deck and [`../report/`](../report/) must agree. Where they have disagreed, it has
been because one was updated after a review and the other was not. Before presenting,
check these against the report and `../results/`:

| Claim | Correct as of 28 Oct 2026 |
| :--- | :--- |
| Target at the ceiling | 35.92 % of records, target **bounded** at 100 — *not* censored |
| China | one site (Hou Bay), wholly in training, **zero test records** |
| Training budgets | Random Forest 100,000 records; XGBoost 1,986,397 — not a like-for-like comparison |
| 8/8 regime | R² 0.9997 — formula emulation, **not** a prediction result |
| 7/8 → 6/8 → 4/8 | R² 0.834 → 0.814 → 0.573 |
| Test set | 415,804 records across four countries |

**Outstanding:** slide 4 still describes the target as *"censored at its ceiling"*. It
should read *bounded* — by the CCME formula a fully compliant record gives
$F_1=F_2=F_3=0$ and therefore exactly 100, so no latent value exists above the bound.
The distinction decides the remedy: censoring would call for a censored-regression
estimator, whereas a bounded target with a point mass calls for stratified reporting,
which is what `src.metrics.ceiling_split_metrics` provides.

## Related

- [`../report/`](../report/) — IEEE report source and compiled PDF
- [`../docs/viva_prep.md`](../docs/viva_prep.md) — 77 questions across nine areas
- [`../docs/mock_viva_plan.md`](../docs/mock_viva_plan.md) — two-round rehearsal agenda
