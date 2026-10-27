# Mock viva plan

The work plan requires **at least two** mock vivas, and the Phase 1 viva is individually
assessed. The failure mode this exists to catch is five people who can each explain their
own file and none who can explain the project.

Question bank: [`viva_prep.md`](viva_prep.md) — 77 questions across nine areas.

---

## Round 1 — cross-examination (90 minutes)

**Rule: nobody presents their own work.** Each member presents a teammate's area for five
minutes, then takes ten minutes of questions from the other four. The owner stays silent
until the end, then corrects what was wrong.

| Presenter | Presents | Owner stays silent |
| :--- | :--- | :--- |
| Taj | XGBoost, tuning and ablation | Jivit |
| Parth | Cleaning, EDA and the imputation finding | Taj |
| Himanshu | Metrics, baselines and the ceiling problem | Satya |
| Jivit | Split design, preprocessing and leakage | Parth |
| Satya | Feature engineering and Random Forest | Himanshu |

Each presenter must cover: what the component does, one design decision and its
alternative, and one number they can defend.

**Scoring.** After each slot the owner says one of: *accurate*, *accurate but thin*, or
*wrong on X*. Record it. Anything marked wrong gets re-presented in round 2.

---

## Round 2 — the hard list (60 minutes)

Cold, no notes, round-robin. Everyone answers:

1. The five questions in `viva_prep.md` §0 — the ones most likely to decide the viva.
2. The five in *"Questions we currently answer badly"*. The correct response to these is a
   clean concession plus the reasoning, **not** a bluff. Practise saying "that was a
   budget decision, not a scientific one" out loud.

Then two curveballs per person, drawn from a section they did **not** present in round 1.

---

## The answers everyone must have cold

No member should enter the viva unable to give these without hesitation:

| Fact | Value |
| :--- | :--- |
| Records, after cleaning | 2,818,006 × 30 |
| Split | Area-grouped 70/15/15, stratified by country, seed 42 |
| Train / val / test | 1,986,397 / 415,805 / 415,804 |
| Sites | 61,689 — none spans two partitions |
| Target | `CCME_Values`, continuous, bounded at 100 |
| Share at the ceiling | 35.92 % |
| 8/8 result | R² 0.9997 — **formula emulation, not prediction** |
| 7/8 → 4/8 | R² 0.834 → 0.814 → 0.573 |
| Why the drop | orthophosphate removed; formula no longer reconstructible |
| Upstream imputation | 5 of 8 columns at their exact median, 11–32 % of rows |
| Where it concentrates | England, 2.04 per record vs ≤0.04 elsewhere |
| Training budgets | RF 100,000 vs XGBoost 1,986,397 — **not a clean comparison** |
| China | one site, all in train, **zero test records** |

---

## Scheduling

Hold round 1 at least a week before the evaluation window so round 2 can target what
round 1 exposed. Both rounds before the deck is finalised — rehearsing the questions
usually changes which slides you want.

| Round | Target date | Owner |
| :--- | :--- | :--- |
| Round 1 — cross-examination | _TBD — schedule this week_ | Taj (lead) |
| Round 2 — hard list | _TBD — at least 3 days after round 1_ | Taj (lead) |

Record after each round: who was marked *wrong on X*, and which areas need a third pass.
