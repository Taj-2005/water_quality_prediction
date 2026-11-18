# What I built and why — Shaik Tajuddin (Phase 2)

| | |
| :--- | :--- |
| **GitHub** | [@Taj-2005](https://github.com/Taj-2005) |
| **Role** | Neural-network data preparation, regularization study |
| **Graded item** | DL dataset and regularization |
| **Branch prefix** | `feat/taj-*` |
| **Theory note** | Regularization: L2 and weight decay, dropout as an implicit ensemble, batch normalization, early stopping |

## 1. Files I own

| File | Step |
| :--- | :--- |
| `src/dl/nn_preprocess.py` | 1 — input pipeline |
| `notebooks/11_nn_data_prep.ipynb` | 1 — evidence |
| `src/dl/regularization.py` | 2 — ablation harness |
| `notebooks/13_regularization.ipynb` | 2 — study and plots |
| `reports/nn_preprocess_*.csv`, `reports/regularization_ablation*.csv` | evidence tables |

## 2. Step 1 — neural-network data preparation

Tree ensembles split on order, so they are invariant to any monotone transform of a
feature. Nothing in Phase 1 depended on the shape of the inputs. A neural network
multiplies inputs by weights and sums them, so it is not invariant, and the Phase 1
pipeline leaves validation values at up to **48 standard deviations**.

Four decisions, each from evidence measured on the training partition only.

**log1p is unusable, and that traces back to my own Phase 1 cleaning.** Five of the eight
measurements contain negative values, so the transform is undefined on them. Those
negatives are the anomalies I deliberately *flagged and retained* rather than deleting.
It is also actively harmful on the columns that are not right-skewed: dissolved oxygen
goes from −1.39 to **−3.17**, and pH from 1.58 to −2.91.

**RobustScaler is the worst option, not the best.** It is the conventional answer when
outliers are present, but it divides by the IQR, and on a long right tail the IQR is
small, so the tail explodes rather than being tamed — max |z| on validation is **740**
against 48 for standardisation.

**Binary flags pass through unscaled.** Standardising a 0/1 column rescales it by its
prevalence in training, so "absent" stops being zero — the flags become −0.445 to −0.690
when off. That fights weight decay, which shrinks toward zero, and dropout, which masks
to zero. `n_imputed_upstream` is an ordered count, so it *is* scaled.

**One-hot, not embeddings.** `Area` is the grouping key for the split, not a feature, so
the only high-cardinality candidate is absent by construction; every test site is unseen
and an embedding could never be looked up. What remains is Country (5 levels) and
Waterbody Type (12), far too small for embeddings to pay. `categorical='ordinal'` is
shipped so the claim can be tested rather than trusted.

**The deciding experiment.** Conditioning statistics describe inputs, not results, so the
choice was settled by training an identical MLP under each strategy over three seeds.
Quantile-normal reached validation MAE **0.8015 ± 0.1002** against Phase 1's
**1.2348 ± 0.0602** — a gap 4.3× the seed-to-seed spread, which clears the work plan's
"clearly better" bar. Yeo-Johnson conditions the inputs well (mean |skew| 6.33 → 1.36)
but did not convert that into accuracy, which is exactly why the plan requires an MLP run
rather than statistics alone.

The Phase 1 pipeline is **unchanged** for Random Forest and XGBoost.

## 3. Step 2 — regularization study

`src/dl/regularization.py` sweeps each family the work plan names — dropout, weight
decay, batch normalization, early-stopping patience and learning-rate schedule — one at a
time against an unregularized reference, then combines the winners.

Everything except the regularizer under test is held fixed: same split, same step-1
preprocessing, same architecture, same optimiser, same epoch budget, same seeds, same
shared `evaluate()`. A row only means something if the rest is constant.

**The adoption rule.** A family is adopted only if its best member beats the
unregularized reference *by more than the seed-to-seed spread*. Comparing means alone is
not enough: this sweep produced a case where the lowest-MAE configuration was ahead by a
margin eighteen times smaller than the noise, and a mean-only rule adopted it. The work
plan's rule — a difference smaller than the noise is not claimed as a win — is now
enforced in `best_combination()` and guarded by a test.

### Results (3 seeds, 200,000 training rows)

| family | best member | val MAE | vs reference | seed sd | verdict |
| --- | --- | --- | --- | --- | --- |
| *(none)* | unregularized | 0.7794 | — | 0.0237 | reference |
| weight decay | `wd=1e-4` | 0.7778 | +0.0016 | 0.0281 | within noise |
| early stopping | `patience=10` | 0.7794 | +0.0000 | 0.0237 | never fired |
| lr schedule | `plateau` | 0.7794 | +0.0000 | 0.0237 | never triggered |
| dropout | `p=0.1` | 0.8111 | −0.0317 | 0.0259 | hurts |
| batch norm | `batch_norm` | 1.0940 | −0.3146 | 0.0470 | hurts |

**The recommendation is no regularization**, and the reason is in the first row of the
table rather than in the comparisons: the unregularized network's generalisation gap is
`+0.0008` WQI points — essentially zero. At 200,000 rows against a 128-64 network the
model is capacity-limited, not variance-limited. There is no overfitting to remove, so
every regularizer can only remove signal, and the two that move the metric at all
(dropout, batch norm) move it the wrong way by an order of magnitude more than the noise.

Three caveats I will not let the table hide:

1. **The early-stopping and plateau rows are not results.** `epochs_run` is 15.0 for every
   run in the sweep, so early stopping never fired at any patience and the scheduler never
   triggered. Those rows are numerically identical to the reference because they *are* the
   reference. They say the mechanism was never exercised, not that it is neutral.
2. **`wd=1e-4` is not a win** even though it sorts first. 0.0016 against 0.0281 is noise.
3. **The conclusion is conditional on capacity and training size.** It is not a general
   claim that this dataset needs no regularization — it is a measurement at one point in
   that space, and it should be re-run if either grows.

Results, the loss curves and the per-family verdicts are in
`notebooks/13_regularization.ipynb`, `reports/regularization_ablation.csv` and
`reports/regularization_ablation_raw.csv` (per-seed, so the spread can be re-checked).

## 4. Theory note — regularization

**The problem.** A network with enough capacity can drive training loss toward zero by
memorising the training set, including its noise. Regularization trades a little training
fit for a model that behaves on data it has not seen. Each technique below does that
differently, which is why they are swept separately before being combined.

**L2 and weight decay.** L2 adds $\lambda\lVert w\rVert^2$ to the loss, so the gradient
gains a $2\lambda w$ term that pulls every weight toward zero in proportion to its size.
The effect is to prefer many small weights over a few large ones, which makes the learned
function smoother and less able to swing sharply to fit one record.

Weight decay is the same idea applied directly to the update — $w \leftarrow (1-\eta\lambda)w$
before the gradient step — and the two are *equivalent only for plain SGD*. For Adam they
diverge, because Adam scales the gradient by its running second moment and so also scales
the L2 term, making the effective decay depend on each parameter's gradient history.
AdamW decouples them, applying the decay outside the adaptive scaling. We use AdamW, so
`weight_decay` in this study means true decoupled decay, not L2 in the loss.

**Dropout as an implicit ensemble.** During training each unit is zeroed independently
with probability $p$, so every forward pass trains a different thinned sub-network. A
network with $n$ units contains $2^n$ such sub-networks sharing weights, and training
samples from them. At test time all units are kept with activations scaled by $1-p$,
which approximates averaging the ensemble's predictions in a single pass.

Two consequences follow. No unit can rely on any particular other unit being present, so
co-adaptation is discouraged and features become individually useful. And because it is an
ensemble over sub-networks, dropout reduces variance rather than bias — it helps a model
that is overfitting and hurts one that is underfitting, which is why the rate is swept
rather than assumed.

**Batch normalization.** Each mini-batch is standardised per feature using the batch mean
and variance, then rescaled by learned $\gamma$ and $\beta$. It was introduced to reduce
internal covariate shift — the drift in each layer's input distribution as the layers
below it update — though later work attributes much of its benefit to a smoother loss
surface that tolerates larger learning rates.

Its regularising effect is a side effect rather than the purpose: because the statistics
come from the batch, each record's normalisation depends on which other records happen to
share its batch, injecting noise much as dropout does. That also means batch norm and
dropout can interact badly, since the train/test discrepancy in the normalisation
statistics compounds with dropout's own — a reason to test them separately first.

**Early stopping.** Training is halted when validation loss stops improving for a set
number of epochs, and the best checkpoint is restored. It is regularization by limiting
*how far* optimisation runs rather than by constraining the weights: with small
initialisation, the weights grow as training proceeds, so stopping early keeps them small,
which for a linear model is provably related to an L2 penalty.

It is the cheapest of the four because it costs no hyperparameter search beyond the
patience value, and the only one that also saves compute. Its risk is a noisy validation
curve triggering a stop on a temporary plateau, which is what the patience parameter buys
tolerance against — and why patience is swept rather than fixed.

**Why sweep before combining.** These act on different failure modes. Weight decay and
early stopping constrain magnitude, dropout reduces co-adaptation, batch norm changes the
optimisation surface. Stacking them without measurement risks over-regularising a model
that was not overfitting in the first place — which, on a dataset of two million rows and
a small MLP, is a real possibility that the unregularized reference row is there to detect.

## 5. How it connects to the rest of the pipeline

- **From Parth:** the persisted split, the training loop, early stopping and the shared `evaluate()`.
- **From Himanshu:** `WaterQualityMLP`, which exposes dropout and batch norm as constructor arguments.
- **To Himanshu:** the chosen input pipeline (quantile-normal), and the instruction to train the
  architecture ablation **without** dropout or batch norm at this data scale, with early stopping
  wired in as an inactive guard. Any regularizer added back needs a measured justification.
- **To Jivit:** the same, for the hybrid's MLP component.
- **To Satya:** the figures and ablation table for the Data and Regularization section of the report.

## 6. What I would do differently

The regularization sweep uses a 200,000-row training subsample so the full grid fits a
sensible compute budget. That is a defensible choice for *selecting* a configuration, but
it means the amount of overfitting being measured is larger than the full 1,986,397-row
partition would show, and a regularizer that helps at 200k may be unnecessary at 2M. The
honest version of this study would re-confirm the winning combination at full scale, which
is the first thing I would add with more time.

A second thing I would change is the epoch budget. Running a fixed 15 epochs meant
validation loss was still falling when training stopped, which silently disabled two of
the five families under test — early stopping and the plateau scheduler never activated,
so a third of the grid returned the reference back to me. A study of early stopping has to
train long enough to reach the point where stopping early is a decision. I would set the
budget from a pilot run that overfits on purpose, then sweep patience against it.
