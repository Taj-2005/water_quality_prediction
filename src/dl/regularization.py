"""Regularization study.   Owner: Shaik Tajuddin.

Phase 2 work plan, Taj Step 2: run the same MLP with each regularizer switched on
and off, one at a time, then in the best combination, and report every change as a
row in a multi-seed ablation table.

Five families are swept, matching the work plan:

  * dropout, at several rates
  * weight decay (L2)
  * batch normalization
  * early-stopping patience
  * learning-rate schedule

Everything except the regularizer under test is held fixed: same split, same
preprocessing, same architecture, same optimiser, same epoch budget, same seeds,
same shared ``evaluate()``. Only then does a difference between two rows mean
anything.

Reuses, unchanged:
  * the persisted Phase 1 split (``src.evaluate_api.get_splits``)
  * the quantile-normal input pipeline chosen in step 1
    (``src.dl.nn_preprocess``, notebooks/11_nn_data_prep.ipynb)
  * the configurable MLP (``src.dl.model.WaterQualityMLP``)
  * the shared training loop and early stopping (``src.dl.train.train``)

Run:
    python -m src.dl.regularization --quick      # 1 seed, small budget, smoke test
    python -m src.dl.regularization              # full sweep, writes reports/
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import SEED

#: Seeds agreed by the team and recorded in docs/literature_review_dl.md.
TEAM_SEEDS = (42, 123, 456, 789, 1011)

#: Held fixed for every row in the ablation. Changing any of these invalidates
#: comparisons between rows, so they live here rather than at each call site.
BASE_CONFIG: dict[str, Any] = {
    "epochs": 15,
    "lr": 1e-3,
    "batch_size": 512,
    "hidden_dims": (128, 64),
    "activation": "relu",
    "optimizer": "adamw",
    "loss": "mse",
    # The unregularized reference: every switch off.
    "dropout": 0.0,
    "weight_decay": 0.0,
    "use_batch_norm": False,
    "early_stopping_patience": 999,  # effectively disabled
    "scheduler": None,
}

#: One-at-a-time variations. Each entry is (family, label, overrides).
REGULARIZER_GRID: list[tuple[str, str, dict[str, Any]]] = [
    ("none", "unregularized", {}),
    ("dropout", "dropout=0.1", {"dropout": 0.1}),
    ("dropout", "dropout=0.2", {"dropout": 0.2}),
    ("dropout", "dropout=0.3", {"dropout": 0.3}),
    ("dropout", "dropout=0.5", {"dropout": 0.5}),
    ("weight_decay", "wd=1e-5", {"weight_decay": 1e-5}),
    ("weight_decay", "wd=1e-4", {"weight_decay": 1e-4}),
    ("weight_decay", "wd=1e-3", {"weight_decay": 1e-3}),
    ("batch_norm", "batch_norm", {"use_batch_norm": True}),
    ("early_stopping", "patience=3", {"early_stopping_patience": 3}),
    ("early_stopping", "patience=5", {"early_stopping_patience": 5}),
    ("early_stopping", "patience=10", {"early_stopping_patience": 10}),
    ("lr_schedule", "cosine", {"scheduler": "cosine"}),
    ("lr_schedule", "plateau", {"scheduler": "plateau", "scheduler_patience": 2}),
]

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "reports"


# ---------------------------------------------------------------------------
# Data: the step-1 pipeline, fitted on train only
# ---------------------------------------------------------------------------
def prepare_data(train_rows: int | None = 200_000, seed: int = SEED, strategy: str = "quantile"):
    """Load the persisted split and apply the step-1 input pipeline.

    Returns ``(X_train, X_val, X_test, y_train, y_val, y_test)`` as float32 arrays,
    with the target standardised using training statistics only.
    """
    from src.dl.nn_preprocess import transform_splits
    from src.evaluate_api import get_splits

    X_tr, X_va, X_te, y_tr, y_va, y_te = get_splits(include_metadata=True, scale_numeric=False)
    if train_rows and train_rows < len(X_tr):
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(X_tr), train_rows, replace=False))
        X_tr, y_tr = X_tr.iloc[idx], y_tr.iloc[idx]

    a, b, c, _ = transform_splits(X_tr, X_va, X_te, strategy=strategy, seed=seed)
    return a, b, c, y_tr.to_numpy(), y_va.to_numpy(), y_te.to_numpy()


def _build_loaders(arrays, batch_size: int, seed: int):
    """Wrap the transformed arrays in the shared Dataset and a seeded DataLoader."""
    import torch
    from torch.utils.data import DataLoader

    from src.dl.data import LoadersBundle, WaterQualityDataset

    Xa, Xb, Xc, ya, yb, yc = arrays
    y_mean, y_std = float(ya.mean()), float(ya.std() or 1.0)
    scale = lambda y: ((y - y_mean) / y_std).astype(np.float32)  # noqa: E731

    g = torch.Generator()
    g.manual_seed(seed)
    train = DataLoader(
        WaterQualityDataset(Xa, scale(ya)), batch_size=batch_size, shuffle=True, generator=g
    )
    val = DataLoader(WaterQualityDataset(Xb, scale(yb)), batch_size=batch_size * 4)
    test = DataLoader(WaterQualityDataset(Xc, scale(yc)), batch_size=batch_size * 4)
    return LoadersBundle(train, val, test, y_mean=y_mean, y_std=y_std, scale_target=True)


# ---------------------------------------------------------------------------
# One configuration
# ---------------------------------------------------------------------------
def run_config(
    arrays,
    overrides: dict[str, Any],
    seed: int = SEED,
    base: dict[str, Any] | None = None,
    eval_test: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    """Train one configuration and return its validation (and optionally test) metrics."""
    from src.dl.model import WaterQualityMLP
    from src.dl.seeds import set_seed
    from src.dl.train import train

    cfg = {**(base or BASE_CONFIG), **overrides}
    set_seed(seed)
    loaders = _build_loaders(arrays, cfg["batch_size"], seed)

    model = WaterQualityMLP(
        input_dim=arrays[0].shape[1],
        hidden_dims=cfg["hidden_dims"],
        activation=cfg["activation"],
        dropout=cfg["dropout"],
        use_batch_norm=cfg["use_batch_norm"],
    )
    train_cfg = {
        "epochs": cfg["epochs"],
        "lr": cfg["lr"],
        "weight_decay": cfg["weight_decay"],
        "optimizer": cfg["optimizer"],
        "loss": cfg["loss"],
        "early_stopping_patience": cfg["early_stopping_patience"],
        "scheduler": cfg["scheduler"],
        "scheduler_patience": cfg.get("scheduler_patience", 2),
        "seed": seed,
        "eval_test": eval_test,
        "log_to_csv": False,
        "verbose": verbose,
    }
    history, best_val = train(model, loaders, train_cfg)

    n_params = sum(p.numel() for p in model.parameters())
    gap = (
        history["val_loss"][-1] - history["train_loss"][-1]
        if history["train_loss"] and history["val_loss"]
        else float("nan")
    )
    return {
        "seed": seed,
        "epochs_run": len(history["train_loss"]),
        "n_params": n_params,
        "val_mae": best_val.get("mae", float("nan")),
        "val_rmse": best_val.get("rmse", float("nan")),
        "val_r2": best_val.get("r2", float("nan")),
        "final_train_loss": history["train_loss"][-1] if history["train_loss"] else float("nan"),
        "final_val_loss": history["val_loss"][-1] if history["val_loss"] else float("nan"),
        "generalisation_gap": gap,
        "history": history,
    }


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------
def run_ablation(
    seeds=(42, 123, 456),
    train_rows: int | None = 200_000,
    grid=None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sweep every regularizer one at a time over several seeds.

    Returns ``(raw, aggregated)``: one row per (config, seed), and the mean and
    standard deviation per config.
    """
    grid = grid or REGULARIZER_GRID
    arrays = prepare_data(train_rows=train_rows)
    rows = []
    for family, label, overrides in grid:
        for s in seeds:
            r = run_config(arrays, overrides, seed=s)
            r.pop("history", None)
            rows.append({"family": family, "config": label, **r})
        if verbose:
            last = [x for x in rows if x["config"] == label]
            mae = np.mean([x["val_mae"] for x in last])
            print(f"  {family:15} {label:16} val MAE {mae:7.4f}  ({len(last)} seeds)")

    raw = pd.DataFrame(rows)
    agg = (
        raw.groupby(["family", "config"])[
            ["val_mae", "val_rmse", "val_r2", "generalisation_gap", "epochs_run"]
        ]
        .agg(["mean", "std"])
        .round(4)
    )
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reset_index().sort_values("val_mae_mean")
    return raw, agg


def best_combination(agg: pd.DataFrame) -> dict[str, Any]:
    """Pick the best setting within each family and combine them.

    Chosen on validation MAE. A family is only adopted if its best member beats
    the unregularized reference; a regularizer that does not help is left off
    rather than carried along because it is conventional.
    """
    lookup = {
        "dropout=0.1": {"dropout": 0.1},
        "dropout=0.2": {"dropout": 0.2},
        "dropout=0.3": {"dropout": 0.3},
        "dropout=0.5": {"dropout": 0.5},
        "wd=1e-5": {"weight_decay": 1e-5},
        "wd=1e-4": {"weight_decay": 1e-4},
        "wd=1e-3": {"weight_decay": 1e-3},
        "batch_norm": {"use_batch_norm": True},
        "patience=3": {"early_stopping_patience": 3},
        "patience=5": {"early_stopping_patience": 5},
        "patience=10": {"early_stopping_patience": 10},
        "cosine": {"scheduler": "cosine"},
        "plateau": {"scheduler": "plateau", "scheduler_patience": 2},
    }
    ref = agg.loc[agg.config == "unregularized"].iloc[0]
    baseline = float(ref.val_mae_mean)
    base_sd = float(ref.val_mae_std)

    combo: dict[str, Any] = {}
    adopted, rejected = [], []
    for fam in ("dropout", "weight_decay", "batch_norm", "early_stopping", "lr_schedule"):
        sub = agg[agg.family == fam]
        if sub.empty:
            continue
        best = sub.loc[sub.val_mae_mean.idxmin()]
        improvement = baseline - float(best.val_mae_mean)
        # The work plan's rule: "a difference smaller than the noise is not claimed as
        # a win". Comparing means alone would adopt a regularizer on an improvement
        # many times smaller than the seed-to-seed spread, which is not evidence.
        noise = max(base_sd, float(best.val_mae_std))
        if improvement > noise:
            combo.update(lookup.get(best.config, {}))
            adopted.append((fam, best.config, round(float(best.val_mae_mean), 4)))
        else:
            rejected.append((fam, best.config, round(improvement, 4), round(noise, 4)))
    return {
        "overrides": combo,
        "adopted": adopted,
        "rejected_within_noise": rejected,
        "baseline_mae": round(baseline, 4),
        "baseline_sd": round(base_sd, 4),
    }


def loss_curves(train_rows: int | None = 200_000, seed: int = SEED, regularized=None):
    """Training and validation loss for an unregularized and a regularized run.

    The plot the work plan asks for: it shows overfitting and its fix on the same
    axes, which an ablation table alone cannot.
    """
    arrays = prepare_data(train_rows=train_rows, seed=seed)
    regularized = regularized or {"dropout": 0.2, "weight_decay": 1e-4}
    off = run_config(arrays, {}, seed=seed)
    on = run_config(arrays, regularized, seed=seed)
    return off["history"], on["history"], regularized


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="1 seed, small budget")
    ap.add_argument("--seeds", type=int, default=3, help="seeds from the agreed list")
    ap.add_argument("--train-rows", type=int, default=200_000)
    args = ap.parse_args()

    seeds = TEAM_SEEDS[: 1 if args.quick else args.seeds]
    rows = 20_000 if args.quick else args.train_rows
    grid = REGULARIZER_GRID[:3] if args.quick else REGULARIZER_GRID

    print(f"seeds {list(seeds)}   train rows {rows:,}   configs {len(grid)}\n")
    raw, agg = run_ablation(seeds=seeds, train_rows=rows, grid=grid)
    print("\nAblation (sorted by validation MAE):")
    print(agg.to_string(index=False))

    best = best_combination(agg)
    print(f"\nReference (unregularized): MAE {best['baseline_mae']} +/- {best['baseline_sd']}")
    print("\nAdopted (improvement exceeds the seed-to-seed noise):")
    for fam, cfg, mae in best["adopted"] or []:
        print(f"  {fam:15} {cfg:16} {mae}")
    if not best["adopted"]:
        print("  none")
    print("\nRejected (best member did not clear the noise):")
    for fam, cfg, imp, noise in best["rejected_within_noise"]:
        print(f"  {fam:15} {cfg:16} improvement {imp:+.4f} vs noise {noise:.4f}")
    print(f"\nBest combination: {best['overrides']}")

    if not args.quick:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        raw.to_csv(RESULTS_DIR / "regularization_ablation_raw.csv", index=False)
        agg.to_csv(RESULTS_DIR / "regularization_ablation.csv", index=False)
        print(f"\nwrote {RESULTS_DIR / 'regularization_ablation.csv'}")


if __name__ == "__main__":
    main()
