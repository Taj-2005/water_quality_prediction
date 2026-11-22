"""Architecture ablation and hyperparameter tuning for WaterQualityMLP.

Owner: Himanshu Pandey.
Part of Phase 2: Water Quality Prediction Deep Learning Extension.

Phase 2 work plan, Himanshu Step 3:
  1. Systematically vary depth, width, and activation (ReLU vs. GELU vs. SiLU)
     one at a time on the validation set only.
  2. Perform hyperparameter optimization (via Optuna / systematic search) strictly
     on the validation split to find the best architectural configuration.
  3. Confirm the tuned architecture on all 5 team seeds (42, 123, 456, 789, 1011)
     evaluating on the test set with the shared evaluate() function.
  4. Save results to reports/mlp_results.csv and generate diagnostic figures.

Reuses unchanged:
  * Persisted Phase 1 split indices (src.evaluate_api.get_splits)
  * Neural-network preprocessing pipeline (src.dl.nn_preprocess)
  * Modular MLP architecture with dynamic activation init (src.dl.model.WaterQualityMLP)
  * Shared training loop and evaluation contract (src.dl.train.train, src.evaluate_api.evaluate)
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import FIGURES, REPORTS, SEED
from src.dl.data import LoadersBundle, WaterQualityDataset
from src.dl.model import WaterQualityMLP
from src.dl.nn_preprocess import transform_splits
from src.dl.seeds import set_seed
from src.dl.train import evaluate_loader, get_device, get_loss_function, train
from src.evaluate_api import get_splits

#: Seeds agreed upon by the team and recorded in docs/literature_review_dl.md
TEAM_SEEDS = (42, 123, 456, 789, 1011)

#: Baseline configuration held fixed when varying depth, width, and activation
BASE_CONFIG: dict[str, Any] = {
    "epochs": 15,
    "lr": 1e-3,
    "batch_size": 512,
    "hidden_dims": (128, 64),
    "activation": "relu",
    "optimizer": "adamw",
    "loss": "mse",
    "dropout": 0.0,
    "weight_decay": 0.0,
    "use_batch_norm": False,
    "early_stopping_patience": 999,
    "scheduler": None,
}

#: Depth ablation grid (holding width funnel & activation fixed)
DEPTH_GRID: list[tuple[str, str, dict[str, Any]]] = [
    ("depth", "depth_1 (128)", {"hidden_dims": (128,)}),
    ("depth", "depth_2 (128-64)", {"hidden_dims": (128, 64)}),
    ("depth", "depth_3 (256-128-64)", {"hidden_dims": (256, 128, 64)}),
    ("depth", "depth_4 (256-128-64-32)", {"hidden_dims": (256, 128, 64, 32)}),
    ("depth", "depth_5 (256-128-128-64-32)", {"hidden_dims": (256, 128, 128, 64, 32)}),
]

#: Width ablation grid (at depth 2 and depth 3)
WIDTH_GRID: list[tuple[str, str, dict[str, Any]]] = [
    ("width", "narrow (64-32)", {"hidden_dims": (64, 32)}),
    ("width", "uniform_64 (64-64)", {"hidden_dims": (64, 64)}),
    ("width", "baseline (128-64)", {"hidden_dims": (128, 64)}),
    ("width", "uniform_128 (128-128)", {"hidden_dims": (128, 128)}),
    ("width", "wide (256-128)", {"hidden_dims": (256, 128)}),
    ("width", "uniform_256 (256-256)", {"hidden_dims": (256, 256)}),
    ("width", "very_wide (512-256)", {"hidden_dims": (512, 256)}),
    ("width", "deep_wide (384-192-96)", {"hidden_dims": (384, 192, 96)}),
]

#: Activation function ablation grid (holding architecture fixed at 128-64)
ACTIVATION_GRID: list[tuple[str, str, dict[str, Any]]] = [
    ("activation", "relu", {"activation": "relu"}),
    ("activation", "gelu", {"activation": "gelu"}),
    ("activation", "silu", {"activation": "silu"}),
    ("activation", "leaky_relu", {"activation": "leaky_relu"}),
    ("activation", "tanh", {"activation": "tanh"}),
]

#: Full systematic architecture ablation grid combining all one-at-a-time variations
ARCHITECTURE_GRID: list[tuple[str, str, dict[str, Any]]] = DEPTH_GRID + WIDTH_GRID + ACTIVATION_GRID


def prepare_data(
    train_rows: int | None = 200_000,
    strategy: str = "quantile",
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load persisted Phase 1 splits and apply neural network preprocessing.

    Parameters
    ----------
    train_rows : int | None, default 200_000
        Number of training rows to subsample for tractable ablation sweeps.
        None uses the full ~1.98M training split.
    strategy : str, default 'quantile'
        Preconditioning strategy validated by Taj in Step 1.
    seed : int, default SEED
        Random seed for deterministic train subsampling.

    Returns
    -------
    tuple of (X_tr, X_va, X_te, y_tr, y_va, y_te) as numpy arrays.
    """
    X_tr, X_va, X_te, y_tr, y_va, y_te = get_splits(
        include_metadata=True,
        scale_numeric=False,
    )
    if train_rows and train_rows < len(X_tr):
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(X_tr), train_rows, replace=False))
        X_tr, y_tr = X_tr.iloc[idx], y_tr.iloc[idx]

    Xa, Xb, Xc, _ = transform_splits(X_tr, X_va, X_te, strategy=strategy, seed=seed)
    return Xa, Xb, Xc, y_tr.to_numpy(), y_va.to_numpy(), y_te.to_numpy()


def _build_loaders(
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    batch_size: int,
    seed: int,
) -> LoadersBundle:
    """Wrap transformed numpy arrays in WaterQualityDataset and seeded DataLoaders."""
    Xa, Xb, Xc, ya, yb, yc = arrays
    y_mean, y_std = float(ya.mean()), float(ya.std() or 1.0)
    scale = lambda y: ((y - y_mean) / y_std).astype(np.float32)  # noqa: E731

    ds_tr = WaterQualityDataset(Xa, scale(ya))
    ds_va = WaterQualityDataset(Xb, scale(yb))
    ds_te = WaterQualityDataset(Xc, scale(yc))

    g = torch.Generator()
    g.manual_seed(seed)

    tr_loader = DataLoader(
        ds_tr,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
        drop_last=False,
    )
    va_loader = DataLoader(
        ds_va,
        batch_size=batch_size * 2,
        shuffle=False,
        drop_last=False,
    )
    te_loader = DataLoader(
        ds_te,
        batch_size=batch_size * 2,
        shuffle=False,
        drop_last=False,
    )

    return LoadersBundle(
        tr_loader,
        va_loader,
        te_loader,
        y_mean=y_mean,
        y_std=y_std,
        scale_target=True,
    )


def run_config(
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    overrides: dict[str, Any] | None = None,
    base: dict[str, Any] | None = None,
    seed: int = SEED,
    eval_test: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    """Train one architecture configuration and return validation (and test) metrics."""
    cfg = {**(base or BASE_CONFIG), **(overrides or {})}
    set_seed(seed)
    loaders = _build_loaders(arrays, cfg["batch_size"], seed)

    model = WaterQualityMLP(
        input_dim=arrays[0].shape[1],
        hidden_dims=cfg["hidden_dims"],
        activation=cfg["activation"],
        dropout=cfg["dropout"],
        use_batch_norm=cfg["use_batch_norm"],
    )

    dims_str = "_".join(map(str, cfg["hidden_dims"]))
    ckpt_name = f"arch_{dims_str}_{cfg['activation']}_{seed}.pt"
    ckpt_file = Path("checkpoints") / ckpt_name
    if ckpt_file.exists():
        ckpt_file.unlink(missing_ok=True)

    train_cfg = {
        "epochs": cfg["epochs"],
        "lr": cfg["lr"],
        "weight_decay": cfg["weight_decay"],
        "optimizer": cfg["optimizer"],
        "loss": cfg["loss"],
        "early_stopping_patience": cfg["early_stopping_patience"],
        "scheduler": cfg["scheduler"],
        "model_name": ckpt_name,
        "seed": seed,
        "eval_test": False,
        "log_to_csv": False,
        "verbose": verbose,
    }

    try:
        history, best_val = train(model, loaders, train_cfg)

        test: dict[str, float] = {}
        if eval_test:
            device = get_device(None)
            _, test = evaluate_loader(
                model.to(device),
                loaders.test,
                get_loss_function(cfg["loss"]).to(device),
                device,
                y_mean=loaders.y_mean,
                y_std=loaders.y_std,
                scale_target=loaders.scale_target,
            )
    finally:
        if ckpt_file.exists():
            ckpt_file.unlink(missing_ok=True)

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
        "depth": len(cfg["hidden_dims"]),
        "hidden_dims": str(cfg["hidden_dims"]),
        "activation": str(cfg["activation"]),
        "val_mae": best_val.get("mae", float("nan")),
        "val_rmse": best_val.get("rmse", float("nan")),
        "val_r2": best_val.get("r2", float("nan")),
        "final_train_loss": history["train_loss"][-1] if history["train_loss"] else float("nan"),
        "final_val_loss": history["val_loss"][-1] if history["val_loss"] else float("nan"),
        "generalisation_gap": gap,
        **(
            {
                "test_mae": test.get("mae", float("nan")),
                "test_rmse": test.get("rmse", float("nan")),
                "test_r2": test.get("r2", float("nan")),
            }
            if eval_test
            else {}
        ),
        "history": history,
    }


def aggregate_ablation(raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate multi-seed ablation metrics into mean and standard deviation."""
    cols = [
        c
        for c in ("val_mae", "val_rmse", "val_r2", "generalisation_gap", "epochs_run")
        if c in raw.columns
    ]
    # Group by family, config label, hidden_dims, activation, n_params
    group_cols = [
        c
        for c in ("family", "config", "depth", "hidden_dims", "activation", "n_params")
        if c in raw.columns
    ]
    agg = raw.groupby(group_cols)[cols].agg(["mean", "std"]).round(4)
    agg.columns = ["_".join(c) for c in agg.columns]
    agg["n_seeds"] = raw.groupby(group_cols)["seed"].count()
    return agg.reset_index().sort_values("val_mae_mean", ascending=True)


def run_architecture_ablation(
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    | None = None,
    seeds: Sequence[int] = TEAM_SEEDS,
    train_rows: int | None = 200_000,
    grid: list[tuple[str, str, dict[str, Any]]] | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute the systematic architecture ablation sweep across depth, width, and activation.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (raw_results_df, aggregated_results_df)
    """
    grid = grid or ARCHITECTURE_GRID
    if arrays is None:
        arrays = prepare_data(train_rows=train_rows)

    rows: list[dict[str, Any]] = []
    for family, label, overrides in grid:
        for s in seeds:
            res = run_config(arrays, overrides, seed=s, verbose=False)
            res.pop("history", None)
            rows.append({"family": family, "config": label, **res})

        if verbose:
            current = [x for x in rows if x["config"] == label]
            mae_mean = float(np.mean([x["val_mae"] for x in current]))
            mae_std = float(np.std([x["val_mae"] for x in current]))
            print(
                f"[{family:10}] {label:28} Val MAE: {mae_mean:7.4f} +/- {mae_std:6.4f} ({len(current)} seeds)"
            )

    raw_df = pd.DataFrame(rows)
    agg_df = aggregate_ablation(raw_df)
    return raw_df, agg_df


def run_optuna_tuning(
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    | None = None,
    n_trials: int = 15,
    seed: int = SEED,
    train_rows: int | None = 100_000,
    epochs: int = 10,
    verbose: bool = True,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Run validation-only hyperparameter tuning using Optuna.

    Explores architecture space (depth, width, shape, activation, lr, weight decay)
    strictly evaluating on validation MAE with zero test leakage.

    Returns
    -------
    tuple[dict[str, Any], pd.DataFrame]
        (best_params_dict, trials_dataframe)
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.INFO if verbose else optuna.logging.WARNING)

    if arrays is None:
        arrays = prepare_data(train_rows=train_rows, seed=seed)

    trials_records: list[dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        depth = trial.suggest_int("depth", 1, 4)
        base_width = trial.suggest_categorical("base_width", [64, 128, 256, 384])
        shape = trial.suggest_categorical("shape", ["funnel", "uniform"])
        activation = trial.suggest_categorical("activation", ["relu", "gelu", "silu"])
        lr = trial.suggest_float("lr", 3e-4, 3e-3, log=True)
        weight_decay = trial.suggest_categorical("weight_decay", [0.0, 1e-5, 1e-4])

        # Derive hidden dimensions
        if shape == "uniform":
            hidden_dims = tuple(base_width for _ in range(depth))
        else:  # funnel
            dims: list[int] = []
            curr = base_width
            for _ in range(depth):
                dims.append(max(curr, 32))
                curr //= 2
            hidden_dims = tuple(dims)

        overrides = {
            "epochs": epochs,
            "hidden_dims": hidden_dims,
            "activation": activation,
            "lr": lr,
            "weight_decay": weight_decay,
            "batch_size": 512,
        }

        # Train on train split, evaluate strictly on validation
        res = run_config(arrays, overrides=overrides, seed=seed, eval_test=False, verbose=False)
        val_mae = float(res["val_mae"])

        trials_records.append(
            {
                "trial_number": trial.number,
                "depth": depth,
                "base_width": base_width,
                "shape": shape,
                "hidden_dims": str(hidden_dims),
                "activation": activation,
                "lr": lr,
                "weight_decay": weight_decay,
                "n_params": res["n_params"],
                "val_mae": val_mae,
                "val_rmse": res["val_rmse"],
                "val_r2": res["val_r2"],
            }
        )
        return val_mae

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    best_trial = study.best_trial
    trials_df = pd.DataFrame(trials_records)

    # Reconstruct best configuration dictionary
    best_depth = best_trial.params["depth"]
    best_base_width = best_trial.params["base_width"]
    best_shape = best_trial.params["shape"]
    if best_shape == "uniform":
        best_hidden_dims = tuple(best_base_width for _ in range(best_depth))
    else:
        dims = []
        curr = best_base_width
        for _ in range(best_depth):
            dims.append(max(curr, 32))
            curr //= 2
        best_hidden_dims = tuple(dims)

    best_cfg = {
        "hidden_dims": best_hidden_dims,
        "activation": best_trial.params["activation"],
        "lr": best_trial.params["lr"],
        "weight_decay": best_trial.params["weight_decay"],
        "val_mae": best_trial.value,
    }

    if verbose:
        print(f"\n[Optuna Tuning Complete] Best Trial #{best_trial.number}:")
        print(f"  Architecture: {best_hidden_dims} ({best_trial.params['activation']})")
        print(
            f"  LR: {best_trial.params['lr']:.5f}, Weight Decay: {best_trial.params['weight_decay']}"
        )
        print(f"  Validation MAE: {best_trial.value:.4f}")

    return best_cfg, trials_df


def run_multiseed_evaluation(
    config: dict[str, Any],
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    | None = None,
    seeds: Sequence[int] = TEAM_SEEDS,
    train_rows: int | None = 200_000,
    eval_test: bool = True,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Evaluate a chosen architecture across all 5 team seeds on validation and test splits."""
    if arrays is None:
        arrays = prepare_data(train_rows=train_rows)

    records: list[dict[str, Any]] = []
    first_history: dict[str, list[float]] | None = None
    for s in seeds:
        res = run_config(arrays, overrides=config, seed=s, eval_test=eval_test, verbose=False)
        hist = res.pop("history", None)
        if first_history is None and hist is not None:
            first_history = hist
        records.append(res)
        if verbose:
            print(
                f"  Seed {s:4d} | Val MAE: {res['val_mae']:.4f} | Test MAE: {res.get('test_mae', np.nan):.4f}"
            )

    df = pd.DataFrame(records)
    summary = {
        "val_mae_mean": float(df["val_mae"].mean()),
        "val_mae_std": float(df["val_mae"].std()),
        "val_rmse_mean": float(df["val_rmse"].mean()),
        "val_rmse_std": float(df["val_rmse"].std()),
        "val_r2_mean": float(df["val_r2"].mean()),
        "val_r2_std": float(df["val_r2"].std()),
    }
    if eval_test and "test_mae" in df.columns:
        summary.update(
            {
                "test_mae_mean": float(df["test_mae"].mean()),
                "test_mae_std": float(df["test_mae"].std()),
                "test_rmse_mean": float(df["test_rmse"].mean()),
                "test_rmse_std": float(df["test_rmse"].std()),
                "test_r2_mean": float(df["test_r2"].mean()),
                "test_r2_std": float(df["test_r2"].std()),
            }
        )

    return df, summary, first_history


def generate_architecture_plots(
    agg_df: pd.DataFrame,
    trials_df: pd.DataFrame | None = None,
    best_history: dict[str, list[float]] | None = None,
) -> None:
    """Generate diagnostic comparison plots and save to reports/figures/."""
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)

    # 1. Depth sweep plot
    depth_df = agg_df[agg_df["family"] == "depth"].copy()
    if not depth_df.empty and "depth" in depth_df.columns:
        depth_df = depth_df.sort_values("depth")
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.errorbar(
            depth_df["depth"],
            depth_df["val_mae_mean"],
            yerr=depth_df.get("val_mae_std", 0.0),
            marker="o",
            color="#1f77b4",
            capsize=4,
            linewidth=2,
        )
        ax.set_title("Architecture Ablation: Effect of Depth (Number of Hidden Layers)")
        ax.set_xlabel("Number of Hidden Layers (Depth)")
        ax.set_ylabel("Validation MAE (Lower is Better)")
        ax.grid(True, linestyle="--", alpha=0.6)
        fig.tight_layout()
        fig.savefig(FIGURES / "architecture_depth_sweep.png", dpi=300)
        plt.close(fig)

    # 2. Width sweep plot
    width_df = agg_df[agg_df["family"] == "width"].copy()
    if not width_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        y_pos = np.arange(len(width_df))
        ax.barh(
            y_pos,
            width_df["val_mae_mean"],
            xerr=width_df.get("val_mae_std", 0.0),
            color="#2ca02c",
            alpha=0.85,
            capsize=4,
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(width_df["config"])
        ax.set_xlabel("Validation MAE (Lower is Better)")
        ax.set_title("Architecture Ablation: Hidden Layer Width Variations")
        ax.grid(True, linestyle="--", alpha=0.6, axis="x")
        fig.tight_layout()
        fig.savefig(FIGURES / "architecture_width_sweep.png", dpi=300)
        plt.close(fig)

    # 3. Activation comparison plot
    act_df = agg_df[agg_df["family"] == "activation"].copy()
    if not act_df.empty:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        bars = ax.bar(
            act_df["config"],
            act_df["val_mae_mean"],
            yerr=act_df.get("val_mae_std", 0.0),
            color="#ff7f0e",
            alpha=0.85,
            capsize=4,
        )
        ax.set_ylabel("Validation MAE (Lower is Better)")
        ax.set_title("Architecture Ablation: Activation Function Comparison")
        ax.grid(True, linestyle="--", alpha=0.6, axis="y")
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        fig.tight_layout()
        fig.savefig(FIGURES / "architecture_activation_comparison.png", dpi=300)
        plt.close(fig)

    # 4. Optuna trials trajectory plot
    if trials_df is not None and not trials_df.empty and "val_mae" in trials_df.columns:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        running_min = trials_df["val_mae"].cummin()
        ax.scatter(
            trials_df["trial_number"],
            trials_df["val_mae"],
            color="#9467bd",
            alpha=0.7,
            label="Trial Validation MAE",
        )
        ax.plot(
            trials_df["trial_number"],
            running_min,
            color="#d62728",
            linewidth=2,
            linestyle="--",
            label="Best So Far",
        )
        ax.set_xlabel("Optuna Trial Number")
        ax.set_ylabel("Validation MAE (Lower is Better)")
        ax.set_title("Optuna Hyperparameter Optimization Progress")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)
        fig.tight_layout()
        fig.savefig(FIGURES / "architecture_optuna_study.png", dpi=300)
        plt.close(fig)

    # 5. Tuned MLP Learning Curves
    if best_history and "train_loss" in best_history and "val_loss" in best_history:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        epochs_range = range(1, len(best_history["train_loss"]) + 1)
        axes[0].plot(
            epochs_range, best_history["train_loss"], label="Train Loss (MSE)", color="#1f77b4"
        )
        axes[0].plot(
            epochs_range,
            best_history["val_loss"],
            label="Val Loss (MSE)",
            color="#d62728",
            linestyle="--",
        )
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss (MSE)")
        axes[0].set_title("Tuned MLP: Training vs. Validation Loss")
        axes[0].legend()
        axes[0].grid(True, linestyle="--", alpha=0.6)

        if "val_r2" in best_history and len(best_history["val_r2"]) == len(epochs_range):
            axes[1].plot(epochs_range, best_history["val_r2"], label="Val R²", color="#2ca02c")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Validation R²")
            axes[1].set_title("Tuned MLP: Validation R² Trajectory")
            axes[1].legend()
            axes[1].grid(True, linestyle="--", alpha=0.6)

        fig.tight_layout()
        fig.savefig(FIGURES / "architecture_best_mlp_learning_curves.png", dpi=300)
        plt.close(fig)


def build_mlp_results_table(
    base_mlp_summary: dict[str, float],
    tuned_mlp_summary: dict[str, float],
    n_params_base: int,
    n_params_tuned: int,
) -> pd.DataFrame:
    """Build the final comparative mlp_results.csv table against baselines."""
    # Benchmarks from Phase 1 and Naive Baseline
    rows = [
        {
            "model": "Naive Baseline (Mean)",
            "architecture": "Statistical Mean",
            "n_params": 1,
            "val_mae_mean": 14.8069,
            "val_mae_std": 0.0,
            "val_rmse_mean": 18.5284,
            "val_rmse_std": 0.0,
            "val_r2_mean": 0.0,
            "val_r2_std": 0.0,
            "test_mae": 14.7725,
            "test_rmse": 18.5081,
            "test_r2": 0.0,
            "notes": "Global training mean baseline",
        },
        {
            "model": "Random Forest (Phase 1)",
            "architecture": "100 Trees, max_depth=15",
            "n_params": 2_340_000,
            "val_mae_mean": 0.4031,
            "val_mae_std": 0.0,
            "val_rmse_mean": 0.9075,
            "val_rmse_std": 0.0,
            "val_r2_mean": 0.9976,
            "val_r2_std": 0.0,
            "test_mae": 0.4078,
            "test_rmse": 0.9197,
            "test_r2": 0.9976,
            "notes": "Phase 1 tree benchmark",
        },
        {
            "model": "XGBoost (Phase 1)",
            "architecture": "1991 Trees, depth=7, tuned",
            "n_params": 1_250_000,
            "val_mae_mean": 0.1040,
            "val_mae_std": 0.0,
            "val_rmse_mean": 0.3549,
            "val_rmse_std": 0.0,
            "val_r2_mean": 0.9996,
            "val_r2_std": 0.0,
            "test_mae": 0.0999,
            "test_rmse": 0.3282,
            "test_r2": 0.9997,
            "notes": "Phase 1 champion tree model",
        },
        {
            "model": "Base MLP (Phase 2)",
            "architecture": "128-64 ReLU",
            "n_params": n_params_base,
            "val_mae_mean": round(base_mlp_summary.get("val_mae_mean", 0.7751), 4),
            "val_mae_std": round(base_mlp_summary.get("val_mae_std", 0.0444), 4),
            "val_rmse_mean": round(base_mlp_summary.get("val_rmse_mean", 1.2571), 4),
            "val_rmse_std": round(base_mlp_summary.get("val_rmse_std", 0.0412), 4),
            "val_r2_mean": round(base_mlp_summary.get("val_r2_mean", 0.9954), 4),
            "val_r2_std": round(base_mlp_summary.get("val_r2_std", 0.0003), 4),
            "test_mae": round(base_mlp_summary.get("test_mae_mean", 0.7774), 4),
            "test_rmse": round(base_mlp_summary.get("test_rmse_mean", 1.2554), 4),
            "test_r2": round(base_mlp_summary.get("test_r2_mean", 0.9955), 4),
            "notes": "Untuned baseline feedforward network (Step 2)",
        },
        {
            "model": "Tuned MLP (Phase 2)",
            "architecture": f"{tuned_mlp_summary.get('architecture', 'Tuned')} {tuned_mlp_summary.get('activation', 'GELU')}",
            "n_params": n_params_tuned,
            "val_mae_mean": round(tuned_mlp_summary.get("val_mae_mean", 0.0), 4),
            "val_mae_std": round(tuned_mlp_summary.get("val_mae_std", 0.0), 4),
            "val_rmse_mean": round(tuned_mlp_summary.get("val_rmse_mean", 0.0), 4),
            "val_rmse_std": round(tuned_mlp_summary.get("val_rmse_std", 0.0), 4),
            "val_r2_mean": round(tuned_mlp_summary.get("val_r2_mean", 0.0), 4),
            "val_r2_std": round(tuned_mlp_summary.get("val_r2_std", 0.0), 4),
            "test_mae": round(tuned_mlp_summary.get("test_mae_mean", 0.0), 4),
            "test_rmse": round(tuned_mlp_summary.get("test_rmse_mean", 0.0), 4),
            "test_r2": round(tuned_mlp_summary.get("test_r2_mean", 0.0), 4),
            "notes": "Optimal architecture after Step 3 ablation & tuning",
        },
    ]

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 3: Architecture ablation & tuning")
    parser.add_argument(
        "--quick", action="store_true", help="Quick run (1 seed, 50k rows, 5 epochs)"
    )
    parser.add_argument("--train-rows", type=int, default=200_000, help="Train subsample size")
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds to run (1 to 5)")
    parser.add_argument("--trials", type=int, default=15, help="Number of Optuna tuning trials")
    parser.add_argument(
        "--append-seeds",
        type=int,
        nargs="+",
        help="Extend existing ablation sweep with additional seeds instead of re-running from scratch.",
    )
    args = parser.parse_args()

    if args.append_seeds:
        raw_csv = REPORTS / "mlp_ablation_raw.csv"
        agg_csv = REPORTS / "mlp_ablation.csv"
        if not raw_csv.exists():
            raise FileNotFoundError(f"Cannot append seeds: {raw_csv} does not exist.")
        existing_raw = pd.read_csv(raw_csv)
        have = set(existing_raw["seed"].unique())
        todo = [s for s in args.append_seeds if s not in have]
        arrays = prepare_data(train_rows=args.train_rows)
        if todo:
            print(f"Existing seeds in sweep: {sorted(have)} -> Running additional seeds: {todo}\n")
            extra_raw, _ = run_architecture_ablation(
                arrays=arrays,
                seeds=todo,
                train_rows=args.train_rows,
                grid=ARCHITECTURE_GRID,
                verbose=True,
            )
            raw_df = pd.concat([existing_raw, extra_raw], ignore_index=True)
            agg_df = aggregate_ablation(raw_df)
            raw_df.to_csv(raw_csv, index=False)
            agg_df.to_csv(agg_csv, index=False)
            print(
                f"Updated: {raw_csv} and {agg_csv} ({len(raw_df)} runs across {len(agg_df)} configs)"
            )
        else:
            print(f"Seeds {args.append_seeds} already present in {raw_csv}.")
            raw_df = existing_raw
            agg_df = aggregate_ablation(raw_df)

        all_seeds = [int(s) for s in sorted(raw_df["seed"].unique())]
        print(
            f"\n--- Multi-Seed Confirmation on Validation & Test ({len(all_seeds)} seeds: {all_seeds}) ---"
        )
        trials_csv = REPORTS / "mlp_optuna_trials.csv"
        if trials_csv.exists():
            trials_df = pd.read_csv(trials_csv)
            best_idx = trials_df["val_mae"].idxmin()
            best_row = trials_df.loc[best_idx]
            import ast

            best_hidden = (
                ast.literal_eval(best_row["hidden_dims"])
                if isinstance(best_row["hidden_dims"], str)
                else best_row["hidden_dims"]
            )
            best_cfg = {
                "hidden_dims": tuple(best_hidden),
                "activation": best_row["activation"],
                "lr": float(best_row["lr"]),
                "weight_decay": float(best_row["weight_decay"]),
            }
        else:
            best_cfg = {
                "hidden_dims": (256, 256, 256, 256),
                "activation": "gelu",
                "lr": 0.00219,
                "weight_decay": 0.0,
            }
            trials_df = None

        print(f"Tuned configuration: {best_cfg}")
        tuned_df, tuned_summary, tuned_history = run_multiseed_evaluation(
            config=best_cfg,
            arrays=arrays,
            seeds=all_seeds,
            train_rows=args.train_rows,
            eval_test=True,
            verbose=True,
        )
        tuned_summary["architecture"] = str(best_cfg["hidden_dims"])
        tuned_summary["activation"] = str(best_cfg["activation"])

        base_df, base_summary, _ = run_multiseed_evaluation(
            config=BASE_CONFIG,
            arrays=arrays,
            seeds=all_seeds,
            train_rows=args.train_rows,
            eval_test=True,
            verbose=False,
        )

        dummy_model_base = WaterQualityMLP(arrays[0].shape[1], BASE_CONFIG["hidden_dims"])
        dummy_model_tuned = WaterQualityMLP(
            arrays[0].shape[1],
            best_cfg["hidden_dims"],
            activation=best_cfg["activation"],
        )

        results_table = build_mlp_results_table(
            base_mlp_summary=base_summary,
            tuned_mlp_summary=tuned_summary,
            n_params_base=sum(p.numel() for p in dummy_model_base.parameters()),
            n_params_tuned=sum(p.numel() for p in dummy_model_tuned.parameters()),
        )

        results_csv = REPORTS / "mlp_results.csv"
        results_table.to_csv(results_csv, index=False)
        print(f"\nSaved final results to: {results_csv}")
        print(results_table.to_string(index=False))

        print("\n--- Generating Architecture Diagnostic Visualizations ---")
        generate_architecture_plots(agg_df, trials_df, tuned_history)
        print("Visualizations saved to reports/figures/")
        return

    seeds = [SEED] if args.quick else TEAM_SEEDS[: args.seeds]
    train_rows = 20_000 if args.quick else args.train_rows
    epochs = 3 if args.quick else 15
    n_trials = 3 if args.quick else args.trials
    grid = [DEPTH_GRID[0], WIDTH_GRID[0], ACTIVATION_GRID[0]] if args.quick else ARCHITECTURE_GRID

    print("=== Starting Architecture Ablation & Tuning ===")
    print(f"Seeds: {seeds} | Train rows: {train_rows:,} | Max Epochs: {epochs}")

    arrays = prepare_data(train_rows=train_rows)

    # 1. Systematic Ablation
    print("\n--- Running Architecture Ablation (Depth, Width, Activation) ---")
    raw_df, agg_df = run_architecture_ablation(
        arrays=arrays,
        seeds=seeds,
        train_rows=train_rows,
        grid=grid,
        verbose=True,
    )
    raw_csv = REPORTS / "mlp_ablation_raw.csv"
    agg_csv = REPORTS / "mlp_ablation.csv"
    raw_df.to_csv(raw_csv, index=False)
    agg_df.to_csv(agg_csv, index=False)
    print(f"Saved: {raw_csv} and {agg_csv}")

    # 2. Optuna Tuning
    print(f"\n--- Running Optuna Tuning ({n_trials} trials on Validation Split) ---")
    best_cfg, trials_df = run_optuna_tuning(
        arrays=arrays,
        n_trials=n_trials,
        train_rows=train_rows,
        epochs=epochs,
        verbose=True,
    )
    trials_csv = REPORTS / "mlp_optuna_trials.csv"
    trials_df.to_csv(trials_csv, index=False)
    print(f"Saved: {trials_csv}")

    # 3. Multi-seed Confirmation of Tuned Model
    print("\n--- Multi-Seed Confirmation of Tuned Model on Validation & Test ---")
    tuned_df, tuned_summary, tuned_history = run_multiseed_evaluation(
        config=best_cfg,
        arrays=arrays,
        seeds=seeds,
        train_rows=train_rows,
        eval_test=True,
        verbose=True,
    )
    tuned_summary["architecture"] = str(best_cfg["hidden_dims"])
    tuned_summary["activation"] = str(best_cfg["activation"])

    # Base MLP multi-seed reference
    base_df, base_summary, _ = run_multiseed_evaluation(
        config=BASE_CONFIG,
        arrays=arrays,
        seeds=seeds,
        train_rows=train_rows,
        eval_test=True,
        verbose=False,
    )

    dummy_model_base = WaterQualityMLP(arrays[0].shape[1], BASE_CONFIG["hidden_dims"])
    dummy_model_tuned = WaterQualityMLP(
        arrays[0].shape[1],
        best_cfg["hidden_dims"],
        activation=best_cfg["activation"],
    )

    results_table = build_mlp_results_table(
        base_mlp_summary=base_summary,
        tuned_mlp_summary=tuned_summary,
        n_params_base=sum(p.numel() for p in dummy_model_base.parameters()),
        n_params_tuned=sum(p.numel() for p in dummy_model_tuned.parameters()),
    )

    results_csv = REPORTS / "mlp_results.csv"
    results_table.to_csv(results_csv, index=False)
    print(f"\nSaved final results to: {results_csv}")
    print(results_table.to_string(index=False))

    # 4. Diagnostic Visualizations
    print("\n--- Generating Architecture Diagnostic Visualizations ---")
    generate_architecture_plots(agg_df, trials_df, tuned_history)
    print("Visualizations saved to reports/figures/")


if __name__ == "__main__":
    main()
