"""Reusable PyTorch training loop and experiment logger for Phase 2 deep learning.

Enforces:
  * Regression loss aligned with continuous CCME_Values target (default: MSELoss).
  * Validation evaluation every epoch using the shared Phase 1 evaluate() contract.
  * Early stopping on validation loss with automatic best-weight checkpoint restoration.
  * Checkpoint persistence to an ignored folder (checkpoints/).
  * Optional learning rate schedulers and configurable weight decay.
  * Centralized experiment logging to reports/dl_runs.csv.

Owner: Parthraj Singh Bhati.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import REPORTS, SEED
from src.dl.seeds import set_seed
from src.evaluate_api import evaluate

DEFAULT_RUNS_CSV = REPORTS / "dl_runs.csv"
DEFAULT_CHECKPOINT_DIR = Path("checkpoints")


def get_loss_function(loss_name: str | nn.Module = "mse") -> nn.Module:
    """Return appropriate regression loss matching continuous CCME_Values target."""
    if isinstance(loss_name, nn.Module):
        return loss_name

    loss_str = str(loss_name).lower()
    if loss_str in ("mse", "mean_squared_error"):
        return nn.MSELoss()
    elif loss_str in ("l1", "mae", "mean_absolute_error"):
        return nn.L1Loss()
    elif loss_str in ("smooth_l1", "huber"):
        return nn.SmoothL1Loss()
    else:
        raise ValueError(
            f"Unsupported loss function '{loss_name}'. Expected 'mse', 'l1', or 'smooth_l1'."
        )


def get_device(device_arg: str | torch.device | None = None) -> torch.device:
    """Resolve target compute device (CPU, CUDA, or MPS)."""
    if device_arg is not None:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def log_run(
    config: dict[str, Any],
    seed: int,
    val_metrics: dict[str, float],
    epochs_trained: int,
    best_epoch: int,
    best_val_loss: float,
    test_metrics: dict[str, float] | None = None,
    scale_target: bool = False,
    csv_path: Path | str | None = None,
) -> Path:
    """Append experiment run metadata, hyperparameters, and evaluation metrics to dl_runs.csv."""
    target_csv = Path(csv_path) if csv_path else DEFAULT_RUNS_CSV
    target_csv.parent.mkdir(parents=True, exist_ok=True)

    row: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment_label": config.get("experiment_label", config.get("model_name", "dl_model")),
        "seed": seed,
        "scale_target": bool(scale_target),
        "epochs_trained": epochs_trained,
        "best_epoch": best_epoch,
        "batch_size": config.get("batch_size", 256),
        "learning_rate": config.get("lr", 1e-3),
        "weight_decay": config.get("weight_decay", 1e-4),
        "optimizer": config.get("optimizer", "adamw"),
        "scheduler": config.get("scheduler", "none"),
        "best_val_loss": round(float(best_val_loss), 6),
        "val_mae": round(float(val_metrics.get("mae", np.nan)), 6),
        "val_rmse": round(float(val_metrics.get("rmse", np.nan)), 6),
        "val_r2": round(float(val_metrics.get("r2", np.nan)), 6),
    }

    if test_metrics is not None:
        row["test_mae"] = round(float(test_metrics.get("mae", np.nan)), 6)
        row["test_rmse"] = round(float(test_metrics.get("rmse", np.nan)), 6)
        row["test_r2"] = round(float(test_metrics.get("r2", np.nan)), 6)
    else:
        row["test_mae"] = np.nan
        row["test_rmse"] = np.nan
        row["test_r2"] = np.nan

    df_row = pd.DataFrame([row])
    if target_csv.exists() and target_csv.stat().st_size > 0:
        existing_df = pd.read_csv(target_csv)
        if "scale_target" not in existing_df.columns:
            loc = (
                list(existing_df.columns).index("seed") + 1 if "seed" in existing_df.columns else 3
            )
            existing_df.insert(loc, "scale_target", False)
        combined_df = pd.concat([existing_df, df_row], ignore_index=True)
        combined_df.to_csv(target_csv, index=False)
    else:
        df_row.to_csv(target_csv, index=False)

    return target_csv


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    y_mean: float = 0.0,
    y_std: float = 1.0,
    scale_target: bool = False,
) -> tuple[float, dict[str, float]]:
    """Evaluate a DataLoader and return average loss plus shared evaluate() metrics.

    Parameters
    ----------
    model : nn.Module
        Neural network model.
    loader : DataLoader
        DataLoader for validation or testing.
    criterion : nn.Module
        Loss function.
    device : torch.device
        Compute device.
    y_mean : float, default 0.0
        Mean used to standardize targets (train split only).
    y_std : float, default 1.0
        Standard deviation used to standardize targets (train split only).
    scale_target : bool, default False
        Whether predictions and targets should be inverse-transformed before evaluate().

    Returns
    -------
    tuple[float, dict[str, float]]
        (average_loss, metrics_dict)
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    y_trues: list[np.ndarray] = []
    y_preds: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                X_batch, y_batch = batch[0].to(device), batch[1].to(device)
            else:
                X_batch = batch.to(device)
                y_batch = None

            preds = model(X_batch)

            if y_batch is not None:
                loss = criterion(preds, y_batch)
                batch_size = len(X_batch)
                total_loss += float(loss.item()) * batch_size
                total_samples += batch_size

                y_trues.append(y_batch.detach().cpu().numpy())
                y_preds.append(preds.detach().cpu().numpy())

    avg_loss = total_loss / max(total_samples, 1)

    if y_trues:
        y_true_cat = np.concatenate(y_trues, axis=0).ravel()
        y_pred_cat = np.concatenate(y_preds, axis=0).ravel()

        if scale_target:
            y_true_eval = y_true_cat * y_std + y_mean
            y_pred_eval = y_pred_cat * y_std + y_mean
        else:
            y_true_eval = y_true_cat
            y_pred_eval = y_pred_cat

        metrics = evaluate(y_true_eval, y_pred_eval)
    else:
        metrics = {}

    return avg_loss, metrics


def train(
    model: nn.Module,
    loaders: Any,
    config: dict[str, Any] | None = None,
) -> tuple[dict[str, list[float]], dict[str, float]]:
    """Reusable training loop for Phase 2 neural network architectures.

    Features:
      1. Target-appropriate regression loss (MSE / L1 / SmoothL1).
      2. Validation every epoch evaluated via shared evaluate() metrics.
      3. Early stopping tracking validation loss with min_delta.
      4. Best-checkpoint saving to an ignored folder, reloaded at completion.
      5. Configurable weight decay and optional learning rate scheduler.
      6. Automatic append of experiment summary to reports/dl_runs.csv.

    Parameters
    ----------
    model : nn.Module
        PyTorch neural network model to train.
    loaders : LoadersBundle | dict[str, DataLoader] | tuple | list
        Collection containing at least 'train' and 'val' DataLoaders.
    config : dict[str, Any], optional
        Training configuration options:
          - 'epochs': int, max training epochs (default: 20)
          - 'lr': float, initial learning rate (default: 1e-3)
          - 'weight_decay': float, L2 regularization (default: 1e-4)
          - 'optimizer': str, 'adamw' | 'adam' | 'sgd' (default: 'adamw')
          - 'loss': str | nn.Module, loss criterion (default: 'mse')
          - 'early_stopping_patience': int, epochs to wait (default: 5)
          - 'early_stopping_min_delta': float, min loss decrease (default: 1e-4)
          - 'scheduler': str | None, 'plateau' | 'cosine' | 'step' | None (default: None)
          - 'checkpoint_dir': Path | str, folder for model weights (default: 'checkpoints')
          - 'model_name': str, filename for saved checkpoint (default: 'best_model.pt')
          - 'device': str | torch.device | None, execution device (default: auto)
          - 'seed': int, seed for reproducibility (default: 42)
          - 'log_to_csv': bool, whether to append to reports/dl_runs.csv (default: True)
          - 'csv_path': Path | str | None, path for run logging
          - 'eval_test': bool, evaluate test_loader with best model at end (default: False)
          - 'verbose': bool, whether to print epoch progress (default: True)

    Returns
    -------
    tuple[dict[str, list[float]], dict[str, float]]
        (history, best_val_metrics)
        - history contains 'train_loss', 'val_loss', 'val_mae', 'val_rmse', 'val_r2', 'lr' per epoch.
        - best_val_metrics contains the evaluate() dictionary at the best validation epoch.
    """
    cfg = config.copy() if config else {}

    seed = int(cfg.get("seed", SEED))
    set_seed(seed)

    # 1. Resolve loaders
    if isinstance(loaders, dict):
        train_loader = loaders["train"]
        val_loader = loaders.get("val")
        test_loader = loaders.get("test")
    elif isinstance(loaders, (tuple, list)):
        train_loader = loaders[0]
        val_loader = loaders[1] if len(loaders) > 1 else None
        test_loader = loaders[2] if len(loaders) > 2 else None
    else:
        train_loader = getattr(loaders, "train", None)
        val_loader = getattr(loaders, "val", None)
        test_loader = getattr(loaders, "test", None)

    if train_loader is None or val_loader is None:
        raise ValueError("loaders must provide both 'train' and 'val' DataLoaders.")

    # Target scaling parameters
    scale_target = bool(cfg.get("scale_target", getattr(loaders, "scale_target", False)))
    y_mean = float(cfg.get("y_mean", getattr(loaders, "y_mean", 0.0)))
    y_std = float(cfg.get("y_std", getattr(loaders, "y_std", 1.0)))

    # 2. Resolve compute device and move model
    device = get_device(cfg.get("device"))
    model = model.to(device)

    # 3. Setup loss criterion
    criterion = get_loss_function(cfg.get("loss", "mse")).to(device)

    # 4. Setup optimizer with weight decay
    lr = float(cfg.get("lr", 1e-3))
    weight_decay = float(cfg.get("weight_decay", 1e-4))
    opt_name = str(cfg.get("optimizer", "adamw")).lower()

    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9
        )
    else:
        raise ValueError(f"Unsupported optimizer '{opt_name}'. Expected 'adamw', 'adam', or 'sgd'.")

    # 5. Setup optional learning rate scheduler
    scheduler_type = cfg.get("scheduler")
    epochs = int(cfg.get("epochs", 20))
    scheduler: Any = None

    if scheduler_type:
        s_type = str(scheduler_type).lower()
        if s_type in ("plateau", "reduce_on_plateau"):
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=float(cfg.get("scheduler_factor", 0.5)),
                patience=int(cfg.get("scheduler_patience", 2)),
                min_lr=float(cfg.get("min_lr", 1e-6)),
            )
        elif s_type in ("cosine", "cosine_annealing"):
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=epochs,
                eta_min=float(cfg.get("min_lr", 1e-6)),
            )
        elif s_type in ("step", "step_lr"):
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=int(cfg.get("step_size", 5)),
                gamma=float(cfg.get("gamma", 0.5)),
            )

    # 6. Setup checkpointing and early stopping
    patience = int(cfg.get("early_stopping_patience", cfg.get("patience", 5)))
    min_delta = float(cfg.get("early_stopping_min_delta", cfg.get("min_delta", 1e-4)))
    checkpoint_dir = Path(cfg.get("checkpoint_dir", DEFAULT_CHECKPOINT_DIR))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_name = str(cfg.get("model_name", "best_model.pt"))
    checkpoint_path = checkpoint_dir / model_name

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_mae": [],
        "val_rmse": [],
        "val_r2": [],
        "lr": [],
    }

    best_val_loss = float("inf")
    best_val_metrics: dict[str, float] = {}
    best_epoch = 0
    patience_counter = 0
    verbose = bool(cfg.get("verbose", True))

    # 7. Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        running_train_loss = 0.0
        train_samples = 0

        for batch in train_loader:
            X_b, y_b = batch[0].to(device), batch[1].to(device)

            optimizer.zero_grad()
            preds = model(X_b)
            loss = criterion(preds, y_b)
            loss.backward()
            optimizer.step()

            b_len = len(X_b)
            running_train_loss += float(loss.item()) * b_len
            train_samples += b_len

        epoch_train_loss = running_train_loss / max(train_samples, 1)

        # Validation step
        val_loss, val_metrics = evaluate_loader(
            model,
            val_loader,
            criterion,
            device,
            y_mean=y_mean,
            y_std=y_std,
            scale_target=scale_target,
        )

        # Update scheduler
        current_lr = float(optimizer.param_groups[0]["lr"])
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Record history
        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(float(val_metrics.get("mae", np.nan)))
        history["val_rmse"].append(float(val_metrics.get("rmse", np.nan)))
        history["val_r2"].append(float(val_metrics.get("r2", np.nan)))
        history["lr"].append(current_lr)

        if verbose:
            print(
                f"Epoch {epoch:02d}/{epochs:02d} | "
                f"Train Loss: {epoch_train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val MAE: {val_metrics.get('mae', 0.0):.4f} | "
                f"Val RMSE: {val_metrics.get('rmse', 0.0):.4f} | "
                f"Val R²: {val_metrics.get('r2', 0.0):.4f} | "
                f"LR: {current_lr:.2e}"
            )

        # Early stopping and checkpointing check
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_val_metrics = dict(val_metrics)
            best_epoch = epoch
            patience_counter = 0

            # Save best checkpoint
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "val_metrics": best_val_metrics,
                    "config": cfg,
                    "seed": seed,
                    "y_mean": y_mean,
                    "y_std": y_std,
                    "scale_target": scale_target,
                },
                checkpoint_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(
                        f"Early stopping triggered at epoch {epoch}. "
                        f"Best validation loss {best_val_loss:.4f} at epoch {best_epoch}."
                    )
                break

    # 8. Reload best checkpoint weights into model
    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if verbose:
            print(f"Loaded best checkpoint weights from {checkpoint_path} (epoch {best_epoch}).")

    # 9. Optional test set evaluation
    test_metrics: dict[str, float] | None = None
    if test_loader is not None and bool(cfg.get("eval_test", False)):
        _, test_metrics = evaluate_loader(
            model,
            test_loader,
            criterion,
            device,
            y_mean=y_mean,
            y_std=y_std,
            scale_target=scale_target,
        )
        if verbose:
            print(
                f"Final Test Evaluation: MAE={test_metrics.get('mae', 0.0):.4f}, "
                f"RMSE={test_metrics.get('rmse', 0.0):.4f}, "
                f"R²={test_metrics.get('r2', 0.0):.4f}"
            )

    # 10. Centralized run logging
    if bool(cfg.get("log_to_csv", True)):
        log_run(
            config=cfg,
            seed=seed,
            val_metrics=best_val_metrics,
            epochs_trained=len(history["train_loss"]),
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
            test_metrics=test_metrics,
            scale_target=scale_target,
            csv_path=cfg.get("csv_path"),
        )

    return history, best_val_metrics
