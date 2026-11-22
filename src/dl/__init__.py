"""Deep learning package for Phase 2 Water Quality Prediction.

Provides:
  * Seeds & reproducibility: set_seed, seed_worker
  * Dataset & DataLoader: WaterQualityDataset, LoadersBundle, get_loaders
  * Models: WaterQualityMLP, create_mlp, get_activation
  * Training & Evaluation: train, evaluate_loader, log_run, get_loss_function, get_device

Contributors:
  * Parthraj Singh Bhati (Data loaders, training loop, seeds)
  * Himanshu Pandey (Model architecture, base MLP, ablations)
"""

from __future__ import annotations

from typing import Any

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "build_nn_preprocessor": ("src.dl.nn_preprocess", "build_nn_preprocessor"),
    "transform_splits": ("src.dl.nn_preprocess", "transform_splits"),
    "set_seed": ("src.dl.seeds", "set_seed"),
    "seed_worker": ("src.dl.seeds", "seed_worker"),
    "WaterQualityDataset": ("src.dl.data", "WaterQualityDataset"),
    "LoadersBundle": ("src.dl.data", "LoadersBundle"),
    "get_loaders": ("src.dl.data", "get_loaders"),
    "WaterQualityMLP": ("src.dl.model", "WaterQualityMLP"),
    "create_mlp": ("src.dl.model", "create_mlp"),
    "get_activation": ("src.dl.model", "get_activation"),
    "train": ("src.dl.train", "train"),
    "evaluate_loader": ("src.dl.train", "evaluate_loader"),
    "log_run": ("src.dl.train", "log_run"),
    "get_loss_function": ("src.dl.train", "get_loss_function"),
    "get_device": ("src.dl.train", "get_device"),
    "aggregate_architecture": ("src.dl.architecture", "aggregate_architecture"),
    "mc_dropout_predictions": ("src.dl.calibration", "mc_dropout_predictions"),
    "prediction_intervals": ("src.dl.calibration", "prediction_intervals"),
    "expected_calibration_error": ("src.dl.calibration", "expected_calibration_error"),
    "summarize_seed_runs": ("src.dl.comparison", "summarize_seed_runs"),
    "make_oof_predictions": ("src.dl.hybrid", "make_oof_predictions"),
    "evaluate_hybrid": ("src.dl.hybrid", "evaluate_hybrid"),
}


def __getattr__(name: str) -> Any:
    """Load public DL classes and functions only when requested."""
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_IMPORTS[name]
    from importlib import import_module

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value

__all__ = [
    "build_nn_preprocessor",
    "transform_splits",
    "set_seed",
    "seed_worker",
    "WaterQualityDataset",
    "LoadersBundle",
    "get_loaders",
    "WaterQualityMLP",
    "create_mlp",
    "get_activation",
    "train",
    "evaluate_loader",
    "log_run",
    "get_loss_function",
    "get_device",
    "aggregate_architecture",
    "mc_dropout_predictions",
    "prediction_intervals",
    "expected_calibration_error",
    "summarize_seed_runs",
    "make_oof_predictions",
    "evaluate_hybrid",
]
