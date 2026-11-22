"""Deep learning package for Phase 2 Water Quality Prediction.

Public APIs are loaded lazily to keep lightweight metric and notebook imports
from eagerly initializing PyTorch modules.
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
    "run_architecture_ablation": ("src.dl.architecture", "run_architecture_ablation"),
    "aggregate_ablation": ("src.dl.architecture", "aggregate_ablation"),
    "summarize_seed_runs": ("src.dl.comparison", "summarize_seed_runs"),
    "train_hybrid": ("src.dl.hybrid", "train_hybrid"),
    "run_hybrid_ablation": ("src.dl.hybrid", "run_ablation"),
    "HYBRID_RESULTS_PATH": ("src.dl.hybrid", "HYBRID_RESULTS_PATH"),
    "AGREED_SEEDS": ("src.dl.hybrid", "AGREED_SEEDS"),
    "prediction_interval_coverage": ("src.dl.calibration", "prediction_interval_coverage"),
    "calibrate_conformal_quantile": ("src.dl.calibration", "calibrate_conformal_quantile"),
    "conformal_prediction_intervals": ("src.dl.calibration", "conformal_prediction_intervals"),
    "enable_dropout_only": ("src.dl.calibration", "enable_dropout_only"),
    "monte_carlo_dropout_predict": ("src.dl.calibration", "monte_carlo_dropout_predict"),
    "mc_dropout_prediction_intervals": ("src.dl.calibration", "mc_dropout_prediction_intervals"),
    "permutation_feature_importance_mlp": (
        "src.dl.calibration",
        "permutation_feature_importance_mlp",
    ),
    "reliability_diagram_data": ("src.dl.calibration", "reliability_diagram_data"),
    "expected_calibration_error": ("src.dl.calibration", "expected_calibration_error"),
    "category_calibration_summary": ("src.dl.calibration", "category_calibration_summary"),
    "calibration_summary": ("src.dl.calibration", "calibration_summary"),
    "mc_dropout_predictions": ("src.dl.calibration", "mc_dropout_predictions"),
    "prediction_intervals": ("src.dl.calibration", "prediction_intervals"),
    "continuous_model_metrics": ("src.dl.calibration", "continuous_model_metrics"),
}


def __getattr__(name: str) -> Any:
    """Load public deep learning classes and functions on first access."""
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_IMPORTS[name]
    from importlib import import_module

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


__all__ = list(_LAZY_IMPORTS)
