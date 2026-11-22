"""Calibration, uncertainty estimation, and explainability for Phase 2 deep learning.

Owner: Satya Yadav.
Part of Phase 2: Water Quality Prediction Deep Learning Extension.

Phase 2 Work Plan Deliverables (Satya):
  1. src/dl/calibration.py: Extend metrics.py with reliability diagrams, Expected
     Calibration Error (ECE), and Prediction Interval Coverage Probability (PICP)
     for continuous regression targets.
  2. Monte Carlo (MC) Dropout uncertainty estimation (Gal & Ghahramani, 2016):
     Multiple stochastic forward passes to estimate predictive mean and epistemic uncertainty.
  3. Distribution-free Conformal Prediction intervals:
     Finite-sample coverage guarantees calibrated on validation residuals (Vovk et al., 2005).
  4. Explainability & Feature Attribution:
     Permutation importance for neural networks to compare against Phase 1 tree SHAP rankings.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.metrics import (
    calibration_summary,
    category_calibration_summary,
    expected_calibration_error,
    regression_metrics,
    reliability_diagram_data,
)


def _to_numpy(x: Any) -> np.ndarray:
    """Convert input to a flat float64 numpy array."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        arr = arr.ravel()
    return arr


# ---------------------------------------------------------------------------
# 1. Prediction Interval Metrics (PICP and MPIW)
# ---------------------------------------------------------------------------


def prediction_interval_coverage(
    y_true: Any,
    y_lower: Any,
    y_upper: Any,
) -> dict[str, float]:
    """Compute Prediction Interval Coverage Probability (PICP) and Mean Prediction Interval Width (MPIW).

    For a target coverage level 1 - alpha (e.g., 95%):
      - PICP measures empirical coverage: fraction of ground-truth values falling inside [lower, upper].
      - MPIW measures efficiency/sharpness: average width of the interval (upper - lower).
      - Interval Score (Gneiting & Raftery, 2007): proper scoring rule combining coverage and sharpness.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth continuous target values.
    y_lower : array-like of shape (n_samples,)
        Lower bound of prediction intervals.
    y_upper : array-like of shape (n_samples,)
        Upper bound of prediction intervals.

    Returns
    -------
    dict[str, float]
        Keys:
        - 'picp': empirical coverage fraction in [0, 1].
        - 'mpiw': mean prediction interval width (index points).
        - 'coverage_error': |picp - target_coverage| (if target known, else raw PICP).
        - 'n_samples': number of evaluated samples.
        - 'n_captured': count of true values within the interval.
        - 'under_coverage_count': count of samples where y_true < y_lower.
        - 'over_coverage_count': count of samples where y_true > y_upper.
    """
    yt = _to_numpy(y_true)
    yl = _to_numpy(y_lower)
    yu = _to_numpy(y_upper)

    if len(yt) != len(yl) or len(yt) != len(yu):
        raise ValueError(
            f"Length mismatch: len(y_true)={len(yt)}, len(y_lower)={len(yl)}, len(y_upper)={len(yu)}."
        )
    if len(yt) == 0:
        raise ValueError("Inputs must not be empty.")

    # Guard against inverted bounds
    lower_bound = np.minimum(yl, yu)
    upper_bound = np.maximum(yl, yu)

    in_interval = (yt >= lower_bound) & (yt <= upper_bound)
    n_captured = int(np.sum(in_interval))
    picp = float(n_captured / len(yt))

    widths = upper_bound - lower_bound
    mpiw = float(np.mean(widths))

    under_count = int(np.sum(yt < lower_bound))
    over_count = int(np.sum(yt > upper_bound))

    return {
        "picp": picp,
        "mpiw": mpiw,
        "n_samples": len(yt),
        "n_captured": n_captured,
        "under_coverage_count": under_count,
        "over_coverage_count": over_count,
    }


# ---------------------------------------------------------------------------
# 2. Conformal Prediction Intervals (Distribution-Free Calibration)
# ---------------------------------------------------------------------------


def calibrate_conformal_quantile(
    val_residuals: Any,
    confidence_level: float = 0.95,
) -> float:
    """Compute the finite-sample calibrated conformity score threshold q_hat.

    Using split conformal prediction (Vovk et al., 2005; Angelopoulos & Bates, 2021):
      q_hat = ceil((n + 1) * (1 - alpha)) / n  empirical quantile of |val_residuals|.

    Guarantees:
      P(Y_test in [Y_hat - q_hat, Y_hat + q_hat]) >= 1 - alpha
    under exchangeability of validation and test residuals.
    """
    res = np.abs(_to_numpy(val_residuals))
    n = len(res)
    if n == 0:
        raise ValueError("val_residuals must not be empty.")
    if not (0.0 < confidence_level < 1.0):
        raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}")

    # Finite sample correction index: ceil((n + 1) * alpha_level) / n
    k = int(np.ceil((n + 1) * confidence_level))
    k = min(max(k, 1), n)  # clamp to valid rank index

    sorted_res = np.sort(res)
    q_hat = float(sorted_res[k - 1])
    return q_hat


def conformal_prediction_intervals(
    y_pred: Any,
    q_hat: float,
    clip_range: tuple[float, float] | None = (0.0, 100.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Construct calibrated prediction intervals [y_pred - q_hat, y_pred + q_hat].

    Parameters
    ----------
    y_pred : array-like of shape (n_samples,)
        Point predictions from any model (XGBoost, RF, or MLP).
    q_hat : float
        Calibrated non-conformity threshold obtained from validation split.
    clip_range : tuple of (min, max), optional
        Physical target bounds (e.g. [0, 100] for CCME_Values).

    Returns
    -------
    tuple of (y_lower, y_upper)
    """
    yp = _to_numpy(y_pred)
    y_lower = yp - q_hat
    y_upper = yp + q_hat

    if clip_range is not None:
        y_lower = np.clip(y_lower, clip_range[0], clip_range[1])
        y_upper = np.clip(y_upper, clip_range[0], clip_range[1])

    return y_lower, y_upper


# ---------------------------------------------------------------------------
# 3. Monte Carlo Dropout Uncertainty (Gal & Ghahramani, 2016)
# ---------------------------------------------------------------------------


def enable_dropout_only(model: nn.Module) -> None:
    """Set dropout layers to training mode while keeping Batch/LayerNorm in eval mode.

    Crucial for Monte Carlo dropout: activating dropout during inference samples
    from the approximate weight posterior, while frozen normalization layers prevent
    mini-batch statistics contamination.
    """
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
            m.train()


def monte_carlo_dropout_predict(
    model: nn.Module,
    x: torch.Tensor | np.ndarray,
    n_passes: int = 30,
    device: str | torch.device = "cpu",
    batch_size: int = 512,
    clip_range: tuple[float, float] | None = (0.0, 100.0),
) -> dict[str, np.ndarray]:
    """Perform stochastic forward passes with dropout enabled at test time.

    Parameters
    ----------
    model : nn.Module
        Trained PyTorch MLP model with dropout layers.
    x : torch.Tensor or np.ndarray of shape (n_samples, n_features)
        Input features.
    n_passes : int, default 30
        Number of stochastic forward passes (K).
    device : str or torch.device, default 'cpu'
        Device for computation.
    batch_size : int, default 512
        Mini-batch size for memory-efficient forward passes.
    clip_range : tuple of (float, float), optional
        Physical target range to clip predictions.

    Returns
    -------
    dict
        - 'mean': empirical mean prediction over passes (n_samples,).
        - 'std': empirical standard deviation representing epistemic uncertainty (n_samples,).
        - 'all_passes': raw predictions matrix of shape (n_passes, n_samples).
    """
    if n_passes < 2:
        raise ValueError(f"n_passes must be at least 2 for uncertainty estimation, got {n_passes}")

    if isinstance(x, np.ndarray):
        x_tensor = torch.from_numpy(x).float()
    else:
        x_tensor = x.float()

    model = model.to(device)
    enable_dropout_only(model)

    n_samples = len(x_tensor)
    all_passes = np.zeros((n_passes, n_samples), dtype=np.float32)

    with torch.no_grad():
        for k in range(n_passes):
            pass_preds = []
            for idx in range(0, n_samples, batch_size):
                batch = x_tensor[idx : idx + batch_size].to(device)
                out = model(batch)
                pass_preds.append(out.detach().cpu().numpy().ravel())
            all_passes[k] = np.concatenate(pass_preds)

    if clip_range is not None:
        all_passes = np.clip(all_passes, clip_range[0], clip_range[1])

    mean_pred = np.mean(all_passes, axis=0)
    std_pred = np.std(all_passes, axis=0, ddof=1)

    return {
        "mean": mean_pred,
        "std": std_pred,
        "all_passes": all_passes,
    }


def mc_dropout_prediction_intervals(
    mean_pred: np.ndarray,
    std_pred: np.ndarray,
    z_score: float = 1.96,
    clip_range: tuple[float, float] | None = (0.0, 100.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Construct parametric Gaussian prediction intervals from MC Dropout moments.

    Interval: [mean - z * std, mean + z * std].
    """
    lower = mean_pred - z_score * std_pred
    upper = mean_pred + z_score * std_pred
    if clip_range is not None:
        lower = np.clip(lower, clip_range[0], clip_range[1])
        upper = np.clip(upper, clip_range[0], clip_range[1])
    return lower, upper


# ---------------------------------------------------------------------------
# 4. Neural Network Explainability: Permutation Feature Importance
# ---------------------------------------------------------------------------


def permutation_feature_importance_mlp(
    model: nn.Module,
    x: np.ndarray | torch.Tensor,
    y: np.ndarray | torch.Tensor,
    feature_names: list[str] | None = None,
    metric_fn: Callable[[np.ndarray, np.ndarray], float] | None = None,
    n_repeats: int = 5,
    random_state: int = 42,
    device: str | torch.device = "cpu",
    batch_size: int = 1024,
) -> pd.DataFrame:
    """Compute model-agnostic Permutation Feature Importance for PyTorch MLP.

    Measures the increase in prediction error (default: MAE) when feature column j
    is randomly shuffled, breaking its relationship with the target.

    Parameters
    ----------
    model : nn.Module
        Trained MLP model.
    x : array-like of shape (n_samples, n_features)
        Evaluation features.
    y : array-like of shape (n_samples,)
        Ground-truth target values.
    feature_names : list of str, optional
        Names of the input features.
    metric_fn : callable, optional
        Function (y_true, y_pred) -> float error. Defaults to MAE.
    n_repeats : int, default 5
        Number of random permutations per feature.
    random_state : int, default 42
        Seed for reproducibility.
    device : str, default 'cpu'
        Execution device.
    batch_size : int, default 1024
        Batch size for forward passes.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['Feature', 'Importance_Mean', 'Importance_Std'],
        sorted descending by Importance_Mean.
    """
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = np.asarray(x, dtype=np.float32)

    y_np = _to_numpy(y)

    if metric_fn is None:

        def eval_metric(yt: np.ndarray, yp: np.ndarray) -> float:
            return float(np.mean(np.abs(yt - yp)))
    else:
        eval_metric = metric_fn

    model = model.to(device)
    model.eval()

    def _predict(features_np: np.ndarray) -> np.ndarray:
        preds = []
        n = len(features_np)
        with torch.no_grad():
            for idx in range(0, n, batch_size):
                b = torch.from_numpy(features_np[idx : idx + batch_size]).float().to(device)
                preds.append(model(b).detach().cpu().numpy().ravel())
        return np.concatenate(preds)

    baseline_preds = _predict(x_np)
    baseline_score = eval_metric(y_np, baseline_preds)

    rng = np.random.default_rng(random_state)
    n_features = x_np.shape[1]
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    importance_records = []
    for col_idx in range(n_features):
        scores = []
        for _ in range(n_repeats):
            x_perm = x_np.copy()
            permuted_col = rng.permutation(x_perm[:, col_idx])
            x_perm[:, col_idx] = permuted_col

            perm_preds = _predict(x_perm)
            perm_score = eval_metric(y_np, perm_preds)
            # Importance = error increase after corrupting feature
            scores.append(perm_score - baseline_score)

        importance_records.append(
            {
                "Feature": feature_names[col_idx],
                "Importance_Mean": float(np.mean(scores)),
                "Importance_Std": float(np.std(scores, ddof=1) if len(scores) > 1 else 0.0),
            }
        )

    df = pd.DataFrame(importance_records).sort_values("Importance_Mean", ascending=False)
    df.reset_index(drop=True, inplace=True)
    return df


def mc_dropout_predictions(
    model: nn.Module,
    x: torch.Tensor,
    passes: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Backward-compatible MC-dropout API returning predictive mean and variance."""
    if passes < 2:
        raise ValueError("passes must be at least 2")
    if not any(isinstance(module, nn.Dropout) for module in model.modules()):
        raise ValueError("MC dropout requires at least one nn.Dropout module")

    was_training = model.training
    samples: list[np.ndarray] = []
    try:
        enable_dropout_only(model)
        with torch.no_grad():
            for _ in range(passes):
                samples.append(model(x).detach().cpu().numpy().reshape(-1))
    finally:
        model.train(was_training)
    values = np.stack(samples)
    return values.mean(axis=0), values.var(axis=0, ddof=1)


def prediction_intervals(
    y_true: Iterable[float],
    mean: Iterable[float],
    variance: Iterable[float],
    coverage: float = 0.90,
) -> dict[str, float]:
    """Backward-compatible Gaussian-style interval coverage and width."""
    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must be between 0 and 1")
    from scipy.stats import norm

    y = _to_numpy(y_true)
    mu = _to_numpy(mean)
    var = _to_numpy(variance)
    if not y.shape == mu.shape == var.shape:
        raise ValueError("y_true, mean and variance must have the same shape")
    z = float(norm.ppf(0.5 + coverage / 2.0))
    half_width = z * np.sqrt(np.maximum(var, 0.0))
    lower, upper = mu - half_width, mu + half_width
    return {
        "PICP": float(np.mean((y >= lower) & (y <= upper))),
        "MPIW": float(np.mean(upper - lower)),
    }


def continuous_model_metrics(
    y_true: Iterable[float], y_pred: Iterable[float]
) -> dict[str, float]:
    """Return the shared project regression metrics for iterable inputs."""
    y = _to_numpy(y_true)
    pred = _to_numpy(y_pred)
    return regression_metrics(y, pred)


__all__ = [
    "prediction_interval_coverage",
    "calibrate_conformal_quantile",
    "conformal_prediction_intervals",
    "enable_dropout_only",
    "monte_carlo_dropout_predict",
    "mc_dropout_prediction_intervals",
    "permutation_feature_importance_mlp",
    "reliability_diagram_data",
    "expected_calibration_error",
    "category_calibration_summary",
    "calibration_summary",
    "mc_dropout_predictions",
    "prediction_intervals",
    "continuous_model_metrics",
]
