"""Common evaluation interface and diagnostic metrics.   Owner: Parthraj Singh Bhati.

Provides a unified, trustworthy evaluation layer for all models:
  - Regression metrics: MAE, MSE, RMSE, R², Explained Variance Score.
  - Residual diagnostics: Median Absolute Error, Max Absolute Error, Residual Mean, Residual Std.
  - Group-stratified evaluations: slicing performance by Country, Waterbody Type, Target Ceiling.
  - Non-parametric bootstrapping: confidence intervals on the fixed test partition.

Usage:
------
    from src.evaluation import evaluate_regression, compute_evaluation_suite
    metrics = evaluate_regression(y_true, y_pred)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _to_array(x: Any) -> np.ndarray:
    """Convert array-like input to a 1D float64 numpy array."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1-D array, got shape {arr.shape}.")
    return arr


def _validate_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Validate shape and non-emptiness of prediction arrays."""
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: len(y_true)={len(y_true)} vs len(y_pred)={len(y_pred)}."
        )
    if len(y_true) == 0:
        raise ValueError("Inputs y_true and y_pred must not be empty.")


def evaluate_regression(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Calculate consistent regression and residual error metrics for any model.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth target values (CCME_Values).
    y_pred : array-like of shape (n_samples,)
        Model predictions.

    Returns
    -------
    dict[str, float]
        Dictionary containing:
        - 'mae': Mean Absolute Error
        - 'mse': Mean Squared Error
        - 'rmse': Root Mean Squared Error
        - 'r2': Coefficient of Determination (R²)
        - 'explained_variance': Explained Variance Score
        - 'medae': Median Absolute Error
        - 'max_error': Maximum Absolute Error
        - 'residual_mean': Mean of residuals (y_true - y_pred)
        - 'residual_std': Standard deviation of residuals
    """
    y_t = _to_array(y_true)
    y_p = _to_array(y_pred)
    _validate_inputs(y_t, y_p)

    residuals = y_t - y_p

    # Standard regression metrics
    mae = float(np.mean(np.abs(residuals)))
    mse = float(np.mean(residuals**2))
    rmse = float(np.sqrt(mse))

    # Total and residual sum of squares
    ss_res = float(np.sum(residuals**2))
    y_true_mean = float(np.mean(y_t))
    ss_tot = float(np.sum((y_t - y_true_mean) ** 2))

    if ss_tot == 0.0:
        r2 = 1.0 if ss_res == 0.0 else 0.0
    else:
        r2 = float(1.0 - (ss_res / ss_tot))

    # Explained variance: 1 - Var(y_true - y_pred) / Var(y_true)
    var_res = float(np.var(residuals))
    var_true = float(np.var(y_t))
    if var_true == 0.0:
        explained_variance = 1.0 if var_res == 0.0 else 0.0
    else:
        explained_variance = float(1.0 - (var_res / var_true))

    # Residual error statistics
    medae = float(np.median(np.abs(residuals)))
    max_error = float(np.max(np.abs(residuals)))
    residual_mean = float(np.mean(residuals))
    residual_std = float(np.std(residuals))

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "explained_variance": explained_variance,
        "medae": medae,
        "max_error": max_error,
        "residual_mean": residual_mean,
        "residual_std": residual_std,
    }


def evaluate_by_group(
    y_true: Any,
    y_pred: Any,
    groups: Any,
    group_name: str = "Group",
    min_samples: int = 1,
) -> pd.DataFrame:
    """Compute regression metrics stratified by a categorical group (e.g., Country, Waterbody Type).

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
    y_pred : array-like of shape (n_samples,)
    groups : array-like of shape (n_samples,)
    group_name : str, default 'Group'
        Column name for group identifiers in the output table.
    min_samples : int, default 1
        Minimum sample size required to compute metrics for a group.

    Returns
    -------
    pd.DataFrame
        Table of metrics per group, sorted descending by sample count.
    """
    y_t = _to_array(y_true)
    y_p = _to_array(y_pred)
    _validate_inputs(y_t, y_p)

    grp_arr = np.asarray(groups)
    if len(grp_arr) != len(y_t):
        raise ValueError(f"groups length ({len(grp_arr)}) does not match targets ({len(y_t)}).")

    unique_groups = pd.unique(grp_arr)
    rows = []

    for grp in unique_groups:
        mask = grp_arr == grp
        n = int(np.sum(mask))
        if n < min_samples:
            continue

        grp_metrics = evaluate_regression(y_t[mask], y_p[mask])
        rows.append(
            {
                group_name: str(grp),
                "n_samples": n,
                "pct_of_total": float(n / len(y_t) * 100.0),
                "mae": grp_metrics["mae"],
                "rmse": grp_metrics["rmse"],
                "r2": grp_metrics["r2"],
                "medae": grp_metrics["medae"],
                "max_error": grp_metrics["max_error"],
                "residual_mean": grp_metrics["residual_mean"],
                "residual_std": grp_metrics["residual_std"],
            }
        )

    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        df_out = df_out.sort_values("n_samples", ascending=False).reset_index(drop=True)
    return df_out


def bootstrap_test_metrics(
    y_true: Any,
    y_pred: Any,
    n_bootstraps: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Compute non-parametric bootstrap confidence intervals on fixed test predictions.

    Does NOT re-split data; samples pairs of (y_true, y_pred) with replacement.

    Parameters
    ----------
    y_true : array-like
    y_pred : array-like
    n_bootstraps : int, default 1000
    ci : float, default 0.95
    seed : int, default 42

    Returns
    -------
    dict[str, dict[str, float]]
        Mapping each metric ('mae', 'rmse', 'r2') to its point estimate,
        lower CI bound, upper CI bound, and bootstrap standard error.
    """
    y_t = _to_array(y_true)
    y_p = _to_array(y_pred)
    _validate_inputs(y_t, y_p)

    rng = np.random.default_rng(seed)
    n = len(y_t)

    point_estimates = evaluate_regression(y_t, y_p)

    alpha_low = ((1.0 - ci) / 2.0) * 100.0
    alpha_high = (1.0 - (1.0 - ci) / 2.0) * 100.0

    boot_mae = np.empty(n_bootstraps, dtype=np.float64)
    boot_rmse = np.empty(n_bootstraps, dtype=np.float64)
    boot_r2 = np.empty(n_bootstraps, dtype=np.float64)

    for i in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        sub_yt = y_t[idx]
        sub_yp = y_p[idx]
        sub_res = sub_yt - sub_yp

        boot_mae[i] = np.mean(np.abs(sub_res))
        boot_rmse[i] = np.sqrt(np.mean(sub_res**2))

        sub_ss_res = np.sum(sub_res**2)
        sub_ss_tot = np.sum((sub_yt - np.mean(sub_yt)) ** 2)
        boot_r2[i] = 1.0 - (sub_ss_res / sub_ss_tot) if sub_ss_tot > 0 else 0.0

    results = {}
    for name, arr, point in [
        ("mae", boot_mae, point_estimates["mae"]),
        ("rmse", boot_rmse, point_estimates["rmse"]),
        ("r2", boot_r2, point_estimates["r2"]),
    ]:
        results[name] = {
            "point_estimate": point,
            "ci_lower": float(np.percentile(arr, alpha_low)),
            "ci_upper": float(np.percentile(arr, alpha_high)),
            "std_error": float(np.std(arr)),
        }

    return results
