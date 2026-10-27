"""Metric definitions.   Owner: Satya Yadav.

Pure functions of (y_true, y_pred), unit-tested on dummy data so the numbers in the
report are trustworthy.

  * continuous target  -> MAE, RMSE, R-squared, MAPE, Median Absolute Error
  * defined classes    -> macro F1, per-class F1, accuracy, calibration data

All functions accept array-like inputs (NumPy arrays, pandas Series, Python lists)
and return plain Python dicts so the caller can log them directly to CSV.

Usage
-----
    from src.metrics import regression_metrics, classification_metrics

    # Regression (primary task — CCME_Values)
    metrics = regression_metrics(y_true, y_pred)
    # {'mae': ..., 'rmse': ..., 'r2': ..., 'mape': ..., 'medae': ...}

    # Classification (CCME_WQI category if needed in ablations)
    metrics = classification_metrics(y_true_labels, y_pred_labels)
    # {'accuracy': ..., 'macro_f1': ..., 'per_class_f1': {...}, ...}
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_array(x: Any) -> np.ndarray:
    """Convert any array-like to a flat float64 NumPy array."""
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1-D array, got shape {arr.shape}.")
    return arr


def _validate_lengths(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have the same length; got {len(y_true)} and {len(y_pred)}."
        )
    if len(y_true) == 0:
        raise ValueError("y_true and y_pred must not be empty.")


# ---------------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------------


def regression_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Compute regression metrics for a continuous target.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth CCME_Values (continuous, range ~ 31.3 – 100.0).
    y_pred : array-like of shape (n_samples,)
        Model predictions.

    Returns
    -------
    dict[str, float]
        Keys:
        - ``mae``   : Mean Absolute Error  (lower is better).
        - ``rmse``  : Root Mean Squared Error  (lower is better).
        - ``r2``    : Coefficient of Determination R²  (higher is better, 1.0 = perfect).
        - ``mape``  : Mean Absolute Percentage Error in %  (lower is better).
                      Samples where y_true == 0 are skipped with a warning.
        - ``medae`` : Median Absolute Error  (lower is better; robust to outliers).
    """
    y_true = _to_array(y_true)
    y_pred = _to_array(y_pred)
    _validate_lengths(y_true, y_pred)

    residuals = y_true - y_pred

    # --- MAE ---
    mae = float(np.mean(np.abs(residuals)))

    # --- RMSE ---
    rmse = float(np.sqrt(np.mean(residuals**2)))

    # --- R² ---
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        if ss_res == 0.0:
            r2 = 1.0
        else:
            warnings.warn(
                "R² is undefined when y_true has zero variance (all values identical). "
                "Returning 0.0.",
                RuntimeWarning,
                stacklevel=2,
            )
            r2 = 0.0
    else:
        r2 = float(1.0 - ss_res / ss_tot)

    # --- MAPE (skip zero-valued true labels) ---
    nonzero_mask = y_true != 0.0
    if not nonzero_mask.any():
        warnings.warn(
            "MAPE is undefined when all y_true values are 0. Returning NaN.",
            RuntimeWarning,
            stacklevel=2,
        )
        mape = float("nan")
    else:
        if not nonzero_mask.all():
            warnings.warn(
                f"{(~nonzero_mask).sum()} sample(s) with y_true == 0 were excluded from MAPE.",
                RuntimeWarning,
                stacklevel=2,
            )
        mape = float(np.mean(np.abs(residuals[nonzero_mask] / y_true[nonzero_mask])) * 100.0)

    # --- Median Absolute Error ---
    medae = float(np.median(np.abs(residuals)))

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mape": mape,
        "medae": medae,
    }


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------


def classification_metrics(
    y_true: Any,
    y_pred: Any,
    *,
    labels: list[str] | None = None,
) -> dict[str, Any]:
    """Compute classification metrics for the CCME_WQI category scheme.

    Used only in ablations where the continuous CCME_Values predictions are
    thresholded back into the CCME category scheme, or where a classifier is
    trained directly. The primary task is regression — this function is a
    secondary diagnostic.

    CCME WQI categories (standard thresholds on the 0–100 scale):
        Excellent : 95 – 100
        Good      : 80 – 94
        Fair      : 65 – 79
        Marginal  : 45 – 64
        Poor      :  0 – 44

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth category labels (strings or ints).
    y_pred : array-like of shape (n_samples,)
        Predicted category labels.
    labels : list of str, optional
        Ordered list of class labels. Defaults to the sorted unique values of
        y_true. Providing the canonical order ensures consistent per-class F1
        reporting across train / val / test even when some classes are absent
        from a split.

    Returns
    -------
    dict
        Keys:
        - ``accuracy``    : Overall accuracy (fraction of correct predictions).
        - ``macro_f1``    : Macro-averaged F1 score (unweighted mean across classes).
        - ``weighted_f1`` : Weighted-averaged F1 score (weighted by support).
        - ``per_class_f1``: dict mapping each class label to its F1 score.
        - ``per_class_precision``: dict mapping each class label to precision.
        - ``per_class_recall``   : dict mapping each class label to recall.
        - ``support``     : dict mapping each class label to its sample count in y_true.
        - ``n_samples``   : Total number of samples evaluated.
        - ``n_classes``   : Number of unique classes in labels.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    if y_true_arr.ndim != 1 or y_pred_arr.ndim != 1:
        raise ValueError("y_true and y_pred must be 1-D arrays.")
    _validate_lengths(y_true_arr, y_pred_arr)

    if labels is None:
        labels = sorted(set(y_true_arr.tolist()) | set(y_pred_arr.tolist()), key=str)

    n_samples = len(y_true_arr)
    accuracy = float(np.mean(y_true_arr == y_pred_arr))

    per_class_precision: dict[str, float] = {}
    per_class_recall: dict[str, float] = {}
    per_class_f1: dict[str, float] = {}
    support: dict[str, int] = {}

    for cls in labels:
        cls_key = str(cls)
        tp = int(np.sum((y_true_arr == cls) & (y_pred_arr == cls)))
        fp = int(np.sum((y_true_arr != cls) & (y_pred_arr == cls)))
        fn = int(np.sum((y_true_arr == cls) & (y_pred_arr != cls)))
        sup = int(np.sum(y_true_arr == cls))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        per_class_precision[cls_key] = float(prec)
        per_class_recall[cls_key] = float(rec)
        per_class_f1[cls_key] = float(f1)
        support[cls_key] = sup

    # Macro F1 — unweighted mean across all classes
    macro_f1 = float(np.mean(list(per_class_f1.values())))

    # Weighted F1 — weighted by true support
    total_support = sum(support.values())
    if total_support > 0:
        weighted_f1 = float(
            sum(per_class_f1[str(cls)] * support[str(cls)] for cls in labels) / total_support
        )
    else:
        weighted_f1 = 0.0

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class_f1": per_class_f1,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "support": support,
        "n_samples": n_samples,
        "n_classes": len(labels),
    }


# ---------------------------------------------------------------------------
# Calibration helper (for classification diagnostics)
# ---------------------------------------------------------------------------


def calibration_data(
    y_true_continuous: Any,
    y_pred_continuous: Any,
    *,
    thresholds: dict[str, tuple[float, float]] | None = None,
) -> dict[str, dict[str, float]]:
    """Bin continuous predictions into CCME categories and return calibration data.

    Converts a continuous regression output into the CCME WQI category scheme
    so that the classification quality of the regressor's predictions can be
    assessed without retraining a classifier.

    Parameters
    ----------
    y_true_continuous : array-like
        Ground-truth CCME_Values (0–100 continuous scale).
    y_pred_continuous : array-like
        Regressor predictions on the same scale.
    thresholds : dict, optional
        Custom category boundaries. Defaults to the standard CCME scheme:
        ``{'Poor': (0, 44), 'Marginal': (45, 64), 'Fair': (65, 79),
           'Good': (80, 94), 'Excellent': (95, 100)}``.

    Returns
    -------
    dict
        ``{'classification_metrics': {...}, 'category_labels': ['Poor', ...]}``
        where ``classification_metrics`` is the output of
        :func:`classification_metrics`.
    """
    DEFAULT_THRESHOLDS: dict[str, tuple[float, float]] = {
        "Poor": (0.0, 44.0),
        "Marginal": (45.0, 64.0),
        "Fair": (65.0, 79.0),
        "Good": (80.0, 94.0),
        "Excellent": (95.0, 100.0),
    }
    thresh = thresholds if thresholds is not None else DEFAULT_THRESHOLDS
    ordered_labels = list(thresh.keys())

    y_true_arr = _to_array(y_true_continuous)
    y_pred_arr = _to_array(y_pred_continuous)
    _validate_lengths(y_true_arr, y_pred_arr)

    def _bin(values: np.ndarray) -> np.ndarray:
        out = np.full(len(values), "Unknown", dtype=object)
        for label, (lo, hi) in thresh.items():
            mask = (values >= lo) & (values <= hi)
            out[mask] = label
        return out

    y_true_labels = _bin(y_true_arr)
    y_pred_labels = _bin(y_pred_arr)

    cls_metrics = classification_metrics(y_true_labels, y_pred_labels, labels=ordered_labels)
    return {
        "classification_metrics": cls_metrics,
        "category_labels": ordered_labels,
    }
