"""Metric definitions.   Owner: Satya Yadav.

Pure functions of (y_true, y_pred), unit-tested on dummy data so the numbers in the
report are trustworthy.

  * continuous target  -> MAE, RMSE, R-squared, MAPE, Median Absolute Error
  * defined classes    -> macro F1, per-class F1, accuracy, calibration data
  * calibration suite  -> reliability_diagram_data, expected_calibration_error,
                          calibration_summary (Phase 2 — covers all models)

All functions accept array-like inputs (NumPy arrays, pandas Series, Python lists)
and return plain Python dicts so the caller can log them directly to CSV.

Usage
-----
    from src.metrics import regression_metrics, classification_metrics
    from src.metrics import reliability_diagram_data, expected_calibration_error
    from src.metrics import calibration_summary

    # Regression (primary task — CCME_Values)
    metrics = regression_metrics(y_true, y_pred)
    # {'mae': ..., 'rmse': ..., 'r2': ..., 'mape': ..., 'medae': ...}

    # Classification (CCME_WQI category if needed in ablations)
    metrics = classification_metrics(y_true_labels, y_pred_labels)
    # {'accuracy': ..., 'macro_f1': ..., 'per_class_f1': {...}, ...}

    # Calibration (Phase 2 — reliability diagram + ECE for all models)
    rd = reliability_diagram_data(y_true, y_pred, n_bins=10)
    ece = expected_calibration_error(y_true, y_pred, n_bins=10)
    summary = calibration_summary(y_true, y_pred)
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


CEILING_VALUE = 100.0
"""The CCME index is bounded above at 100: a record meeting every guideline gives
F1 = F2 = F3 = 0 and therefore exactly 100. 35.92% of records sit on that bound."""


def ceiling_split_metrics(
    y_true: Any,
    y_pred: Any,
    ceiling: float = CEILING_VALUE,
    tol: float = 1e-9,
) -> dict[str, float]:
    """Report regression metrics separately for records at the ceiling and below it.

    The target is bounded with a large point mass at ``ceiling`` -- not censored: no
    latent value exists above the bound, so the remedy is stratified reporting rather
    than a censored-regression estimator. A single pooled statistic answers two
    questions at once: whether a record is fully guideline-compliant, and how far from
    compliance the remainder fall. Those are different tasks and a model can be good at
    one and poor at the other.

    Parameters
    ----------
    y_true, y_pred : array-like of shape (n_samples,)
        Ground truth and predictions.
    ceiling : float
        The upper bound of the index. Defaults to 100.0.
    tol : float
        Absolute tolerance for treating a value as sitting on the bound, which guards
        against float round-trips through CSV.

    Returns
    -------
    dict[str, float]
        ``pooled_*`` over all records, ``at_ceiling_*`` over records whose true value
        is the bound, and ``below_ceiling_*`` over the rest. Each group contributes
        mae, rmse, r2 and n. Groups with no records yield ``float("nan")`` rather than
        raising, so the function is safe on any subset.

        Also returns ``ceiling_share`` (fraction of records at the bound),
        ``at_ceiling_mean_shortfall`` (mean predicted distance below the bound for
        records that belong on it -- a model that systematically under-predicts
        compliance shows up here), and ``below_ceiling_overshoot_rate`` (fraction of
        non-ceiling records predicted at or above the bound).
    """
    y_true = _to_array(y_true)
    y_pred = _to_array(y_pred)
    _validate_lengths(y_true, y_pred)

    at = np.abs(y_true - ceiling) <= tol
    below = ~at

    out: dict[str, float] = {"ceiling_share": float(at.mean()) if len(at) else float("nan")}

    for name, mask in (
        ("pooled", np.ones_like(at, dtype=bool)),
        ("at_ceiling", at),
        ("below_ceiling", below),
    ):
        n = int(mask.sum())
        out[f"{name}_n"] = float(n)
        if n == 0:
            out[f"{name}_mae"] = out[f"{name}_rmse"] = out[f"{name}_r2"] = float("nan")
            continue
        yt, yp = y_true[mask], y_pred[mask]
        resid = yt - yp
        out[f"{name}_mae"] = float(np.mean(np.abs(resid)))
        out[f"{name}_rmse"] = float(np.sqrt(np.mean(resid**2)))
        # R2 is undefined when the group has no variance, which is exactly the case for
        # the at-ceiling group: every true value is the bound. Report nan rather than a
        # divide-by-zero artefact, and rely on MAE/RMSE for that group.
        ss_tot = float(np.sum((yt - yt.mean()) ** 2))
        out[f"{name}_r2"] = float(1.0 - np.sum(resid**2) / ss_tot) if ss_tot > 0 else float("nan")

    out["at_ceiling_mean_shortfall"] = (
        float(np.mean(ceiling - y_pred[at])) if at.any() else float("nan")
    )
    out["below_ceiling_overshoot_rate"] = (
        float(np.mean(y_pred[below] >= ceiling - tol)) if below.any() else float("nan")
    )
    return out


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


# ---------------------------------------------------------------------------
# Calibration suite — Phase 2.  Owner: Satya Yadav.
#
# Context
# -------
# CCME_Values is a continuous index in [31.3, 100]. The five CCME WQI categories
# (Poor / Marginal / Fair / Good / Excellent) partition that range. A model's
# *regression* error (MAE, RMSE) tells us how far off it is in index points, but
# it does not tell us whether the model correctly identifies the *category* a
# sample belongs to, nor how confident (in the fraction-of-time sense) it is
# about that decision.
#
# The calibration suite answers: for every interval of predicted index values,
# what fraction of true values actually land in that interval?  A perfectly
# calibrated regressor would have this fraction equal to the interval width.
# We operationalise this using the same framework as reliability diagrams for
# classifiers (DeGroot & Fienberg 1983; Niculescu-Mizil & Caruana 2005):
#
#   • Split the prediction axis into equal-width bins.
#   • For each bin, record the mean predicted value and the mean true value.
#   • A perfect calibration curve lies on the y = x diagonal.
#   • ECE (Expected Calibration Error) is the bin-size-weighted mean absolute
#     deviation from the diagonal — one number summarising overall calibration.
#   • MCE (Maximum Calibration Error) is the worst-case bin deviation.
#
# For a *regression* target we adapt the standard classification reliability
# diagram: instead of treating confidence as the probability of the positive
# class, we treat the predicted index value as the "confidence" and the true
# index value as the "outcome", both normalised to [0, 1] so the ECE is in
# the same units regardless of model.
#
# We also provide a CCME-category–level calibration view: for each of the five
# categories we compute the fraction of samples whose predicted category matches
# their true category (category accuracy), mirroring the per-class reliability
# concept.
# ---------------------------------------------------------------------------

CCME_INDEX_RANGE: tuple[float, float] = (0.0, 100.0)
"""Full range of the CCME index used for normalisation in calibration metrics."""


def reliability_diagram_data(
    y_true: Any,
    y_pred: Any,
    *,
    n_bins: int = 10,
    value_range: tuple[float, float] = CCME_INDEX_RANGE,
) -> dict[str, Any]:
    """Compute reliability diagram data for continuous regression predictions.

    Splits the prediction axis into ``n_bins`` equal-width bins, then for each
    populated bin records the mean predicted value and the mean true value.  On a
    perfectly calibrated model the two agree, so the curve lies on the diagonal.

    Both axes are expressed in the original CCME index scale (not normalised to
    [0, 1]) so the numbers in the diagram are directly interpretable.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth CCME_Values.
    y_pred : array-like of shape (n_samples,)
        Model predictions (clipped to ``value_range`` before binning).
    n_bins : int, default 10
        Number of equal-width bins covering ``value_range``.
    value_range : tuple[float, float], default (0.0, 100.0)
        (min, max) of the index used to define bin edges.

    Returns
    -------
    dict
        Keys:

        - ``bin_edges``        : list of float — the n_bins + 1 bin boundary values.
        - ``bin_centers``      : list of float — midpoint of each bin.
        - ``mean_predicted``   : list of float | None — mean predicted value in each
          bin; ``None`` for empty bins.
        - ``mean_true``        : list of float | None — mean true value in each bin;
          ``None`` for empty bins.
        - ``bin_counts``       : list of int — number of samples in each bin.
        - ``n_bins``           : int — number of bins requested.
        - ``n_populated_bins`` : int — number of bins with at least one sample.
        - ``calibration_gaps`` : list of float | None — ``mean_predicted - mean_true``
          for each bin; positive means over-prediction, negative means
          under-prediction, ``None`` for empty bins.

    Notes
    -----
    Predictions outside ``value_range`` are clipped before binning.  Empty bins
    appear as ``None`` rather than ``nan`` so callers can distinguish them from
    numerically invalid results.
    """
    y_true = _to_array(y_true)
    y_pred = _to_array(y_pred)
    _validate_lengths(y_true, y_pred)
    if n_bins < 1:
        raise ValueError(f"n_bins must be at least 1; got {n_bins}.")

    lo, hi = float(value_range[0]), float(value_range[1])
    if lo >= hi:
        raise ValueError(f"value_range must satisfy min < max; got {value_range}.")

    # Clip predictions into the valid range before binning
    y_pred_clipped = np.clip(y_pred, lo, hi)

    edges = np.linspace(lo, hi, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    mean_predicted: list[float | None] = []
    mean_true: list[float | None] = []
    bin_counts: list[int] = []
    gaps: list[float | None] = []

    for i in range(n_bins):
        lo_b, hi_b = edges[i], edges[i + 1]
        # Include the right edge only in the last bin to avoid gaps
        if i < n_bins - 1:
            mask = (y_pred_clipped >= lo_b) & (y_pred_clipped < hi_b)
        else:
            mask = (y_pred_clipped >= lo_b) & (y_pred_clipped <= hi_b)

        n = int(mask.sum())
        bin_counts.append(n)

        if n == 0:
            mean_predicted.append(None)
            mean_true.append(None)
            gaps.append(None)
        else:
            mp = float(np.mean(y_pred_clipped[mask]))
            mt = float(np.mean(y_true[mask]))
            mean_predicted.append(mp)
            mean_true.append(mt)
            gaps.append(mp - mt)

    n_populated = sum(1 for c in bin_counts if c > 0)

    return {
        "bin_edges": edges.tolist(),
        "bin_centers": centers.tolist(),
        "mean_predicted": mean_predicted,
        "mean_true": mean_true,
        "bin_counts": bin_counts,
        "n_bins": n_bins,
        "n_populated_bins": n_populated,
        "calibration_gaps": gaps,
    }


def expected_calibration_error(
    y_true: Any,
    y_pred: Any,
    *,
    n_bins: int = 10,
    value_range: tuple[float, float] = CCME_INDEX_RANGE,
    normalise: bool = True,
) -> dict[str, float]:
    """Compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).

    ECE is the sample-size-weighted mean of |mean_predicted - mean_true| across
    populated bins.  MCE is the maximum such gap over populated bins.  Both
    measure how well the model's predicted index values track the actual index
    values across the prediction range.

    When ``normalise=True`` (the default) both are expressed as a fraction of the
    index range so they are comparable across different value ranges and models.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth CCME_Values.
    y_pred : array-like of shape (n_samples,)
        Model predictions.
    n_bins : int, default 10
        Number of equal-width bins.
    value_range : tuple[float, float], default (0.0, 100.0)
        (min, max) of the index used to define bin edges and for normalisation.
    normalise : bool, default True
        If True, divide ECE and MCE by the index range (hi - lo) so they are
        dimensionless fractions in [0, 1].  If False, return raw index-point units.

    Returns
    -------
    dict[str, float]
        Keys:

        - ``ece``              : Expected Calibration Error.
        - ``mce``              : Maximum Calibration Error.
        - ``mean_gap``         : Unweighted mean calibration gap across populated bins.
        - ``n_bins``           : Number of bins used.
        - ``n_populated_bins`` : Number of bins with at least one sample.
        - ``n_samples``        : Total number of samples.
        - ``normalised``       : Whether ECE and MCE are normalised by range.
    """
    y_true = _to_array(y_true)
    y_pred = _to_array(y_pred)
    _validate_lengths(y_true, y_pred)

    rd = reliability_diagram_data(
        y_true, y_pred, n_bins=n_bins, value_range=value_range
    )

    lo, hi = float(value_range[0]), float(value_range[1])
    index_range = hi - lo  # 100.0 by default

    n_total = len(y_true)
    populated_abs_gaps: list[float] = []
    weighted_gaps: float = 0.0

    for i in range(n_bins):
        gap = rd["calibration_gaps"][i]
        if gap is None:
            continue
        n_bin = rd["bin_counts"][i]
        abs_gap = abs(gap)
        populated_abs_gaps.append(abs_gap)
        weighted_gaps += (n_bin / n_total) * abs_gap

    if not populated_abs_gaps:
        ece = float("nan")
        mce = float("nan")
        mean_gap = float("nan")
    else:
        ece = weighted_gaps
        mce = float(max(populated_abs_gaps))
        mean_gap = float(np.mean(populated_abs_gaps))

    if normalise and index_range > 0:
        ece = ece / index_range if not (ece != ece) else float("nan")  # nan check
        mce = mce / index_range if not (mce != mce) else float("nan")
        mean_gap = mean_gap / index_range if not (mean_gap != mean_gap) else float("nan")

    return {
        "ece": ece,
        "mce": mce,
        "mean_gap": mean_gap,
        "n_bins": n_bins,
        "n_populated_bins": rd["n_populated_bins"],
        "n_samples": n_total,
        "normalised": normalise,
    }


def category_calibration_summary(
    y_true: Any,
    y_pred: Any,
    *,
    thresholds: dict[str, tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """CCME-category–level calibration summary.

    For each CCME WQI category (Poor / Marginal / Fair / Good / Excellent) this
    function reports:
      - The number of true samples in the category.
      - The number of predicted samples assigned to the category.
      - The fraction of true samples correctly classified into their category
        (per-category recall / hit rate).
      - The mean regression error (MAE) restricted to true samples in the category.
      - The mean predicted index for samples whose true value is in the category.

    This provides a richer picture than a single macro-F1: a model might be
    well-calibrated on Excellent water (the dominant class, 35.9 % at ceiling)
    but systematically misclassify Poor water, which is the operationally
    critical failure mode.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth CCME_Values (continuous, 0–100 scale).
    y_pred : array-like of shape (n_samples,)
        Model predictions on the same scale.
    thresholds : dict, optional
        Custom category boundaries.  Defaults to the standard five-class CCME scheme.

    Returns
    -------
    dict
        ``{'categories': { <name>: { 'true_n', 'pred_n', 'hit_rate',
           'mae', 'mean_pred', 'mean_true' } }, 'ordered_labels': [...],
           'overall_category_accuracy': float}``
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

    y_true_arr = _to_array(y_true)
    y_pred_arr = _to_array(y_pred)
    _validate_lengths(y_true_arr, y_pred_arr)

    def _bin(values: np.ndarray) -> np.ndarray:
        out = np.full(len(values), "Unknown", dtype=object)
        for label, (lo, hi) in thresh.items():
            out[(values >= lo) & (values <= hi)] = label
        return out

    true_cats = _bin(y_true_arr)
    pred_cats = _bin(y_pred_arr)

    total_correct = int(np.sum(true_cats == pred_cats))
    overall_accuracy = float(total_correct / len(y_true_arr))

    categories: dict[str, dict[str, float | int]] = {}
    for label in ordered_labels:
        true_mask = true_cats == label
        pred_mask = pred_cats == label

        true_n = int(true_mask.sum())
        pred_n = int(pred_mask.sum())

        if true_n > 0:
            correct_in_cat = int(np.sum((true_cats == label) & (pred_cats == label)))
            hit_rate = float(correct_in_cat / true_n)
            mae_cat = float(np.mean(np.abs(y_true_arr[true_mask] - y_pred_arr[true_mask])))
            mean_pred = float(np.mean(y_pred_arr[true_mask]))
            mean_true = float(np.mean(y_true_arr[true_mask]))
        else:
            hit_rate = float("nan")
            mae_cat = float("nan")
            mean_pred = float("nan")
            mean_true = float("nan")

        categories[label] = {
            "true_n": true_n,
            "pred_n": pred_n,
            "hit_rate": hit_rate,
            "mae": mae_cat,
            "mean_pred": mean_pred,
            "mean_true": mean_true,
        }

    return {
        "categories": categories,
        "ordered_labels": ordered_labels,
        "overall_category_accuracy": overall_accuracy,
    }


def calibration_summary(
    y_true: Any,
    y_pred: Any,
    *,
    n_bins: int = 10,
    value_range: tuple[float, float] = CCME_INDEX_RANGE,
    thresholds: dict[str, tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """Full calibration summary for a single model's predictions.

    Convenience wrapper that runs the complete Phase 2 calibration suite in one
    call and returns a nested dict suitable for logging to a CSV or for display
    in a notebook.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth CCME_Values.
    y_pred : array-like of shape (n_samples,)
        Model predictions.
    n_bins : int, default 10
        Number of equal-width bins for the reliability diagram and ECE.
    value_range : tuple[float, float], default (0.0, 100.0)
        Index range for bin definition and ECE normalisation.
    thresholds : dict, optional
        Custom CCME category boundaries.

    Returns
    -------
    dict
        ``{'ece_metrics': {...}, 'reliability_diagram': {...},
           'category_calibration': {...}, 'regression_metrics': {...}}``

        All sub-dicts are the outputs of their respective functions.
    """
    y_true = _to_array(y_true)
    y_pred = _to_array(y_pred)
    _validate_lengths(y_true, y_pred)

    return {
        "regression_metrics": regression_metrics(y_true, y_pred),
        "ece_metrics": expected_calibration_error(
            y_true, y_pred, n_bins=n_bins, value_range=value_range
        ),
        "reliability_diagram": reliability_diagram_data(
            y_true, y_pred, n_bins=n_bins, value_range=value_range
        ),
        "category_calibration": category_calibration_summary(
            y_true, y_pred, thresholds=thresholds
        ),
    }
