"""Calibration and uncertainty utilities for the Phase 2 MLP.

Owner: Satya Prakash.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch
import torch.nn as nn

from src.metrics import regression_metrics


def mc_dropout_predictions(
    model: nn.Module,
    x: torch.Tensor,
    passes: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Run stochastic forward passes and return mean prediction and sample variance."""
    if passes < 2:
        raise ValueError("passes must be at least 2")
    dropout_present = any(isinstance(module, nn.Dropout) for module in model.modules())
    if not dropout_present:
        raise ValueError("MC dropout requires at least one nn.Dropout module")

    was_training = model.training
    model.train()
    samples = []
    try:
        with torch.no_grad():
            for _ in range(passes):
                samples.append(model(x).detach().cpu().numpy().reshape(-1))
    finally:
        model.train(was_training)
    mean = np.mean(samples, axis=0)
    variance = np.var(samples, axis=0, ddof=1)
    return mean, variance


def prediction_intervals(
    y_true: Iterable[float],
    mean: Iterable[float],
    variance: Iterable[float],
    coverage: float = 0.90,
) -> dict[str, float]:
    """Compute PICP and MPIW for Gaussian-style approximate regression intervals."""
    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must be between 0 and 1")
    from scipy.stats import norm

    y = np.asarray(list(y_true), dtype=np.float64).ravel()
    mu = np.asarray(list(mean), dtype=np.float64).ravel()
    var = np.asarray(list(variance), dtype=np.float64).ravel()
    if not y.shape == mu.shape == var.shape:
        raise ValueError("y_true, mean and variance must have the same shape")

    z = float(norm.ppf(0.5 + coverage / 2.0))
    std = np.sqrt(np.maximum(var, 0.0))
    lower = mu - z * std
    upper = mu + z * std
    return {
        "PICP": float(np.mean((y >= lower) & (y <= upper))),
        "MPIW": float(np.mean(upper - lower)),
    }


def reliability_diagram(
    y_true: Iterable[float],
    y_prob: Iterable[float],
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Return mean confidence and empirical fraction for positive-class probabilities."""
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    y = np.asarray(list(y_true), dtype=np.float64).ravel()
    probs = np.asarray(list(y_prob), dtype=np.float64).ravel()
    if y.shape != probs.shape:
        raise ValueError("y_true and y_prob must have the same shape")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    confidences = np.full(n_bins, np.nan)
    accuracies = np.full(n_bins, np.nan)
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs <= bins[i + 1] if i == n_bins - 1 else probs < bins[i + 1])
        if mask.any():
            confidences[i] = float(probs[mask].mean())
            accuracies[i] = float(y[mask].mean())
    return confidences, accuracies


def expected_calibration_error(
    y_true: Iterable[float],
    y_prob: Iterable[float],
    n_bins: int = 10,
) -> float:
    """Compute binary ECE from reliability bins."""
    y = np.asarray(list(y_true), dtype=np.float64).ravel()
    probs = np.asarray(list(y_prob), dtype=np.float64).ravel()
    if y.shape != probs.shape:
        raise ValueError("y_true and y_prob must have the same shape")
    confidences, accuracies = reliability_diagram(y, probs, n_bins=n_bins)
    valid = ~np.isnan(confidences)
    if not valid.any():
        return 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    weights = []
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (
            probs <= bins[i + 1] if i == n_bins - 1 else probs < bins[i + 1]
        )
        weights.append(float(mask.mean()))
    weights_array = np.asarray(weights)
    return float(np.sum(weights_array[valid] * np.abs(accuracies[valid] - confidences[valid])))


def continuous_model_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float]:
    """Preserve the shared project metric contract for calibration summaries."""
    y = np.asarray(list(y_true), dtype=np.float64).ravel()
    pred = np.asarray(list(y_pred), dtype=np.float64).ravel()
    return regression_metrics(y, pred)
