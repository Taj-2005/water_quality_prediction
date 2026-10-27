"""Unit tests for src/metrics.py.   Owner: Satya Yadav.

All tests run on small synthetic arrays and NEVER touch the real dataset.
CI does not have the 305 MiB raw file, so the tests must be self-contained.

Run:
    pytest tests/test_metrics.py -v
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from src.metrics import (
    calibration_data,
    classification_metrics,
    regression_metrics,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _perfect_preds(
    n: int = 10, lo: float = 31.3, hi: float = 100.0
) -> tuple[np.ndarray, np.ndarray]:
    """y_true == y_pred (perfect regression)."""
    rng = np.random.default_rng(0)
    y = rng.uniform(lo, hi, n)
    return y, y.copy()


# ---------------------------------------------------------------------------
# regression_metrics — correctness
# ---------------------------------------------------------------------------


class TestRegressionMetrics:
    def test_perfect_predictions_give_zero_error(self):
        y_true, y_pred = _perfect_preds()
        m = regression_metrics(y_true, y_pred)
        assert m["mae"] == pytest.approx(0.0, abs=1e-12)
        assert m["rmse"] == pytest.approx(0.0, abs=1e-12)
        assert m["r2"] == pytest.approx(1.0, abs=1e-10)
        assert m["mape"] == pytest.approx(0.0, abs=1e-10)
        assert m["medae"] == pytest.approx(0.0, abs=1e-12)

    def test_constant_baseline_r2_is_zero(self):
        """Predicting the mean always yields R² = 0."""
        rng = np.random.default_rng(1)
        y_true = rng.uniform(40, 100, 50)
        y_pred = np.full_like(y_true, y_true.mean())
        m = regression_metrics(y_true, y_pred)
        assert m["r2"] == pytest.approx(0.0, abs=1e-10)

    def test_worse_than_mean_r2_is_negative(self):
        y_true = np.array([50.0, 60.0, 70.0, 80.0])
        # Predict the wrong direction
        y_pred = np.array([80.0, 70.0, 60.0, 50.0])
        m = regression_metrics(y_true, y_pred)
        assert m["r2"] < 0.0

    def test_mae_manual_computation(self):
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([12.0, 18.0, 33.0])
        # |10-12| + |20-18| + |30-33| = 2 + 2 + 3 = 7 → mean = 7/3
        expected_mae = 7.0 / 3.0
        m = regression_metrics(y_true, y_pred)
        assert m["mae"] == pytest.approx(expected_mae, rel=1e-9)

    def test_rmse_manual_computation(self):
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred = np.array([3.0, 4.0, 0.0])
        # residuals = [-3, -4, 0], squared = [9, 16, 0], mean = 25/3
        expected_rmse = math.sqrt(25.0 / 3.0)
        m = regression_metrics(y_true, y_pred)
        assert m["rmse"] == pytest.approx(expected_rmse, rel=1e-9)

    def test_mape_manual_computation(self):
        y_true = np.array([100.0, 50.0])
        y_pred = np.array([110.0, 40.0])
        # |100-110|/100 * 100 = 10 %, |50-40|/50 * 100 = 20 % → mean = 15 %
        m = regression_metrics(y_true, y_pred)
        assert m["mape"] == pytest.approx(15.0, rel=1e-9)

    def test_mape_skips_zero_true_values(self):
        y_true = np.array([0.0, 100.0])
        y_pred = np.array([5.0, 110.0])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            m = regression_metrics(y_true, y_pred)
            assert any("excluded from MAPE" in str(warning.message) for warning in w)
        # Only the second sample contributes: |100-110|/100 = 10 %
        assert m["mape"] == pytest.approx(10.0, rel=1e-9)

    def test_mape_all_zero_true_returns_nan(self):
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([1.0, 2.0])
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            m = regression_metrics(y_true, y_pred)
        assert math.isnan(m["mape"])

    def test_medae_is_robust_to_outliers(self):
        """Adding a single huge outlier should not shift the median AE much."""
        rng = np.random.default_rng(2)
        y_true = rng.uniform(50, 80, 100)
        y_pred = y_true + rng.normal(0, 0.1, 100)  # small noise
        m_clean = regression_metrics(y_true, y_pred)

        # Add one massive outlier
        y_true_out = np.append(y_true, 75.0)
        y_pred_out = np.append(y_pred, 0.0)
        m_out = regression_metrics(y_true_out, y_pred_out)

        # RMSE blows up, medae stays close
        assert m_out["rmse"] > m_clean["rmse"] * 5
        assert abs(m_out["medae"] - m_clean["medae"]) < 1.0

    def test_returns_dict_with_correct_keys(self):
        y_true, y_pred = _perfect_preds()
        m = regression_metrics(y_true, y_pred)
        assert set(m.keys()) == {"mae", "rmse", "r2", "mape", "medae"}

    def test_all_values_are_floats(self):
        y_true, y_pred = _perfect_preds()
        m = regression_metrics(y_true, y_pred)
        for key, val in m.items():
            assert isinstance(val, float), f"Expected float for {key}, got {type(val)}"

    def test_accepts_python_lists(self):
        m = regression_metrics([70.0, 80.0], [71.0, 79.0])
        assert "mae" in m

    def test_accepts_pandas_series(self):
        import pandas as pd

        y_true = pd.Series([60.0, 70.0, 80.0])
        y_pred = pd.Series([61.0, 69.0, 81.0])
        m = regression_metrics(y_true, y_pred)
        assert m["mae"] == pytest.approx(1.0)

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="same length"):
            regression_metrics([1.0, 2.0], [1.0])

    def test_empty_arrays_raise(self):
        with pytest.raises(ValueError, match="empty"):
            regression_metrics([], [])

    def test_zero_variance_true_warns_and_returns_zero_r2(self):
        y_true = np.array([50.0, 50.0, 50.0])
        y_pred = np.array([51.0, 52.0, 49.0])
        with pytest.warns(RuntimeWarning, match="R²"):
            m = regression_metrics(y_true, y_pred)
        assert m["r2"] == 0.0

    def test_r2_range_real_data(self):
        """R² from a sensible prediction should lie in (0, 1)."""
        rng = np.random.default_rng(3)
        y_true = rng.uniform(31.3, 100.0, 200)
        noise = rng.normal(0, 3.0, 200)
        y_pred = np.clip(y_true + noise, 0, 100)
        m = regression_metrics(y_true, y_pred)
        assert 0.0 < m["r2"] < 1.0


# ---------------------------------------------------------------------------
# classification_metrics — correctness
# ---------------------------------------------------------------------------


class TestClassificationMetrics:
    def test_perfect_classification(self):
        labels = ["Poor", "Fair", "Good"]
        y = np.array(labels)
        m = classification_metrics(y, y)
        assert m["accuracy"] == pytest.approx(1.0)
        assert m["macro_f1"] == pytest.approx(1.0)
        for cls in labels:
            assert m["per_class_f1"][cls] == pytest.approx(1.0)

    def test_all_wrong_gives_zero_f1(self):
        y_true = np.array(["A", "A", "B", "B"])
        y_pred = np.array(["B", "B", "A", "A"])
        m = classification_metrics(y_true, y_pred, labels=["A", "B"])
        assert m["accuracy"] == pytest.approx(0.0)
        assert m["macro_f1"] == pytest.approx(0.0)

    def test_macro_f1_manual(self):
        # y_pred = all "A": A gets all positives but also all false-positives.
        # tp_A=2, fp_A=2, fn_A=0 → prec_A=0.5, rec_A=1.0, f1_A=2/3
        # tp_B=0, fp_B=0, fn_B=2 → prec_B=0,   rec_B=0,   f1_B=0
        # macro_f1 = (2/3 + 0) / 2 = 1/3
        y_true = np.array(["A", "A", "B", "B"])
        y_pred = np.array(["A", "A", "A", "A"])  # class B never predicted
        m = classification_metrics(y_true, y_pred, labels=["A", "B"])
        assert m["macro_f1"] == pytest.approx(1.0 / 3.0, rel=1e-9)

    def test_accuracy_manual(self):
        y_true = np.array(["A", "B", "A", "B"])
        y_pred = np.array(["A", "A", "A", "B"])
        m = classification_metrics(y_true, y_pred)
        assert m["accuracy"] == pytest.approx(3 / 4)

    def test_support_counts(self):
        y_true = np.array(["A", "A", "B"])
        y_pred = np.array(["A", "B", "B"])
        m = classification_metrics(y_true, y_pred, labels=["A", "B"])
        assert m["support"]["A"] == 2
        assert m["support"]["B"] == 1

    def test_returns_correct_keys(self):
        y = np.array(["Good", "Fair"])
        m = classification_metrics(y, y)
        expected_keys = {
            "accuracy",
            "macro_f1",
            "weighted_f1",
            "per_class_f1",
            "per_class_precision",
            "per_class_recall",
            "support",
            "n_samples",
            "n_classes",
        }
        assert set(m.keys()) == expected_keys

    def test_n_samples_and_n_classes(self):
        y = np.array(["Good", "Fair", "Poor"])
        m = classification_metrics(y, y)
        assert m["n_samples"] == 3
        assert m["n_classes"] == 3

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="same length"):
            classification_metrics(["A", "B"], ["A"])

    def test_empty_arrays_raise(self):
        with pytest.raises(ValueError):
            classification_metrics([], [])

    def test_custom_labels_order_preserved(self):
        y_true = np.array(["Good", "Fair"])
        y_pred = np.array(["Good", "Good"])
        labels = ["Poor", "Fair", "Good"]
        m = classification_metrics(y_true, y_pred, labels=labels)
        assert list(m["per_class_f1"].keys()) == ["Poor", "Fair", "Good"]

    def test_weighted_f1_equals_macro_when_balanced(self):
        """Equal class support => weighted F1 == macro F1."""
        y_true = np.array(["A", "A", "B", "B"])
        y_pred = np.array(["A", "B", "B", "A"])
        m = classification_metrics(y_true, y_pred, labels=["A", "B"])
        assert m["weighted_f1"] == pytest.approx(m["macro_f1"], abs=1e-10)


# ---------------------------------------------------------------------------
# calibration_data — smoke + regression-to-classification pipeline
# ---------------------------------------------------------------------------


class TestCalibrationData:
    def test_perfect_regression_gives_perfect_classification(self):
        """If regression predictions are exact, classification is also perfect."""
        # Spread across all CCME categories
        y_true = np.array([20.0, 50.0, 70.0, 85.0, 97.0])
        y_pred = y_true.copy()
        result = calibration_data(y_true, y_pred)
        assert result["classification_metrics"]["macro_f1"] == pytest.approx(1.0)

    def test_returns_expected_keys(self):
        y = np.array([50.0, 75.0, 90.0])
        result = calibration_data(y, y)
        assert "classification_metrics" in result
        assert "category_labels" in result

    def test_default_category_labels_are_ccme_scheme(self):
        y = np.array([50.0])
        result = calibration_data(y, y)
        assert result["category_labels"] == ["Poor", "Marginal", "Fair", "Good", "Excellent"]

    def test_custom_thresholds(self):
        custom = {"Low": (0.0, 50.0), "High": (51.0, 100.0)}
        y_true = np.array([25.0, 75.0])
        y_pred = np.array([30.0, 80.0])
        result = calibration_data(y_true, y_pred, thresholds=custom)
        assert result["category_labels"] == ["Low", "High"]
        assert result["classification_metrics"]["macro_f1"] == pytest.approx(1.0)

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="same length"):
            calibration_data([60.0, 70.0], [60.0])


# ---------------------------------------------------------------------------
# Integration: regression_metrics + calibration_data on synthetic dataset
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_mean_baseline_r2_is_zero_and_mae_matches_std(self):
        """
        A mean predictor achieves R² = 0 and MAE ≈ mean absolute deviation.
        """
        rng = np.random.default_rng(5)
        y_true = rng.uniform(31.3, 100.0, 1000)
        y_pred = np.full_like(y_true, y_true.mean())
        m = regression_metrics(y_true, y_pred)
        assert m["r2"] == pytest.approx(0.0, abs=1e-10)
        expected_mae = float(np.mean(np.abs(y_true - y_true.mean())))
        assert m["mae"] == pytest.approx(expected_mae, rel=1e-9)

    def test_noisy_predictor_metrics_are_in_plausible_range(self):
        rng = np.random.default_rng(6)
        y_true = rng.uniform(31.3, 100.0, 500)
        y_pred = y_true + rng.normal(0, 5, 500)
        m = regression_metrics(y_true, y_pred)
        assert 0 < m["r2"] < 1
        assert m["mae"] > 0
        assert m["rmse"] >= m["mae"]  # RMSE ≥ MAE always
        assert m["medae"] > 0


class TestCeilingSplitMetrics:
    """The target is bounded at 100 with a large point mass there; pooled metrics
    conflate identifying compliance with estimating shortfall."""

    def test_groups_partition_the_records(self):
        from src.metrics import ceiling_split_metrics

        y_true = np.array([100.0, 100.0, 80.0, 60.0, 45.0])
        m = ceiling_split_metrics(y_true, y_true.copy())
        assert m["at_ceiling_n"] + m["below_ceiling_n"] == m["pooled_n"] == 5
        assert m["ceiling_share"] == pytest.approx(0.4)

    def test_pooled_hides_a_group_specific_failure(self):
        from src.metrics import ceiling_split_metrics

        # Perfect on the ceiling, badly wrong below it.
        y_true = np.array([100.0, 100.0, 50.0, 50.0])
        y_pred = np.array([100.0, 100.0, 70.0, 70.0])
        m = ceiling_split_metrics(y_true, y_pred)
        assert m["at_ceiling_mae"] == pytest.approx(0.0)
        assert m["below_ceiling_mae"] == pytest.approx(20.0)
        assert m["pooled_mae"] == pytest.approx(10.0), "pooled averages the failure away"

    def test_r2_is_nan_at_the_ceiling(self):
        """Every true value equals the bound, so total variance is zero and R2 is
        undefined. Reporting nan is correct; reporting 0 or 1 would be an artefact."""
        from src.metrics import ceiling_split_metrics

        m = ceiling_split_metrics(np.array([100.0, 100.0]), np.array([99.0, 100.0]))
        assert np.isnan(m["at_ceiling_r2"])
        assert not np.isnan(m["at_ceiling_mae"])

    def test_shortfall_and_overshoot_diagnostics(self):
        from src.metrics import ceiling_split_metrics

        y_true = np.array([100.0, 100.0, 60.0, 70.0])
        y_pred = np.array([95.0, 97.0, 100.0, 65.0])
        m = ceiling_split_metrics(y_true, y_pred)
        assert m["at_ceiling_mean_shortfall"] == pytest.approx(4.0)
        assert m["below_ceiling_overshoot_rate"] == pytest.approx(0.5)

    def test_empty_group_does_not_raise(self):
        from src.metrics import ceiling_split_metrics

        m = ceiling_split_metrics(np.array([50.0, 60.0]), np.array([55.0, 61.0]))
        assert m["at_ceiling_n"] == 0
        assert np.isnan(m["at_ceiling_mae"])
        assert not np.isnan(m["below_ceiling_mae"])
