"""Unit tests for the Phase 2 calibration functions in src/metrics.py.

Owner: Satya Yadav.

Covers:
  - reliability_diagram_data
  - expected_calibration_error
  - category_calibration_summary
  - calibration_summary

All tests run on small synthetic arrays and NEVER touch the real dataset.
CI does not have the 305 MiB raw file, so the tests must be self-contained.

Run:
    pytest tests/test_calibration.py -v
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.metrics import (
    calibration_summary,
    category_calibration_summary,
    expected_calibration_error,
    reliability_diagram_data,
)

# ---------------------------------------------------------------------------
# Shared synthetic data helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)


def _ccme_range_data(n: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Uniform y_true in [31.3, 100] with small Gaussian noise for y_pred."""
    y_true = RNG.uniform(31.3, 100.0, n)
    y_pred = y_true + RNG.normal(0, 2.0, n)
    y_pred = np.clip(y_pred, 0.0, 100.0)
    return y_true, y_pred


def _perfect_data(n: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """y_true == y_pred."""
    y = RNG.uniform(0.0, 100.0, n)
    return y, y.copy()


def _constant_pred(n: int = 100, value: float = 85.0) -> tuple[np.ndarray, np.ndarray]:
    """y_pred is a single constant (mean-baseline style)."""
    y_true = RNG.uniform(31.3, 100.0, n)
    y_pred = np.full(n, value)
    return y_true, y_pred


# ===========================================================================
# TestReliabilityDiagramData
# ===========================================================================


class TestReliabilityDiagramData:
    def test_returns_correct_keys(self):
        y_true, y_pred = _ccme_range_data()
        rd = reliability_diagram_data(y_true, y_pred)
        for key in (
            "bin_edges",
            "bin_centers",
            "mean_predicted",
            "mean_true",
            "bin_counts",
            "n_bins",
            "n_populated_bins",
            "calibration_gaps",
        ):
            assert key in rd, f"Missing key: {key}"

    def test_default_produces_10_bins(self):
        y_true, y_pred = _ccme_range_data()
        rd = reliability_diagram_data(y_true, y_pred)
        assert rd["n_bins"] == 10
        assert len(rd["bin_edges"]) == 11
        assert len(rd["bin_centers"]) == 10
        assert len(rd["mean_predicted"]) == 10
        assert len(rd["mean_true"]) == 10
        assert len(rd["bin_counts"]) == 10
        assert len(rd["calibration_gaps"]) == 10

    def test_custom_n_bins(self):
        y_true, y_pred = _ccme_range_data()
        for n in (5, 20, 1):
            rd = reliability_diagram_data(y_true, y_pred, n_bins=n)
            assert rd["n_bins"] == n
            assert len(rd["bin_centers"]) == n

    def test_counts_sum_to_n_samples(self):
        y_true, y_pred = _ccme_range_data(300)
        rd = reliability_diagram_data(y_true, y_pred, n_bins=10)
        assert sum(rd["bin_counts"]) == len(y_true)

    def test_perfect_model_gaps_near_zero(self):
        y_true, y_pred = _perfect_data(500)
        rd = reliability_diagram_data(y_true, y_pred, n_bins=10)
        for gap in rd["calibration_gaps"]:
            if gap is not None:
                assert abs(gap) < 1e-10, f"Gap should be 0 for perfect predictor: {gap}"

    def test_none_returned_for_empty_bins(self):
        # Predictions concentrated in [80, 100]; bins [0, 50] will be empty
        y_true = np.array([85.0, 90.0, 95.0, 100.0])
        y_pred = np.array([86.0, 91.0, 96.0, 99.0])
        rd = reliability_diagram_data(y_true, y_pred, n_bins=10)
        # Bins 0-4 (covering [0, 50]) should be empty
        for i in range(5):
            assert rd["mean_predicted"][i] is None
            assert rd["mean_true"][i] is None
            assert rd["calibration_gaps"][i] is None

    def test_bin_edges_cover_full_range(self):
        y_true, y_pred = _ccme_range_data()
        rd = reliability_diagram_data(y_true, y_pred, value_range=(0.0, 100.0))
        assert rd["bin_edges"][0] == pytest.approx(0.0)
        assert rd["bin_edges"][-1] == pytest.approx(100.0)

    def test_custom_value_range(self):
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([11.0, 19.0, 31.0])
        rd = reliability_diagram_data(y_true, y_pred, n_bins=3, value_range=(0.0, 40.0))
        assert rd["bin_edges"][0] == pytest.approx(0.0)
        assert rd["bin_edges"][-1] == pytest.approx(40.0)

    def test_out_of_range_predictions_are_clipped(self):
        # y_pred has values outside [0, 100] — should not raise
        y_true = np.array([50.0, 60.0, 70.0])
        y_pred = np.array([-5.0, 110.0, 70.0])
        rd = reliability_diagram_data(y_true, y_pred, n_bins=5)
        assert sum(rd["bin_counts"]) == 3

    def test_invalid_n_bins_raises(self):
        y_true, y_pred = _ccme_range_data(10)
        with pytest.raises(ValueError, match="n_bins"):
            reliability_diagram_data(y_true, y_pred, n_bins=0)

    def test_invalid_value_range_raises(self):
        y_true, y_pred = _ccme_range_data(10)
        with pytest.raises(ValueError, match="value_range"):
            reliability_diagram_data(y_true, y_pred, value_range=(100.0, 0.0))

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="same length"):
            reliability_diagram_data([50.0, 60.0], [50.0])

    def test_n_populated_bins_counts_nonempty(self):
        # All predictions in one bin
        y_true = np.array([85.0] * 10)
        y_pred = np.array([85.0] * 10)
        rd = reliability_diagram_data(y_true, y_pred, n_bins=10)
        assert rd["n_populated_bins"] == 1

    def test_gap_direction(self):
        """Over-prediction gives positive gaps; under-prediction gives negative."""
        # All predictions 10 points above true values → positive gap
        y_true = np.array([50.0, 60.0, 70.0, 80.0])
        y_pred = y_true + 10.0
        y_pred = np.clip(y_pred, 0.0, 100.0)
        rd = reliability_diagram_data(y_true, y_pred, n_bins=5)
        populated_gaps = [g for g in rd["calibration_gaps"] if g is not None]
        for gap in populated_gaps:
            assert gap > 0, "Over-predicting by 10 should give positive gap"


# ===========================================================================
# TestExpectedCalibrationError
# ===========================================================================


class TestExpectedCalibrationError:
    def test_returns_correct_keys(self):
        y_true, y_pred = _ccme_range_data()
        result = expected_calibration_error(y_true, y_pred)
        for key in (
            "ece",
            "mce",
            "mean_gap",
            "n_bins",
            "n_populated_bins",
            "n_samples",
            "normalised",
        ):
            assert key in result

    def test_perfect_model_gives_zero_ece(self):
        y_true, y_pred = _perfect_data(500)
        result = expected_calibration_error(y_true, y_pred)
        assert result["ece"] == pytest.approx(0.0, abs=1e-10)
        assert result["mce"] == pytest.approx(0.0, abs=1e-10)

    def test_ece_is_non_negative(self):
        y_true, y_pred = _ccme_range_data(300)
        result = expected_calibration_error(y_true, y_pred)
        assert result["ece"] >= 0.0
        assert result["mce"] >= 0.0

    def test_mce_geq_ece(self):
        """MCE is the worst bin; ECE is weighted average — MCE ≥ ECE always."""
        y_true, y_pred = _ccme_range_data(500)
        result = expected_calibration_error(y_true, y_pred)
        assert result["mce"] >= result["ece"] - 1e-12

    def test_constant_predictor_has_large_ece(self):
        """A constant predictor is very poorly calibrated across the full range."""
        y_true, y_pred = _constant_pred(500, value=85.0)
        result = expected_calibration_error(y_true, y_pred)
        # ECE should be meaningfully above 0 (constant pred is miscalibrated)
        assert result["ece"] > 0.01

    def test_normalised_flag_true_gives_fraction(self):
        y_true, y_pred = _ccme_range_data(200)
        r = expected_calibration_error(y_true, y_pred, normalise=True, value_range=(0.0, 100.0))
        assert r["normalised"] is True
        # ECE normalised by 100 must be in [0, 1]
        if not math.isnan(r["ece"]):
            assert 0.0 <= r["ece"] <= 1.0

    def test_normalised_flag_false_gives_index_points(self):
        y_true, y_pred = _ccme_range_data(200)
        r_norm = expected_calibration_error(
            y_true, y_pred, normalise=True, value_range=(0.0, 100.0)
        )
        r_raw = expected_calibration_error(
            y_true, y_pred, normalise=False, value_range=(0.0, 100.0)
        )
        assert r_raw["normalised"] is False
        if not (math.isnan(r_norm["ece"]) or math.isnan(r_raw["ece"])):
            assert r_raw["ece"] == pytest.approx(r_norm["ece"] * 100.0, rel=1e-9)

    def test_n_samples_matches_input_length(self):
        n = 150
        y_true, y_pred = _ccme_range_data(n)
        result = expected_calibration_error(y_true, y_pred)
        assert result["n_samples"] == n

    def test_custom_n_bins(self):
        y_true, y_pred = _ccme_range_data(300)
        for n in (5, 20):
            result = expected_calibration_error(y_true, y_pred, n_bins=n)
            assert result["n_bins"] == n

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="same length"):
            expected_calibration_error([50.0, 60.0], [50.0])

    def test_ece_manual_single_bin(self):
        """With n_bins=1 the ECE equals |mean_pred - mean_true| / range."""
        y_true = np.array([40.0, 60.0, 80.0])  # mean = 60
        y_pred = np.array([50.0, 70.0, 90.0])  # mean = 70; gap = 10
        result = expected_calibration_error(
            y_true, y_pred, n_bins=1, normalise=True, value_range=(0.0, 100.0)
        )
        expected_ece = 10.0 / 100.0  # = 0.1
        assert result["ece"] == pytest.approx(expected_ece, rel=1e-9)


# ===========================================================================
# TestCategoryCalibrationSummary
# ===========================================================================


class TestCategoryCalibrationSummary:
    def test_returns_correct_top_level_keys(self):
        y_true, y_pred = _ccme_range_data()
        result = category_calibration_summary(y_true, y_pred)
        assert "categories" in result
        assert "ordered_labels" in result
        assert "overall_category_accuracy" in result

    def test_default_ccme_category_labels(self):
        y_true, y_pred = _perfect_data()
        result = category_calibration_summary(y_true, y_pred)
        assert result["ordered_labels"] == ["Poor", "Marginal", "Fair", "Good", "Excellent"]

    def test_per_category_keys(self):
        y_true, y_pred = _ccme_range_data()
        result = category_calibration_summary(y_true, y_pred)
        for cat in result["ordered_labels"]:
            d = result["categories"][cat]
            for key in ("true_n", "pred_n", "hit_rate", "mae", "mean_pred", "mean_true"):
                assert key in d, f"Missing key '{key}' for category '{cat}'"

    def test_true_n_sums_to_n_samples(self):
        y_true, y_pred = _ccme_range_data(400)
        result = category_calibration_summary(y_true, y_pred)
        total = sum(d["true_n"] for d in result["categories"].values())
        # Allow for "Unknown" (values falling exactly on category boundaries)
        assert total <= len(y_true)

    def test_perfect_model_hit_rate_is_one(self):
        """A perfect regressor classifies every sample correctly."""
        y_true = np.array([20.0, 50.0, 70.0, 85.0, 97.0])
        y_pred = y_true.copy()
        result = category_calibration_summary(y_true, y_pred)
        for cat in result["ordered_labels"]:
            d = result["categories"][cat]
            if d["true_n"] > 0:
                assert d["hit_rate"] == pytest.approx(1.0), f"Category {cat}: expected hit_rate=1"

    def test_perfect_model_mae_is_zero(self):
        y_true = np.array([20.0, 50.0, 70.0, 85.0, 97.0])
        y_pred = y_true.copy()
        result = category_calibration_summary(y_true, y_pred)
        for cat in result["ordered_labels"]:
            d = result["categories"][cat]
            if d["true_n"] > 0:
                assert d["mae"] == pytest.approx(0.0, abs=1e-10)

    def test_overall_accuracy_in_unit_interval(self):
        y_true, y_pred = _ccme_range_data(300)
        result = category_calibration_summary(y_true, y_pred)
        assert 0.0 <= result["overall_category_accuracy"] <= 1.0

    def test_overall_accuracy_is_one_for_perfect(self):
        y_true = np.array([20.0, 50.0, 70.0, 85.0, 97.0])
        y_pred = y_true.copy()
        result = category_calibration_summary(y_true, y_pred)
        assert result["overall_category_accuracy"] == pytest.approx(1.0)

    def test_custom_thresholds(self):
        custom = {"Low": (0.0, 50.0), "High": (51.0, 100.0)}
        y_true = np.array([25.0, 75.0])
        y_pred = np.array([30.0, 80.0])
        result = category_calibration_summary(y_true, y_pred, thresholds=custom)
        assert result["ordered_labels"] == ["Low", "High"]
        assert result["categories"]["Low"]["hit_rate"] == pytest.approx(1.0)
        assert result["categories"]["High"]["hit_rate"] == pytest.approx(1.0)

    def test_empty_category_gives_nan_hit_rate(self):
        """If no true samples fall in a category, hit_rate should be nan."""
        # All samples in Excellent range
        y_true = np.array([96.0, 97.0, 98.0])
        y_pred = np.array([96.0, 97.0, 98.0])
        result = category_calibration_summary(y_true, y_pred)
        for cat in ["Poor", "Marginal", "Fair", "Good"]:
            assert math.isnan(result["categories"][cat]["hit_rate"]), (
                f"Category {cat} should have nan hit_rate when empty"
            )

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="same length"):
            category_calibration_summary([50.0, 60.0], [50.0])

    def test_hit_rate_is_per_category_recall(self):
        """hit_rate = TP / (TP + FN) for each category."""
        # 2 Poor samples: model predicts Poor for one, Marginal for other
        y_true = np.array([20.0, 25.0, 85.0])
        y_pred = np.array([22.0, 55.0, 85.0])  # 25 -> ~55 = Marginal
        result = category_calibration_summary(y_true, y_pred)
        poor = result["categories"]["Poor"]
        assert poor["true_n"] == 2
        assert poor["hit_rate"] == pytest.approx(0.5, rel=1e-9)


# ===========================================================================
# TestCalibrationSummary
# ===========================================================================


class TestCalibrationSummary:
    def test_returns_all_top_level_keys(self):
        y_true, y_pred = _ccme_range_data()
        result = calibration_summary(y_true, y_pred)
        for key in (
            "regression_metrics",
            "ece_metrics",
            "reliability_diagram",
            "category_calibration",
        ):
            assert key in result

    def test_regression_metrics_matches_standalone(self):
        from src.metrics import regression_metrics

        y_true, y_pred = _ccme_range_data(200)
        full = calibration_summary(y_true, y_pred)
        standalone = regression_metrics(y_true, y_pred)
        for k in standalone:
            assert full["regression_metrics"][k] == pytest.approx(standalone[k], rel=1e-9)

    def test_ece_metrics_matches_standalone(self):
        y_true, y_pred = _ccme_range_data(200)
        full = calibration_summary(y_true, y_pred)
        standalone = expected_calibration_error(y_true, y_pred)
        assert full["ece_metrics"]["ece"] == pytest.approx(standalone["ece"], rel=1e-9)

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="same length"):
            calibration_summary([50.0, 60.0], [50.0])

    def test_perfect_predictor_full_summary(self):
        """All calibration metrics should indicate no error for a perfect predictor."""
        y_true, y_pred = _perfect_data(300)
        result = calibration_summary(y_true, y_pred)
        assert result["regression_metrics"]["mae"] == pytest.approx(0.0, abs=1e-10)
        assert result["ece_metrics"]["ece"] == pytest.approx(0.0, abs=1e-10)
        assert result["category_calibration"]["overall_category_accuracy"] == pytest.approx(1.0)

    def test_custom_n_bins_propagates(self):
        y_true, y_pred = _ccme_range_data(300)
        result = calibration_summary(y_true, y_pred, n_bins=5)
        assert result["ece_metrics"]["n_bins"] == 5
        assert result["reliability_diagram"]["n_bins"] == 5
