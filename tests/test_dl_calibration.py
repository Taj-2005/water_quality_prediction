"""Unit tests for Phase 2 DL calibration, uncertainty and explainability.

Owner: Satya Yadav.
Part of Phase 2: Technical validation & Calibration.

Run:
    pytest tests/test_dl_calibration.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch is required for DL calibration tests")
nn = torch.nn

from src.dl.calibration import (
    calibrate_conformal_quantile,
    conformal_prediction_intervals,
    enable_dropout_only,
    mc_dropout_prediction_intervals,
    monte_carlo_dropout_predict,
    permutation_feature_importance_mlp,
    prediction_interval_coverage,
)


class DummyMLPWithDropout(nn.Module):
    """Simple 2-layer MLP with Dropout for unit testing."""

    def __init__(self, input_dim: int = 4, hidden_dim: int = 16, dropout: float = 0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc1(x)
        out = self.bn(out)
        out = self.act(out)
        out = self.drop(out)
        return self.fc2(out)


# ---------------------------------------------------------------------------
# Tests: prediction_interval_coverage
# ---------------------------------------------------------------------------


class TestPredictionIntervalCoverage:
    def test_perfect_coverage(self):
        y_true = np.array([50.0, 60.0, 70.0, 80.0])
        y_lower = np.array([40.0, 50.0, 60.0, 70.0])
        y_upper = np.array([60.0, 70.0, 80.0, 90.0])
        res = prediction_interval_coverage(y_true, y_lower, y_upper)
        assert res["picp"] == pytest.approx(1.0)
        assert res["mpiw"] == pytest.approx(20.0)
        assert res["n_captured"] == 4
        assert res["under_coverage_count"] == 0
        assert res["over_coverage_count"] == 0

    def test_partial_coverage(self):
        y_true = np.array([30.0, 60.0, 70.0, 100.0])
        y_lower = np.array([40.0, 50.0, 60.0, 70.0])
        y_upper = np.array([60.0, 70.0, 80.0, 90.0])
        res = prediction_interval_coverage(y_true, y_lower, y_upper)
        assert res["picp"] == pytest.approx(0.5)
        assert res["n_captured"] == 2
        assert res["under_coverage_count"] == 1
        assert res["over_coverage_count"] == 1

    def test_inverted_bounds_auto_corrected(self):
        y_true = np.array([50.0])
        y_lower = np.array([60.0])  # lower > upper
        y_upper = np.array([40.0])
        res = prediction_interval_coverage(y_true, y_lower, y_upper)
        assert res["picp"] == pytest.approx(1.0)
        assert res["mpiw"] == pytest.approx(20.0)

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="Length mismatch"):
            prediction_interval_coverage([50.0], [40.0, 30.0], [60.0])


# ---------------------------------------------------------------------------
# Tests: Conformal Prediction
# ---------------------------------------------------------------------------


class TestConformalPrediction:
    def test_calibrate_conformal_quantile_finite_sample(self):
        # 100 residuals uniformly spaced 0 to 10
        val_residuals = np.linspace(0, 10, 100)
        q_hat = calibrate_conformal_quantile(val_residuals, confidence_level=0.90)
        assert 8.5 <= q_hat <= 10.0

    def test_conformal_prediction_intervals(self):
        y_pred = np.array([50.0, 95.0, 5.0])
        q_hat = 10.0
        lower, upper = conformal_prediction_intervals(y_pred, q_hat, clip_range=(0.0, 100.0))
        assert lower[0] == pytest.approx(40.0)
        assert upper[0] == pytest.approx(60.0)
        # Upper clipped at 100
        assert upper[1] == pytest.approx(100.0)
        # Lower clipped at 0
        assert lower[2] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Tests: Monte Carlo Dropout
# ---------------------------------------------------------------------------


class TestMCDropout:
    def test_enable_dropout_only(self):
        model = DummyMLPWithDropout(dropout=0.5)
        enable_dropout_only(model)
        assert model.drop.training is True
        assert model.bn.training is False

    def test_monte_carlo_dropout_stochasticity(self):
        model = DummyMLPWithDropout(input_dim=4, hidden_dim=16, dropout=0.5)
        x = np.random.randn(20, 4).astype(np.float32)

        res = monte_carlo_dropout_predict(model, x, n_passes=15)
        assert res["mean"].shape == (20,)
        assert res["std"].shape == (20,)
        assert res["all_passes"].shape == (15, 20)
        # Standard deviation should be positive due to active dropout
        assert np.any(res["std"] > 0)

    def test_mc_dropout_prediction_intervals(self):
        mean = np.array([50.0, 80.0])
        std = np.array([5.0, 2.0])
        lower, upper = mc_dropout_prediction_intervals(mean, std, z_score=2.0)
        assert lower[0] == pytest.approx(40.0)
        assert upper[0] == pytest.approx(60.0)
        assert lower[1] == pytest.approx(76.0)
        assert upper[1] == pytest.approx(84.0)


# ---------------------------------------------------------------------------
# Tests: Permutation Feature Importance
# ---------------------------------------------------------------------------


class TestPermutationImportance:
    def test_permutation_importance_identifies_dominant_feature(self):
        # Create synthetic linear target y = 10 * x0 + 0.01 * x1
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, (100, 2)).astype(np.float32)
        y = 10.0 * x[:, 0] + 0.01 * x[:, 1]

        # Simple linear model simulating MLP
        class SimpleNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(2, 1, bias=False)
                with torch.no_grad():
                    self.fc.weight[0, 0] = 10.0
                    self.fc.weight[0, 1] = 0.01

            def forward(self, t):
                return self.fc(t)

        net = SimpleNet()
        df = permutation_feature_importance_mlp(
            net, x, y, feature_names=["important_feat", "unimportant_feat"], n_repeats=3
        )
        assert df.iloc[0]["Feature"] == "important_feat"
        assert df.iloc[0]["Importance_Mean"] > df.iloc[1]["Importance_Mean"]
