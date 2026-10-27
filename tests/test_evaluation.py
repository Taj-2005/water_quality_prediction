"""Unit tests for the common evaluation interface.   Owner: Parthraj Singh Bhati."""

import numpy as np
import pytest

from src.evaluation import (
    bootstrap_test_metrics,
    evaluate_by_group,
    evaluate_regression,
)


class TestEvaluateRegression:
    def test_perfect_prediction(self):
        y_true = np.array([50.0, 60.0, 70.0, 80.0, 100.0])
        y_pred = np.array([50.0, 60.0, 70.0, 80.0, 100.0])
        metrics = evaluate_regression(y_true, y_pred)

        assert metrics["mae"] == 0.0
        assert metrics["mse"] == 0.0
        assert metrics["rmse"] == 0.0
        assert metrics["r2"] == 1.0
        assert metrics["explained_variance"] == 1.0
        assert metrics["medae"] == 0.0
        assert metrics["max_error"] == 0.0
        assert metrics["residual_mean"] == 0.0
        assert metrics["residual_std"] == 0.0

    def test_known_numerical_values(self):
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([12.0, 19.0, 34.0])
        # residuals: y_true - y_pred = [-2, +1, -4]
        # abs residuals: [2, 1, 4] -> mae = 7/3
        # sq residuals: [4, 1, 16] -> mse = 21/3 = 7.0 -> rmse = sqrt(7)
        # medae: 2.0
        # max_error: 4.0
        # residual_mean: (-2 + 1 - 4)/3 = -5/3 = -1.666667
        metrics = evaluate_regression(y_true, y_pred)
        assert pytest.approx(metrics["mae"], 1e-5) == 7.0 / 3.0
        assert pytest.approx(metrics["mse"], 1e-5) == 7.0
        assert pytest.approx(metrics["rmse"], 1e-5) == np.sqrt(7.0)
        assert pytest.approx(metrics["medae"], 1e-5) == 2.0
        assert pytest.approx(metrics["max_error"], 1e-5) == 4.0
        assert pytest.approx(metrics["residual_mean"], 1e-5) == -5.0 / 3.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="Length mismatch"):
            evaluate_regression([1.0, 2.0], [1.0])

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            evaluate_regression([], [])


class TestEvaluateByGroup:
    def test_group_stratification(self):
        y_true = np.array([10.0, 20.0, 30.0, 40.0])
        y_pred = np.array([10.0, 20.0, 32.0, 42.0])
        groups = np.array(["CountryA", "CountryA", "CountryB", "CountryB"])

        df_grp = evaluate_by_group(y_true, y_pred, groups, group_name="Country")
        assert len(df_grp) == 2
        assert "Country" in df_grp.columns
        assert "CountryA" in df_grp["Country"].values
        assert "CountryB" in df_grp["Country"].values

        row_a = df_grp[df_grp["Country"] == "CountryA"].iloc[0]
        assert row_a["mae"] == 0.0

        row_b = df_grp[df_grp["Country"] == "CountryB"].iloc[0]
        assert row_b["mae"] == 2.0


class TestBootstrapTestMetrics:
    def test_bootstrap_ci(self):
        y_true = np.linspace(10.0, 100.0, 100)
        rng = np.random.default_rng(42)
        y_pred = y_true + rng.normal(0, 1, 100)

        ci_res = bootstrap_test_metrics(y_true, y_pred, n_bootstraps=100, ci=0.95, seed=42)
        assert "mae" in ci_res
        assert "rmse" in ci_res
        assert "r2" in ci_res

        mae_info = ci_res["mae"]
        assert mae_info["ci_lower"] <= mae_info["point_estimate"] <= mae_info["ci_upper"]
