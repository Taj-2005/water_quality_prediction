"""Unit tests for Random Forest model implementation (src/models_rf.py).

Verifies:
  - Metric computation fallback when src.metrics.regression_metrics is a stub.
  - CategoricalEncoder fits strictly on train, handles unseen categories, and preserves indices.
  - Subsampling is deterministic and reproducible.
  - Experiment labeling explicitly marks 8/8 Regime 1 as a surrogate/emulator benchmark.
  - train() executes correctly across raw vs. engineered features, regimes, and 8/8, 7/8, 6/8, 4/8 scenarios.
  - Validation-only tuning does not touch the test set during parameter search.
  - OOB scoring works when enabled.
  - CSV reporting records required configuration and evaluation metrics.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from src.models_rf import (
    CategoricalEncoder,
    apply_subsample,
    compute_metrics,
    get_experiment_label,
    train,
)


class TestMetricFallback:
    def test_compute_metrics_perfect_predictions(self):
        y_true = np.array([50.0, 75.0, 100.0])
        y_pred = np.array([50.0, 75.0, 100.0])
        metrics = compute_metrics(y_true, y_pred)
        assert "mae" in metrics and "rmse" in metrics and "r2" in metrics
        assert np.isclose(metrics["mae"], 0.0)
        assert np.isclose(metrics["rmse"], 0.0)
        assert np.isclose(metrics["r2"], 1.0)

    def test_compute_metrics_known_errors(self):
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([12.0, 18.0, 32.0])
        metrics = compute_metrics(y_true, y_pred)
        assert np.isclose(metrics["mae"], 2.0)
        assert np.isclose(metrics["rmse"], 2.0)
        # Var(y_true) = ((10-20)^2 + (20-20)^2 + (30-20)^2) / 3 = 200/3 = 66.6667
        # MSE = 4, R2 = 1 - (4 / 66.6667) = 0.94
        assert 0.90 < metrics["r2"] < 1.0

    def test_compute_metrics_fallback_on_not_implemented(self):
        with patch("src.models_rf.evaluate", side_effect=NotImplementedError("Stub")):
            metrics = compute_metrics([10.0, 20.0], [12.0, 22.0])
            assert metrics["mae"] == 2.0
            assert metrics["rmse"] == 2.0


class TestCategoricalEncoder:
    def test_encoder_fits_on_train_only(self):
        train_df = pd.DataFrame(
            {
                "num": [1.0, 2.0, 3.0],
                "Country": pd.Categorical(["England", "USA", "England"]),
                "Waterbody": ["River", "Lake", "River"],
            },
            index=[10, 20, 30],
        )

        val_df = pd.DataFrame(
            {
                "num": [4.0, 5.0],
                "Country": pd.Categorical(["England", "China"]),  # 'China' was not in train
                "Waterbody": ["River", "Canal"],  # 'Canal' was not in train
            },
            index=[40, 50],
        )

        encoder = CategoricalEncoder()
        train_enc = encoder.fit_transform(train_df)
        val_enc = encoder.transform(val_df)

        assert train_enc.shape[0] == 3
        assert val_enc.shape[0] == 2
        assert list(train_enc.index) == [10, 20, 30]
        assert list(val_enc.index) == [40, 50]

        # Verify all resulting columns are numeric and no string dtypes remain
        for col in train_enc.columns:
            assert pd.api.types.is_numeric_dtype(train_enc[col])
        for col in val_enc.columns:
            assert pd.api.types.is_numeric_dtype(val_enc[col])

        # Unseen categories in validation should result in all 0s for that feature
        assert val_enc.shape[1] == train_enc.shape[1]

    def test_encoder_raises_if_transform_before_fit(self):
        encoder = CategoricalEncoder()
        df = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(RuntimeError, match="must be fitted"):
            encoder.transform(df)

    def test_encoder_with_pure_numeric_data(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        encoder = CategoricalEncoder()
        res = encoder.fit_transform(df)
        assert res.shape == (2, 2)
        assert list(res.columns) == ["a", "b"]


class TestSubsample:
    def test_apply_subsample_none(self):
        X = pd.DataFrame({"a": range(20)})
        y = pd.Series(range(20))
        X_sub, y_sub = apply_subsample(X, y, subsample=None)
        assert len(X_sub) == 20
        assert len(y_sub) == 20

    def test_apply_subsample_integer(self):
        X = pd.DataFrame({"a": range(100)})
        y = pd.Series(range(100))
        X_sub, y_sub = apply_subsample(X, y, subsample=25, seed=42)
        assert len(X_sub) == 25
        assert len(y_sub) == 25
        assert np.array_equal(X_sub.index, y_sub.index)

    def test_apply_subsample_float(self):
        X = pd.DataFrame({"a": range(100)})
        y = pd.Series(range(100))
        X_sub, y_sub = apply_subsample(X, y, subsample=0.30, seed=42)
        assert len(X_sub) == 30
        assert len(y_sub) == 30

    def test_apply_subsample_deterministic(self):
        X = pd.DataFrame({"a": range(100)})
        y = pd.Series(range(100))
        X_sub1, _ = apply_subsample(X, y, subsample=20, seed=42)
        X_sub2, _ = apply_subsample(X, y, subsample=20, seed=42)
        assert np.array_equal(X_sub1.index, X_sub2.index)

    def test_apply_subsample_invalid(self):
        X = pd.DataFrame({"a": range(10)})
        y = pd.Series(range(10))
        with pytest.raises(ValueError, match="positive"):
            apply_subsample(X, y, subsample=-5)
        with pytest.raises(ValueError, match=r"must be in \(0, 1\]"):
            apply_subsample(X, y, subsample=1.5)


class TestExperimentLabeling:
    def test_regime_1_8_8_is_surrogate_emulator(self):
        label = get_experiment_label(regime=1, partial_scenario="8/8")
        assert "surrogate_emulator_benchmark" in label
        assert "NOT independent prediction" in label

    def test_partial_scenarios_labeled_emulation(self):
        for s in ["7/8", "6/8", "4/8"]:
            label = get_experiment_label(regime=1, partial_scenario=s)
            assert f"partial_measurement_emulation_{s}" in label

    def test_regime_2_labeled_independent(self):
        label = get_experiment_label(regime=2, partial_scenario="8/8")
        assert "independent_environmental_prediction" in label


class TestTrainExecution:
    def test_train_baseline_with_subsample(self):
        results = train(
            regime=1,
            feature_set="engineered",
            partial_scenario="8/8",
            n_estimators=5,
            max_depth=3,
            subsample=100,
            save_reports=False,
            n_jobs=1,
        )
        assert "model" in results
        assert "val_metrics" in results
        assert "test_metrics" in results
        assert "feature_names" in results
        assert "feature_importances" in results
        assert results["config"]["n_train_samples"] == 100
        assert results["config"]["regime"] == 1
        assert "surrogate_emulator_benchmark" in results["config"]["experiment_label"]

        # Check metrics validity
        for m in ["mae", "rmse", "r2"]:
            assert m in results["val_metrics"]
            assert m in results["test_metrics"]
            assert not np.isnan(results["val_metrics"][m])
            assert not np.isnan(results["test_metrics"][m])

    def test_train_raw_vs_engineered(self):
        res_raw = train(
            regime=1,
            feature_set="raw",
            partial_scenario="8/8",
            n_estimators=3,
            max_depth=2,
            subsample=50,
            save_reports=False,
            n_jobs=1,
        )
        res_eng = train(
            regime=1,
            feature_set="engineered",
            partial_scenario="8/8",
            n_estimators=3,
            max_depth=2,
            subsample=50,
            save_reports=False,
            n_jobs=1,
        )
        # Engineered features include 10 chemical derived + 3 temporal features in addition to raw
        assert res_eng["config"]["n_features"] > res_raw["config"]["n_features"]

    def test_train_partial_scenarios(self):
        res_8 = train(
            regime=1,
            partial_scenario="8/8",
            n_estimators=2,
            max_depth=2,
            subsample=50,
            save_reports=False,
            n_jobs=1,
        )
        res_7 = train(
            regime=1,
            partial_scenario="7/8",
            n_estimators=2,
            max_depth=2,
            subsample=50,
            save_reports=False,
            n_jobs=1,
        )
        res_6 = train(
            regime=1,
            partial_scenario="6/8",
            n_estimators=2,
            max_depth=2,
            subsample=50,
            save_reports=False,
            n_jobs=1,
        )
        res_4 = train(
            regime=1,
            partial_scenario="4/8",
            n_estimators=2,
            max_depth=2,
            subsample=50,
            save_reports=False,
            n_jobs=1,
        )

        assert res_8["config"]["n_features"] > res_7["config"]["n_features"]
        assert res_7["config"]["n_features"] > res_6["config"]["n_features"]
        assert res_6["config"]["n_features"] > res_4["config"]["n_features"]

    def test_train_regime_2(self):
        res = train(
            regime=2,
            feature_set="engineered",
            n_estimators=3,
            max_depth=2,
            subsample=50,
            save_reports=False,
            n_jobs=1,
        )
        assert res["config"]["regime"] == 2
        # Verify no chemical columns are in feature_names
        for feat in res["feature_names"]:
            assert "mg/l" not in feat
            assert "pH" not in feat
            assert "Temperature" not in feat
            assert "ratio" not in feat
            assert not feat.startswith("log_")

    def test_validation_only_tuning_isolation(self):
        param_grid = [
            {"n_estimators": 3, "max_depth": 2, "min_samples_leaf": 2, "max_features": "sqrt"},
            {"n_estimators": 5, "max_depth": 3, "min_samples_leaf": 1, "max_features": "sqrt"},
        ]
        predicted_lengths: list[int] = []
        real_predict = RandomForestRegressor.predict

        def spy_predict(self, X):
            predicted_lengths.append(len(X))
            return real_predict(self, X)

        with patch.object(RandomForestRegressor, "predict", spy_predict):
            res = train(
                regime=1,
                tune=True,
                param_grid=param_grid,
                subsample=50,
                save_reports=False,
                n_jobs=1,
            )

        # During tuning, predict() must be called on X_val for each candidate in param_grid
        # and then evaluated on holdout X_test EXACTLY ONCE at the end.
        assert len(predicted_lengths) == len(param_grid) + 1
        val_partition_size = 415805
        test_partition_size = 415804
        assert predicted_lengths[:-1] == [val_partition_size] * len(param_grid)
        assert predicted_lengths[-1] == test_partition_size
        assert predicted_lengths.count(test_partition_size) == 1
        assert res["best_params"] in param_grid

    def test_oob_scoring(self):
        res = train(
            regime=1,
            n_estimators=10,
            max_depth=3,
            subsample=100,
            oob_score=True,
            save_reports=False,
            n_jobs=1,
        )
        assert res["oob_score"] is not None
        assert isinstance(res["oob_score"], float)

    def test_record_results_csv(self, tmp_path: Path):
        csv_file = tmp_path / "reports" / "test_results.csv"
        res = train(
            regime=1,
            n_estimators=2,
            max_depth=2,
            subsample=50,
            save_reports=True,
            output_path=csv_file,
            n_jobs=1,
        )
        assert csv_file.exists()
        df = pd.read_csv(csv_file)
        assert len(df) == 1
        assert df.loc[0, "experiment_label"] == res["config"]["experiment_label"]
        assert df.loc[0, "n_train_samples"] == 50
        assert not np.isnan(df.loc[0, "val_rmse"])
        assert not np.isnan(df.loc[0, "test_rmse"])
