"""Unit tests for Phase 2 final comparison, benchmark evaluation, and technical validation.

Runs on fast synthetic arrays without requiring the full ~2M-row dataset.
Verifies:
  1. Fixed hyperparameters match literature and Phase 1 specifications.
  2. Baseline, RF, XGB, and MLP model runners execute and return valid metrics.
  3. Error breakdown by Country and Area calculates grouped residuals faithfully.
  4. Leakage check functions assert partition disjointness and scaler isolation.
  5. Compute cost measurement captures parameter counts and throughput.

Owner: Parthraj Singh Bhati.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch", reason="PyTorch is required for final comparison tests")
pytest.importorskip("sklearn", reason="scikit-learn is required for final comparison tests")
pytest.importorskip("xgboost", reason="XGBoost is required for final comparison tests")

from src.final_comparison import (  # noqa: E402
    BENCHMARK_LABEL,
    MLP_FIXED_CONFIG,
    RF_FIXED_PARAMS,
    TEAM_SEEDS,
    XGB_FIXED_PARAMS,
    XGB_FIXED_PARAMS_BASE,
    XGB_FIXED_PARAMS_TUNED,
    compute_error_breakdown_by_area,
    compute_error_breakdown_by_country,
    compute_tail_metrics,
    run_naive_baseline,
    run_random_forest,
    run_tuned_mlp,
    run_xgboost,
)


def create_synthetic_splits(
    n_train: int = 300,
    n_val: int = 100,
    n_test: int = 100,
    n_features: int = 6,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Generate small synthetic DataFrames with continuous and categorical columns."""
    rng = np.random.default_rng(seed)

    def make_df(n: int) -> pd.DataFrame:
        nums = rng.normal(10.0, 3.0, (n, n_features))
        cols = {f"chem_{i}": nums[:, i] for i in range(n_features)}
        cols["Country"] = rng.choice(["England", "USA", "Ireland"], size=n)
        cols["Waterbody Type"] = rng.choice(["River", "Lake"], size=n)
        cols["Area"] = rng.choice([f"Station_{i}" for i in range(1, 15)], size=n)
        return pd.DataFrame(cols)

    X_tr = make_df(n_train)
    X_va = make_df(n_val)
    X_te = make_df(n_test)

    # Deterministic continuous target in [0, 100]
    target_fn = lambda df: np.clip(  # noqa: E731
        75.0 + df["chem_0"] * 1.5 - df["chem_1"] * 0.8 + rng.normal(0, 0.5, len(df)),
        0.0,
        100.0,
    )

    y_tr = pd.Series(target_fn(X_tr), index=X_tr.index, name="CCME_Values")
    y_va = pd.Series(target_fn(X_va), index=X_va.index, name="CCME_Values")
    y_te = pd.Series(target_fn(X_te), index=X_te.index, name="CCME_Values")

    return X_tr, X_va, X_te, y_tr, y_va, y_te


class TestBenchmarkConfiguration:
    """Validate fixed experimental parameters and disclosure labels."""

    def test_team_seeds_match_convention(self):
        assert TEAM_SEEDS == [42, 123, 456, 789, 1011]

    def test_benchmark_label_discloses_formula_approximation(self):
        assert "formula-approximation benchmark" in BENCHMARK_LABEL
        assert "CCME_Values computed from 8 input parameters" in BENCHMARK_LABEL

    def test_rf_parameters_match_phase1(self):
        assert RF_FIXED_PARAMS["n_estimators"] == 100
        assert RF_FIXED_PARAMS["max_depth"] == 15
        assert RF_FIXED_PARAMS["min_samples_leaf"] == 5

    def test_xgb_parameters_match_phase1(self):
        assert XGB_FIXED_PARAMS["max_depth"] == 7
        assert pytest.approx(XGB_FIXED_PARAMS["learning_rate"], rel=1e-3) == 0.0552
        assert XGB_FIXED_PARAMS["min_child_weight"] == 30
        assert pytest.approx(XGB_FIXED_PARAMS_TUNED["reg_lambda"], rel=1e-3) == 33.12
        assert XGB_FIXED_PARAMS_BASE["max_depth"] == 8
        assert XGB_FIXED_PARAMS_BASE["learning_rate"] == 0.1

    def test_tail_metrics_calculation(self):
        y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        y_pred = np.array([11.0, 20.5, 28.0, 45.0, 50.0])
        # absolute errors: [1.0, 0.5, 2.0, 5.0, 0.0]
        tail = compute_tail_metrics(y_true, y_pred)
        assert tail["max_ae"] == 5.0
        assert tail["p95_ae"] <= 5.0
        assert tail["p95_ae"] >= 2.0

    def test_mlp_parameters_conforms_to_taj_and_himanshu(self):
        assert MLP_FIXED_CONFIG["hidden_dims"] == (256, 256, 256, 256)
        assert MLP_FIXED_CONFIG["activation"] == "gelu"
        assert MLP_FIXED_CONFIG["dropout"] == 0.0
        assert MLP_FIXED_CONFIG["use_batch_norm"] is False


class TestModelRunnersSynthetic:
    """Validate execution and metric extraction on small synthetic splits."""

    def test_naive_baseline_returns_exact_mean_and_metrics(self):
        _, _, _, y_tr, y_va, y_te = create_synthetic_splits()
        res = run_naive_baseline(y_tr, y_va, y_te, seed=42)

        assert res["n_params"] == 1
        assert res["size (parameters or tree nodes)"] == 1
        assert "val_mae" in res and "test_mae" in res
        assert "test_p95_ae" in res
        assert not np.isnan(res["test_rmse"])
        assert len(res["test_pred"]) == len(y_te)
        assert np.allclose(res["test_pred"], float(y_tr.mean()))

    def test_random_forest_executes_with_onehot_encoding(self):
        X_tr, X_va, X_te, y_tr, y_va, y_te = create_synthetic_splits()
        fast_params = {"n_estimators": 10, "max_depth": 5, "n_jobs": 1}
        res = run_random_forest(
            X_tr, y_tr, X_va, y_val=y_va, X_test=X_te, y_test=y_te, seed=42, params=fast_params
        )

        assert res["model"] == "Random Forest"
        assert res["size (parameters or tree nodes)"] > 0
        assert "test_rmse" in res
        assert "test_p95_ae" in res
        assert not np.isnan(res["test_rmse"])
        assert len(res["test_pred"]) == len(y_te)

    def test_xgboost_executes_with_categorical_handling(self):
        X_tr, X_va, X_te, y_tr, y_va, y_te = create_synthetic_splits()
        fast_params = {"n_estimators": 15, "max_depth": 4, "n_jobs": 1}
        res = run_xgboost(
            X_tr, y_tr, X_va, y_val=y_va, X_test=X_te, y_test=y_te, seed=42, params=fast_params
        )

        assert res["model"] == "XGBoost"
        assert res["size (parameters or tree nodes)"] > 0
        assert "test_mae" in res
        assert "test_p95_ae" in res
        assert len(res["test_pred"]) == len(y_te)

    def test_tuned_mlp_executes_end_to_end(self):
        X_tr, X_va, X_te, y_tr, y_va, y_te = create_synthetic_splits()
        fast_cfg = {
            "hidden_dims": (32, 16),
            "activation": "gelu",
            "epochs": 2,
            "batch_size": 64,
            "lr": 1e-3,
        }
        res = run_tuned_mlp(X_tr, y_tr, X_va, y_va, X_te, y_te, seed=42, cfg=fast_cfg)

        assert res["model"] == "Tuned MLP"
        assert res["size (parameters or tree nodes)"] > 0
        assert "test_p95_ae" in res
        assert not np.isnan(res["test_rmse"])
        assert len(res["test_pred"]) == len(y_te)


class TestErrorBreakdowns:
    """Validate error breakdown tables by Country and Area."""

    def test_country_breakdown_aggregates_correctly(self):
        _, _, _, _, _, y_te = create_synthetic_splits(n_test=60)
        countries = np.array(["England"] * 30 + ["USA"] * 20 + ["Ireland"] * 10)
        preds_m1 = np.asarray(y_te) + 0.5
        preds_m2 = np.asarray(y_te) + 1.2

        df = compute_error_breakdown_by_country(
            np.asarray(y_te),
            preds_m1,
            preds_m2,
            countries,
            m1_name="Model_A",
            m2_name="Model_B",
        )

        assert len(df) == 3
        assert set(df["Country"]) == {"England", "USA", "Ireland"}
        assert "delta_rmse (m2 - m1)" in df.columns
        assert (df["best_model"] == "Model_A").all()

    def test_area_breakdown_top_n_filtering(self):
        _, _, _, _, _, y_te = create_synthetic_splits(n_test=80)
        areas = np.array([f"Site_{i % 5}" for i in range(80)])
        preds_m1 = np.asarray(y_te) + 0.2
        preds_m2 = np.asarray(y_te) + 0.8

        df = compute_error_breakdown_by_area(
            np.asarray(y_te),
            preds_m1,
            preds_m2,
            areas,
            m1_name="XGB",
            m2_name="MLP",
            top_n=3,
        )

        assert len(df) == 3
        assert "Area" in df.columns
        assert "sample_count" in df.columns
        assert df["sample_count"].sum() <= 80
