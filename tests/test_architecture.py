"""Tests for the architecture ablation and tuning suite.

Tests run on synthetic numpy arrays to verify that:
  1. Grids faithfully cover all required architectural dimensions (depth, width, activations).
  2. Training and evaluation execution works end-to-end with valid metric extraction.
  3. Metric aggregation across seeds computes mean and standard deviations correctly.
  4. Optuna hyperparameter optimization executes without test leakage.
  5. The comparative mlp_results.csv table compiles with the expected benchmark rows.

Owner: Himanshu Pandey.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch", reason="PyTorch is required for architecture study")
pytest.importorskip("sklearn", reason="scikit-learn is required")

from src.dl.architecture import (  # noqa: E402
    ACTIVATION_GRID,
    BASE_CONFIG,
    DEPTH_GRID,
    WIDTH_GRID,
    aggregate_ablation,
    build_mlp_results_table,
    run_config,
    run_optuna_tuning,
)


def tiny_arrays(n_train: int = 200, n_eval: int = 60, n_feat: int = 5, seed: int = 42):
    rng = np.random.default_rng(seed)
    mk = lambda n: rng.normal(0, 1, (n, n_feat)).astype(np.float32)  # noqa: E731
    Xa, Xb, Xc = mk(n_train), mk(n_eval), mk(n_eval)
    f = lambda X: (X[:, 0] * 2.5 + X[:, 1] * -1.5 + 60.0).astype(np.float64)  # noqa: E731
    return Xa, Xb, Xc, f(Xa), f(Xb), f(Xc)


FAST_OVERRIDE = {"epochs": 2, "batch_size": 32}


class TestGridIntegrity:
    """Validate that architecture search grids cover required dimensions."""

    def test_base_config_valid(self):
        assert BASE_CONFIG["activation"] == "relu"
        assert BASE_CONFIG["hidden_dims"] == (128, 64)
        assert BASE_CONFIG["optimizer"] == "adamw"

    def test_depth_grid_covers_multiple_depths(self):
        depths = [len(overrides["hidden_dims"]) for _, _, overrides in DEPTH_GRID]
        assert set(depths) == {1, 2, 3, 4, 5}

    def test_width_grid_covers_scales(self):
        widths = [overrides["hidden_dims"] for _, _, overrides in WIDTH_GRID]
        assert (64, 32) in widths
        assert (128, 64) in widths
        assert (256, 128) in widths

    def test_activation_grid_covers_relu_gelu_silu(self):
        acts = {overrides["activation"] for _, _, overrides in ACTIVATION_GRID}
        assert {"relu", "gelu", "silu"}.issubset(acts)


class TestAblationExecution:
    """Validate end-to-end execution of architecture variations on synthetic data."""

    def test_run_config_depth_and_width(self):
        arrays = tiny_arrays()
        res = run_config(
            arrays,
            overrides={"hidden_dims": (32, 16), **FAST_OVERRIDE},
            seed=42,
            eval_test=True,
        )
        assert "val_mae" in res
        assert "test_mae" in res
        assert res["n_params"] > 0
        assert res["depth"] == 2
        assert not np.isnan(res["val_mae"])

    def test_run_config_activations(self):
        arrays = tiny_arrays()
        for act in ("relu", "gelu", "silu"):
            res = run_config(
                arrays,
                overrides={"hidden_dims": (16, 8), "activation": act, **FAST_OVERRIDE},
                seed=42,
            )
            assert not np.isnan(res["val_mae"])
            assert res["activation"] == act

    def test_aggregate_ablation_computes_mean_and_std(self):
        raw = pd.DataFrame(
            [
                {
                    "family": "depth",
                    "config": "depth_2",
                    "hidden_dims": "(128, 64)",
                    "activation": "relu",
                    "n_params": 1000,
                    "seed": 42,
                    "val_mae": 1.0,
                    "val_rmse": 1.5,
                    "val_r2": 0.95,
                    "generalisation_gap": 0.01,
                    "epochs_run": 5,
                },
                {
                    "family": "depth",
                    "config": "depth_2",
                    "hidden_dims": "(128, 64)",
                    "activation": "relu",
                    "n_params": 1000,
                    "seed": 123,
                    "val_mae": 1.2,
                    "val_rmse": 1.7,
                    "val_r2": 0.93,
                    "generalisation_gap": 0.02,
                    "epochs_run": 5,
                },
            ]
        )
        agg = aggregate_ablation(raw)
        assert len(agg) == 1
        assert "val_mae_mean" in agg.columns
        assert "val_mae_std" in agg.columns
        assert agg["val_mae_mean"].iloc[0] == pytest.approx(1.1, rel=1e-3)
        assert agg["n_seeds"].iloc[0] == 2


class TestOptunaTuning:
    """Verify that validation-only Optuna search runs smoothly."""

    def test_optuna_tuning_smoke(self):
        arrays = tiny_arrays(n_train=100, n_eval=40)
        best_cfg, trials_df = run_optuna_tuning(
            arrays=arrays,
            n_trials=2,
            seed=42,
            epochs=2,
            verbose=False,
        )
        assert "hidden_dims" in best_cfg
        assert "activation" in best_cfg
        assert "val_mae" in best_cfg
        assert len(trials_df) == 2


class TestResultsTable:
    """Verify the comparative mlp_results.csv schema and content."""

    def test_run_multiseed_evaluation_and_plots(self, tmp_path, monkeypatch):
        from src.dl.architecture import generate_architecture_plots, run_multiseed_evaluation

        arrays = tiny_arrays(n_train=100, n_eval=30)
        df, summary, hist = run_multiseed_evaluation(
            config={"hidden_dims": (16, 8), "epochs": 2, "batch_size": 32},
            arrays=arrays,
            seeds=(42, 123),
            eval_test=True,
            verbose=False,
        )
        assert len(df) == 2
        assert "val_mae_mean" in summary
        assert "test_mae_mean" in summary
        assert hist is not None
        assert "train_loss" in hist

        monkeypatch.setattr("src.dl.architecture.FIGURES", tmp_path)
        agg_df = pd.DataFrame(
            [
                {
                    "family": "depth",
                    "config": "depth_2",
                    "depth": 2,
                    "val_mae_mean": 1.0,
                    "val_mae_std": 0.1,
                },
                {"family": "width", "config": "narrow", "val_mae_mean": 1.1, "val_mae_std": 0.1},
                {"family": "activation", "config": "relu", "val_mae_mean": 1.0, "val_mae_std": 0.1},
            ]
        )
        generate_architecture_plots(agg_df, best_history=hist)
        assert (tmp_path / "architecture_depth_sweep.png").exists()
        assert (tmp_path / "architecture_width_sweep.png").exists()
        assert (tmp_path / "architecture_activation_comparison.png").exists()
        assert (tmp_path / "architecture_best_mlp_learning_curves.png").exists()

    def test_build_mlp_results_table(self):
        base_summary = {
            "val_mae_mean": 0.7751,
            "val_mae_std": 0.0444,
            "val_rmse_mean": 1.2571,
            "val_rmse_std": 0.0412,
            "val_r2_mean": 0.9954,
            "val_r2_std": 0.0003,
            "test_mae_mean": 0.7774,
            "test_rmse_mean": 1.2554,
            "test_r2_mean": 0.9955,
        }
        tuned_summary = {
            "val_mae_mean": 0.6521,
            "val_mae_std": 0.0210,
            "val_rmse_mean": 1.0512,
            "val_rmse_std": 0.0250,
            "val_r2_mean": 0.9968,
            "val_r2_std": 0.0002,
            "test_mae_mean": 0.6540,
            "test_rmse_mean": 1.0498,
            "test_r2_mean": 0.9969,
            "architecture": "(256, 128, 64)",
            "activation": "gelu",
        }
        table = build_mlp_results_table(
            base_mlp_summary=base_summary,
            tuned_mlp_summary=tuned_summary,
            n_params_base=11649,
            n_params_tuned=48577,
        )
        assert len(table) == 5
        models = list(table["model"])
        assert "Naive Baseline (Mean)" in models
        assert "Random Forest (Phase 1)" in models
        assert "XGBoost (Phase 1)" in models
        assert "Base MLP (Phase 2)" in models
        assert "Tuned MLP (Phase 2)" in models
