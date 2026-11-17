"""Tests for Phase 2 deep learning foundation (seeds, data loaders, training loop, evaluation).

Verifies:
  1. Split size match: Phase 1 split indices match exactly (1,986,397 train, 415,805 val, 415,804 test).
  2. End-to-end training: Tiny placeholder MLP trains on CPU and loss decreases.
  3. Seed determinism: The exact same seed gives identical losses across separate runs.
  4. Checkpointing: Best checkpoint is saved to disk and reloaded.
  5. Run logging: Reports are appended to reports/dl_runs.csv.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch", reason="PyTorch is required for deep learning tests")
pytest.importorskip("sklearn", reason="scikit-learn is required for deep learning tests")

import torch
import torch.nn as nn

from src.config import DATA_SPLITS
from src.dl.data import WaterQualityDataset, get_loaders
from src.dl.seeds import set_seed
from src.dl.train import log_run, train
from src.split import load_splits


class TinyPlaceholderMLP(nn.Module):
    """Tiny placeholder MLP for testing and verification only.

    The official architecture is developed by Himanshu in src/dl/model.py.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TestDLFoundation:
    def test_split_sizes_match_phase1_indices(self):
        """Confirm that get_loaders consumes the exact Phase 1 split indices."""
        if (
            not (DATA_SPLITS / "splits.parquet").exists()
            and not (DATA_SPLITS / "splits.csv").exists()
        ):
            pytest.skip("Phase 1 splits not available on disk (CI runner)")

        splits = load_splits()
        assert len(splits) == 2_818_006

        expected_train = int((splits == "train").sum())
        expected_val = int((splits == "val").sum())
        expected_test = int((splits == "test").sum())

        assert expected_train == 1_986_397
        assert expected_val == 415_805
        assert expected_test == 415_804

        # Verify get_loaders produces datasets of the exact identical sizes
        loaders, input_dim = get_loaders(batch_size=1024, include_metadata=True)

        assert len(loaders.train.dataset) == expected_train
        assert len(loaders.val.dataset) == expected_val
        assert len(loaders.test.dataset) == expected_test
        assert input_dim > 0

    def test_dataset_tensor_conversion(self):
        """Test WaterQualityDataset types and shapes."""
        X = np.random.randn(20, 5).astype(np.float32)
        y = np.random.randn(20).astype(np.float32)

        ds = WaterQualityDataset(X, y)
        assert len(ds) == 20

        x_sample, y_sample = ds[0]
        assert isinstance(x_sample, torch.Tensor)
        assert isinstance(y_sample, torch.Tensor)
        assert x_sample.dtype == torch.float32
        assert y_sample.dtype == torch.float32
        assert y_sample.shape == torch.Size([1])

    def test_cpu_smoke_test_loss_decreases(self):
        """Run 3 epochs on CPU with a subsample and confirm training loss decreases."""
        set_seed(42)
        loaders, input_dim = get_loaders(
            batch_size=64,
            subsample=500,
            include_metadata=False,
            scale_numeric=True,
            seed=42,
        )

        model = TinyPlaceholderMLP(input_dim=input_dim, hidden_dim=16)

        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "epochs": 3,
                "lr": 1e-2,
                "weight_decay": 1e-4,
                "optimizer": "adamw",
                "loss": "mse",
                "device": "cpu",
                "seed": 42,
                "checkpoint_dir": tmpdir,
                "model_name": "smoke_test.pt",
                "log_to_csv": False,
                "verbose": False,
            }

            history, best_val_metrics = train(model, loaders, config)

            # Confirm loss decreases
            assert len(history["train_loss"]) == 3
            assert history["train_loss"][-1] < history["train_loss"][0], (
                f"Loss did not decrease: initial={history['train_loss'][0]:.4f}, "
                f"final={history['train_loss'][-1]:.4f}"
            )

            # Confirm metrics dictionary has standard regression keys
            assert "mae" in best_val_metrics
            assert "rmse" in best_val_metrics
            assert "r2" in best_val_metrics
            assert Path(tmpdir, "smoke_test.pt").exists()

    def test_seed_determinism(self):
        """Confirm that the exact same seed yields identical losses twice."""
        results = []

        for _ in range(2):
            set_seed(123)
            loaders, input_dim = get_loaders(
                batch_size=64,
                subsample=300,
                include_metadata=False,
                scale_numeric=True,
                seed=123,
            )

            model = TinyPlaceholderMLP(input_dim=input_dim, hidden_dim=16)

            with tempfile.TemporaryDirectory() as tmpdir:
                config = {
                    "epochs": 2,
                    "lr": 1e-2,
                    "optimizer": "adamw",
                    "device": "cpu",
                    "seed": 123,
                    "checkpoint_dir": tmpdir,
                    "log_to_csv": False,
                    "verbose": False,
                }
                history, val_metrics = train(model, loaders, config)
                results.append((history["train_loss"], val_metrics["rmse"]))

        run1_losses, run1_rmse = results[0]
        run2_losses, run2_rmse = results[1]

        for l1, l2 in zip(run1_losses, run2_losses, strict=True):
            assert np.isclose(l1, l2, atol=1e-5), f"Loss mismatch: {l1} vs {l2}"
        assert np.isclose(run1_rmse, run2_rmse, atol=1e-5)

    def test_run_logger_appends_csv(self):
        """Test logging to reports/dl_runs.csv."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "dl_runs.csv"

            config = {
                "experiment_label": "test_experiment",
                "batch_size": 128,
                "lr": 1e-3,
                "weight_decay": 1e-4,
                "optimizer": "adamw",
            }
            val_metrics = {"mae": 0.45, "rmse": 0.85, "r2": 0.99}

            out = log_run(
                config=config,
                seed=42,
                val_metrics=val_metrics,
                epochs_trained=5,
                best_epoch=4,
                best_val_loss=0.72,
                scale_target=True,
                csv_path=csv_path,
            )

            assert out.exists()
            content = out.read_text()
            assert "test_experiment" in content
            assert "True" in content  # scale_target logged
            assert "0.45" in content
            assert "0.85" in content
            assert "0.99" in content

    def test_target_stats_from_train_only(self):
        """Verify that y_mean and y_std are computed strictly from the train split."""
        if (
            not (DATA_SPLITS / "splits.parquet").exists()
            and not (DATA_SPLITS / "splits.csv").exists()
        ):
            pytest.skip("Phase 1 splits not available on disk (CI runner)")

        from src.evaluate_api import get_splits

        # 1. Direct Phase 1 splits
        _, _, _, y_train, y_val, y_test = get_splits(scale_numeric=True)

        expected_mean = float(y_train.mean())
        expected_std = float(y_train.std())

        val_mean = float(y_val.mean())
        test_mean = float(y_test.mean())

        # Confirm train mean differs from val/test due to grouping
        assert not np.isclose(expected_mean, val_mean, atol=1e-3)
        assert not np.isclose(expected_mean, test_mean, atol=1e-3)

        # 2. Get loaders with target scaling
        loaders, _ = get_loaders(batch_size=1024, scale_target=True)

        # 3. Assert loaders expose exact train statistics
        assert np.isclose(loaders.y_mean, expected_mean, atol=1e-4)
        assert np.isclose(loaders.y_std, expected_std, atol=1e-4)
        assert loaders.scale_target is True

        # 4. Confirm validation targets inside val_dataset are scaled using TRAIN statistics
        y_val_raw = y_val.to_numpy(dtype=np.float32)
        expected_scaled_val = (y_val_raw - expected_mean) / expected_std
        actual_scaled_val = loaders.val.dataset.y.numpy().ravel()
        assert np.allclose(actual_scaled_val, expected_scaled_val, atol=1e-4)

    def test_inverse_transform_round_trip(self):
        """Verify that scaling and inverse transforming CCME_Values round-trips exactly."""
        rng = np.random.default_rng(42)
        y_orig = rng.uniform(31.304, 100.0, size=1000).astype(np.float32)

        y_mean = float(np.mean(y_orig))
        y_std = float(np.std(y_orig))

        # Forward transform (standardization)
        y_scaled = (y_orig - y_mean) / y_std

        # Inverse transform
        y_reconstructed = y_scaled * y_std + y_mean

        assert np.allclose(y_orig, y_reconstructed, atol=1e-5)

    def test_scaled_and_unscaled_runs_report_metrics_on_same_scale(self):
        """Verify that scaled and unscaled runs report metrics on the original 0-100 scale."""
        # Run 1: scale_target = True
        set_seed(42)
        loaders_scaled, input_dim = get_loaders(
            batch_size=64,
            subsample=400,
            include_metadata=False,
            scale_numeric=True,
            scale_target=True,
            seed=42,
        )
        model_scaled = TinyPlaceholderMLP(input_dim=input_dim, hidden_dim=16)

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_scaled = {
                "epochs": 2,
                "lr": 1e-2,
                "optimizer": "adamw",
                "loss": "mse",
                "device": "cpu",
                "seed": 42,
                "checkpoint_dir": tmpdir,
                "model_name": "scaled.pt",
                "log_to_csv": False,
                "verbose": False,
            }
            hist_scaled, val_metrics_scaled = train(model_scaled, loaders_scaled, cfg_scaled)

            # Checkpoint contains target scaling metadata
            ckpt = torch.load(Path(tmpdir, "scaled.pt"), map_location="cpu")
            assert "y_mean" in ckpt
            assert "y_std" in ckpt
            assert ckpt["scale_target"] is True
            assert np.isclose(ckpt["y_mean"], loaders_scaled.y_mean, atol=1e-4)

        # Run 2: scale_target = False
        set_seed(42)
        loaders_unscaled, _ = get_loaders(
            batch_size=64,
            subsample=400,
            include_metadata=False,
            scale_numeric=True,
            scale_target=False,
            seed=42,
        )
        model_unscaled = TinyPlaceholderMLP(input_dim=input_dim, hidden_dim=16)

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_unscaled = {
                "epochs": 2,
                "lr": 1e-2,
                "optimizer": "adamw",
                "loss": "mse",
                "device": "cpu",
                "seed": 42,
                "checkpoint_dir": tmpdir,
                "model_name": "unscaled.pt",
                "log_to_csv": False,
                "verbose": False,
            }
            hist_unscaled, val_metrics_unscaled = train(
                model_unscaled, loaders_unscaled, cfg_unscaled
            )

        # 1. Validation loss scales differ: scaled val loss is O(1), unscaled is O(1000)
        assert hist_scaled["val_loss"][-1] < 50.0, (
            f"Scaled val_loss unexpectedly high: {hist_scaled['val_loss'][-1]}"
        )
        assert hist_unscaled["val_loss"][-1] > 500.0, (
            f"Unscaled val_loss unexpectedly low: {hist_unscaled['val_loss'][-1]}"
        )

        # 2. But logged metrics (MAE, RMSE) are on the ORIGINAL 0-100 scale for BOTH runs!
        assert 1.0 < val_metrics_scaled["mae"] < 100.0, (
            f"Scaled MAE not on original scale: {val_metrics_scaled['mae']}"
        )
        assert 1.0 < val_metrics_unscaled["mae"] < 100.0, (
            f"Unscaled MAE not on original scale: {val_metrics_unscaled['mae']}"
        )
        assert 1.0 < val_metrics_scaled["rmse"] < 100.0, (
            f"Scaled RMSE not on original scale: {val_metrics_scaled['rmse']}"
        )
        assert 1.0 < val_metrics_unscaled["rmse"] < 100.0, (
            f"Unscaled RMSE not on original scale: {val_metrics_unscaled['rmse']}"
        )
