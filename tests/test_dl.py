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
import torch
import torch.nn as nn

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

        for l1, l2 in zip(run1_losses, run2_losses):
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
                csv_path=csv_path,
            )

            assert out.exists()
            content = out.read_text()
            assert "test_experiment" in content
            assert "0.45" in content
            assert "0.85" in content
            assert "0.99" in content
