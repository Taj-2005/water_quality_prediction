"""Deep learning package for Phase 2 Water Quality Prediction.

Provides:
  * Seeds & reproducibility: set_seed, seed_worker
  * Dataset & DataLoader: WaterQualityDataset, LoadersBundle, get_loaders
  * Training & Evaluation: train, evaluate_loader, log_run, get_loss_function, get_device

Owner: Parthraj Singh Bhati.
"""

from __future__ import annotations

from src.dl.data import LoadersBundle, WaterQualityDataset, get_loaders
from src.dl.nn_preprocess import build_nn_preprocessor, transform_splits
from src.dl.seeds import seed_worker, set_seed
from src.dl.train import evaluate_loader, get_device, get_loss_function, log_run, train

__all__ = [
    "build_nn_preprocessor",
    "transform_splits",
    "set_seed",
    "seed_worker",
    "WaterQualityDataset",
    "LoadersBundle",
    "get_loaders",
    "train",
    "evaluate_loader",
    "log_run",
    "get_loss_function",
    "get_device",
]
