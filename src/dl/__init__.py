"""Deep learning package for Phase 2 Water Quality Prediction.

Provides:
  * Seeds & reproducibility: set_seed, seed_worker
  * Dataset & DataLoader: WaterQualityDataset, LoadersBundle, get_loaders
  * Models: WaterQualityMLP, create_mlp, get_activation
  * Training & Evaluation: train, evaluate_loader, log_run, get_loss_function, get_device

Contributors:
  * Parthraj Singh Bhati (Data loaders, training loop, seeds)
  * Himanshu Pandey (Model architecture, base MLP, ablations)
"""

from __future__ import annotations

from src.dl.data import LoadersBundle, WaterQualityDataset, get_loaders
from src.dl.model import WaterQualityMLP, create_mlp, get_activation
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
    "WaterQualityMLP",
    "create_mlp",
    "get_activation",
    "train",
    "evaluate_loader",
    "log_run",
    "get_loss_function",
    "get_device",
]
