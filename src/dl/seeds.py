"""Reproducibility utilities for deep learning.

Provides deterministic seed setting across Python, NumPy, and PyTorch (CPU/CUDA/MPS).
Owner: Parthraj Singh Bhati.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch

from src.config import SEED


def set_seed(seed: int = SEED, deterministic: bool = True) -> None:
    """Fix random seeds across Python, NumPy, and PyTorch.

    Ensures reproducible dataset shuffling, parameter initialization,
    and mini-batch batching across all deep learning experiments.

    Parameters
    ----------
    seed : int, default 42
        Integer random seed (defaults to project-wide SEED from src.config).
    deterministic : bool, default True
        Whether to configure PyTorch for deterministic algorithms where practical.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
        try:
            torch.mps.manual_seed(seed)
        except Exception:
            pass

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn to preserve seed determinism in multiprocessing."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
