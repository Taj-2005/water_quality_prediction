"""PyTorch dataset and reproducible data loaders for Phase 2 deep learning.

Enforces:
  * Zero split regeneration: loads saved Phase 1 Area-grouped split indices.
  * Train-only preprocessing: numeric scaling and categorical encoding fitted strictly on train.
  * Deterministic data loading: seeded PyTorch DataLoader generator and worker init functions.

Owner: Parthraj Singh Bhati.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.config import DATA_SPLITS, SEED
from src.dl.seeds import seed_worker, set_seed
from src.evaluate_api import get_splits
from src.split import SPLITS_CSV, SPLITS_PARQUET


class WaterQualityDataset(Dataset):
    """PyTorch Dataset wrapping water quality features and CCME_Values target.

    Parameters
    ----------
    X : np.ndarray | pd.DataFrame | torch.Tensor
        Feature matrix.
    y : np.ndarray | pd.Series | torch.Tensor | None, optional
        Continuous CCME_Values target vector.
    """

    def __init__(
        self,
        X: np.ndarray | pd.DataFrame | torch.Tensor,
        y: np.ndarray | pd.Series | torch.Tensor | None = None,
    ) -> None:
        if isinstance(X, pd.DataFrame):
            X_arr = X.to_numpy(dtype=np.float32)
        elif isinstance(X, np.ndarray):
            X_arr = X.astype(np.float32)
        elif isinstance(X, torch.Tensor):
            X_arr = X.detach().cpu().to(torch.float32).numpy()
        else:
            X_arr = np.asarray(X, dtype=np.float32)

        self.X = torch.from_numpy(X_arr)

        if y is not None:
            if isinstance(y, (pd.Series, pd.DataFrame)):
                y_arr = y.to_numpy(dtype=np.float32)
            elif isinstance(y, np.ndarray):
                y_arr = y.astype(np.float32)
            elif isinstance(y, torch.Tensor):
                y_arr = y.detach().cpu().to(torch.float32).numpy()
            else:
                y_arr = np.asarray(y, dtype=np.float32)

            if y_arr.ndim == 1:
                y_arr = y_arr[:, np.newaxis]
            self.y: torch.Tensor | None = torch.from_numpy(y_arr)
        else:
            self.y = None

        if self.y is not None and len(self.X) != len(self.y):
            raise ValueError(
                f"Length mismatch: X has {len(self.X)} rows, y has {len(self.y)} rows."
            )

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


class LoadersBundle(dict):
    """Container holding train, validation, and test PyTorch DataLoaders.

    Supports dictionary lookup (`loaders['train']`), attribute access (`loaders.train`),
    and 3-tuple iterable unpacking:
        train_loader, val_loader, test_loader = loaders
    """

    def __init__(
        self,
        train: DataLoader,
        val: DataLoader,
        test: DataLoader,
    ) -> None:
        super().__init__({"train": train, "val": val, "test": test})
        self.train = train
        self.val = val
        self.test = test

    def __iter__(self) -> Iterator[DataLoader]:
        yield self.train
        yield self.val
        yield self.test


def get_loaders(
    batch_size: int = 256,
    seed: int = SEED,
    splits_dir: Path | str | None = None,
    include_metadata: bool = True,
    scale_numeric: bool = True,
    partial_scenario: str | Sequence[str] | None = None,
    subsample: int | float | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> tuple[LoadersBundle, int]:
    """Load persisted Phase 1 splits and return PyTorch DataLoaders with input dimension.

    Guarantees:
      1. Loads saved Phase 1 split indices (never regenerates a split).
      2. Preprocessing pipeline fitted STRICTLY on X_train.
      3. Categorical metadata (if present) one-hot encoded strictly on X_train.
      4. Fully reproducible DataLoader with seeded generator and worker seeding.

    Parameters
    ----------
    batch_size : int, default 256
        Batch size for training and evaluation loaders.
    seed : int, default 42
        Random seed for shuffling and DataLoader generators.
    splits_dir : Path | str | None, optional
        Directory containing persisted splits (defaults to data/splits/).
    include_metadata : bool, default True
        Whether to include non-chemical metadata covariates (Country, Waterbody Type).
    scale_numeric : bool, default True
        Whether to z-score standardize numeric measurements using train statistics.
    partial_scenario : str | Sequence[str] | None, optional
        Partial measurement scenario ('8/8', '7/8', '6/8', '4/8').
    subsample : int | float | None, optional
        Deterministic subsample of training partition for fast debugging / smoke tests.
    num_workers : int, default 0
        Number of subprocess workers for PyTorch DataLoaders.
    pin_memory : bool, default False
        Whether to pin memory in DataLoader for faster CUDA transfer.

    Returns
    -------
    tuple[LoadersBundle, int]
        (loaders_bundle, input_dim) where loaders_bundle holds train, val, and test loaders,
        and input_dim is the number of features in X.
    """
    # 1. Enforce that saved Phase 1 split files exist
    target_splits_dir = Path(splits_dir) if splits_dir else DATA_SPLITS
    parquet_path = target_splits_dir / "splits.parquet"
    csv_path = target_splits_dir / "splits.csv"
    if not parquet_path.exists() and not csv_path.exists():
        raise FileNotFoundError(
            f"Phase 1 saved split indices not found in {target_splits_dir}. "
            "Never regenerate splits in Phase 2; ensure data/splits/splits.parquet exists."
        )

    # 2. Fix random seeds
    set_seed(seed)

    # 3. Load preprocessed splits through Phase 1 evaluate_api
    X_train, X_val, X_test, y_train, y_val, y_test = get_splits(
        splits_dir=target_splits_dir,
        include_metadata=include_metadata,
        include_imputation_flags=False,
        scale_numeric=scale_numeric,
        partial_scenario=partial_scenario,
    )

    # 4. Optional deterministic training subsample (train only)
    if subsample is not None:
        if isinstance(subsample, float):
            if not 0.0 < subsample <= 1.0:
                raise ValueError(f"subsample float must be in (0, 1], got {subsample}")
            n_sub = int(len(X_train) * subsample)
        else:
            n_sub = int(min(subsample, len(X_train)))

        rng = np.random.default_rng(seed)
        sub_indices = rng.choice(len(X_train), size=n_sub, replace=False)
        sub_indices.sort()
        X_train = X_train.iloc[sub_indices].reset_index(drop=True)
        y_train = y_train.iloc[sub_indices].reset_index(drop=True)

    # 5. One-hot encode categorical columns if present (fitted STRICTLY on X_train)
    cat_cols = [c for c in X_train.columns if not pd.api.types.is_numeric_dtype(X_train[c])]
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    if cat_cols:
        from sklearn.preprocessing import OneHotEncoder

        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore", dtype=np.float32)
        encoder.fit(X_train[cat_cols].astype(str))

        X_train_num = X_train[num_cols].to_numpy(dtype=np.float32)
        X_val_num = X_val[num_cols].to_numpy(dtype=np.float32)
        X_test_num = X_test[num_cols].to_numpy(dtype=np.float32)

        X_train_cat = encoder.transform(X_train[cat_cols].astype(str))
        X_val_cat = encoder.transform(X_val[cat_cols].astype(str))
        X_test_cat = encoder.transform(X_test[cat_cols].astype(str))

        X_train_arr = np.hstack([X_train_num, X_train_cat])
        X_val_arr = np.hstack([X_val_num, X_val_cat])
        X_test_arr = np.hstack([X_test_num, X_test_cat])
    else:
        X_train_arr = X_train.to_numpy(dtype=np.float32)
        X_val_arr = X_val.to_numpy(dtype=np.float32)
        X_test_arr = X_test.to_numpy(dtype=np.float32)

    y_train_arr = y_train.to_numpy(dtype=np.float32)
    y_val_arr = y_val.to_numpy(dtype=np.float32)
    y_test_arr = y_test.to_numpy(dtype=np.float32)

    # 6. Wrap into PyTorch Datasets
    train_dataset = WaterQualityDataset(X_train_arr, y_train_arr)
    val_dataset = WaterQualityDataset(X_val_arr, y_val_arr)
    test_dataset = WaterQualityDataset(X_test_arr, y_test_arr)

    # 7. Seeded generator for DataLoader shuffle determinism
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
        worker_init_fn=seed_worker,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    input_dim = int(X_train_arr.shape[1])
    loaders = LoadersBundle(train_loader, val_loader, test_loader)
    return loaders, input_dim
