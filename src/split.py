"""Reproducible grouped data splitting.   Owner: Parthraj Singh Bhati.

Creates and persists the project-wide train / validation / test split.

Rules:
  * Area-based grouping: observations from the same monitoring station ('Area')
    NEVER appear across train, validation, and test. Prevents spatial leakage.
  * Stratification by Country: where technically feasible, each country's areas
    are distributed to achieve target proportions.
  * Deterministic: uses SEED = 42 from src.config.
  * Shared: all models (RF, XGBoost, MLP, baselines) consume the EXACT SAME split.
  * Target proportions: 70% train / 15% validation / 15% test.
  * Persistence: saves assignments to data/splits/splits.parquet without duplicating
    the full dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import DATA_SPLITS, SEED, TARGET_COLUMN

SPLITS_PARQUET = DATA_SPLITS / "splits.parquet"
SPLITS_CSV = DATA_SPLITS / "splits.csv"


def make_grouped_split(
    df: pd.DataFrame,
    group_col: str = "Area",
    stratify_col: str = "Country",
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = SEED,
) -> pd.Series:
    """Generate a reproducible grouped split stratified by country.

    Allocates whole groups (unique values of ``group_col``) to train, val, or test
    such that:
      1. No group appears in more than one partition (strict zero group leakage).
      2. For each country, groups are allocated using greedy bin-packing to match
         the desired proportions as closely as integer group sizes allow.
      3. If a country contains only 1 group (e.g. China with 1 bay), that group is
         allocated to the largest partition (train) to preserve the zero-leakage guarantee.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least ``group_col`` and ``stratify_col``.
    group_col : str
        Grouping / site variable ('Area').
    stratify_col : str
        Stratification variable ('Country').
    train_frac, val_frac, test_frac : float
        Target partition fractions (must sum to 1.0).
    seed : int
        Random seed for shuffling group processing order.

    Returns
    -------
    pd.Series
        Series with index matching ``df.index``, values in {'train', 'val', 'test'}.
    """
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError(f"Fractions must sum to 1.0, got {train_frac} + {val_frac} + {test_frac}")

    rng = np.random.default_rng(seed)

    # Compute row counts per (Country, Area) group
    area_sizes = df.groupby([stratify_col, group_col], observed=True).size()

    area_to_split: dict[Any, str] = {}

    for _, sub in area_sizes.groupby(level=0, observed=True):
        areas = sub.droplevel(0)
        n_areas = len(areas)
        if n_areas == 0:
            continue

        # Shuffle area order deterministically before greedy assignment
        perm = rng.permutation(n_areas)
        shuffled_areas = areas.iloc[perm]

        total_rows = sub.sum()
        targets = {
            "train": total_rows * train_frac,
            "val": total_rows * val_frac,
            "test": total_rows * test_frac,
        }
        counts = {"train": 0, "val": 0, "test": 0}

        # Sort areas descending by size so large sites are placed first
        sorted_areas = shuffled_areas.sort_values(ascending=False)

        for area, count in sorted_areas.items():
            # Pick partition with largest remaining deficit
            best_split = max(targets.keys(), key=lambda s: targets[s] - counts[s])
            counts[best_split] += count
            area_to_split[area] = best_split

    splits = df[group_col].map(area_to_split).astype("category")
    splits.name = "split"
    return splits


def validate_splits(
    df: pd.DataFrame,
    splits: pd.Series,
    group_col: str = "Area",
    target_col: str = TARGET_COLUMN,
    min_train_frac: float = 0.60,
    max_train_frac: float = 0.80,
) -> dict[str, Any]:
    """Verify that split assignments satisfy all project integrity requirements.

    Raises AssertionError if any check fails:
      1. Exact row count matches.
      2. Zero row index overlap between splits.
      3. Strict zero Area group leakage.
      4. Target column presence and non-emptiness.
      5. Reasonable partition proportions.

    Returns summary metrics dictionary.
    """
    assert len(df) == len(splits), "DataFrame and splits length mismatch"
    assert set(splits.unique()).issubset({"train", "val", "test"}), "Invalid split label"

    train_idx = set(splits[splits == "train"].index)
    val_idx = set(splits[splits == "val"].index)
    test_idx = set(splits[splits == "test"].index)

    # 1. Zero row overlap
    assert len(train_idx.intersection(val_idx)) == 0, "Row overlap between train and val"
    assert len(train_idx.intersection(test_idx)) == 0, "Row overlap between train and test"
    assert len(val_idx.intersection(test_idx)) == 0, "Row overlap between val and test"

    # 2. Strict zero Area group leakage
    train_areas = set(df.loc[splits == "train", group_col].unique())
    val_areas = set(df.loc[splits == "val", group_col].unique())
    test_areas = set(df.loc[splits == "test", group_col].unique())

    area_leak_train_val = train_areas.intersection(val_areas)
    area_leak_train_test = train_areas.intersection(test_areas)
    area_leak_val_test = val_areas.intersection(test_areas)

    assert not area_leak_train_val, (
        f"Area leakage between train and val: {len(area_leak_train_val)} areas"
    )
    assert not area_leak_train_test, (
        f"Area leakage between train and test: {len(area_leak_train_test)} areas"
    )
    assert not area_leak_val_test, (
        f"Area leakage between val and test: {len(area_leak_val_test)} areas"
    )

    # 3. Target verification
    if target_col in df.columns:
        for s in ["train", "val", "test"]:
            sub_y = df.loc[splits == s, target_col]
            if len(sub_y) > 0:
                assert sub_y.notna().any(), (
                    f"Target column '{target_col}' has no valid values in {s}"
                )

    # 4. Proportions check
    props = splits.value_counts(normalize=True).to_dict()
    assert min_train_frac <= props.get("train", 0.0) <= max_train_frac, (
        f"Train proportion {props.get('train', 0.0):.3f} outside expected [{min_train_frac}, {max_train_frac}]"
    )

    return {
        "total_rows": len(df),
        "train_rows": len(train_idx),
        "val_rows": len(val_idx),
        "test_rows": len(test_idx),
        "proportions": {k: round(v, 4) for k, v in props.items()},
        "unique_areas_train": len(train_areas),
        "unique_areas_val": len(val_areas),
        "unique_areas_test": len(test_areas),
    }


def save_splits(
    splits: pd.Series,
    splits_dir: Path | str | None = None,
    save_csv: bool = False,
) -> Path:
    """Persist split assignments to parquet without duplicating the full dataset."""
    out_dir = Path(splits_dir) if splits_dir else DATA_SPLITS
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_file = out_dir / "splits.parquet"
    csv_file = out_dir / "splits.csv"
    split_df = pd.DataFrame({"split": splits.astype("category")}, index=splits.index)

    saved_file = parquet_file
    try:
        split_df.to_parquet(parquet_file, compression="zstd")
    except (ImportError, ValueError):
        split_df.to_csv(csv_file, index=True)
        saved_file = csv_file

    if save_csv and saved_file != csv_file:
        split_df.to_csv(csv_file, index=True)

    return saved_file


def load_splits(
    splits_dir: Path | str | None = None,
    df: pd.DataFrame | None = None,
) -> pd.Series:
    """Load persisted split assignments. If missing, generates and saves them.

    Returns
    -------
    pd.Series
        Series mapping row indices to partition ('train', 'val', 'test').
    """
    out_dir = Path(splits_dir) if splits_dir else DATA_SPLITS
    parquet_target = out_dir / "splits.parquet"
    csv_target = out_dir / "splits.csv"

    if parquet_target.exists():
        try:
            loaded = pd.read_parquet(parquet_target)["split"]
            if df is not None and len(loaded) != len(df):
                raise ValueError(
                    f"Persisted split size ({len(loaded)}) does not match dataframe ({len(df)}). "
                    "Regenerate splits using make_grouped_split() or python -m src.split."
                )
            return loaded
        except (ImportError, ValueError):
            if csv_target.exists():
                return pd.read_csv(csv_target, index_col=0)["split"].astype("category")
            raise
    elif csv_target.exists():
        return pd.read_csv(csv_target, index_col=0)["split"].astype("category")

    if df is None:
        from src.evaluate_api import load_data

        df = load_data()

    splits = make_grouped_split(df)
    validate_splits(df, splits)
    save_splits(splits, out_dir, save_csv=True)
    return splits


def get_split_indices(
    splits_dir: Path | str | None = None,
    df: pd.DataFrame | None = None,
) -> dict[str, np.ndarray]:
    """Return integer positional index arrays for each partition.

    Returns
    -------
    dict[str, np.ndarray]
        {'train': array, 'val': array, 'test': array}
    """
    splits = load_splits(splits_dir=splits_dir, df=df)
    categories = ["train", "val", "test"]
    return {cat: np.where(splits.values == cat)[0] for cat in categories}


def get_split_data(
    df: pd.DataFrame | None = None,
    splits_dir: Path | str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return the raw partitioned DataFrames (train_df, val_df, test_df)."""
    if df is None:
        from src.evaluate_api import load_data

        df = load_data()

    splits = load_splits(splits_dir=splits_dir, df=df)
    return (
        df.loc[splits == "train"].copy(),
        df.loc[splits == "val"].copy(),
        df.loc[splits == "test"].copy(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and validate project splits.")
    parser.add_argument("--save-csv", action="store_true", help="Also save splits.csv")
    parser.add_argument("--force", action="store_true", help="Regenerate even if exists")
    args = parser.parse_args()

    from src.evaluate_api import load_data

    print("Loading cleaned dataset...")
    df = load_data()
    print(f"Loaded {len(df):,} rows.")

    if SPLITS_PARQUET.exists() and not args.force:
        print(f"Splits already exist at {SPLITS_PARQUET}. Validating existing splits...")
        splits = load_splits(df=df)
    else:
        print("Generating grouped split by Area stratified by Country...")
        splits = make_grouped_split(df)
        save_splits(splits, save_csv=args.save_csv)
        print(f"Saved split assignments to {SPLITS_PARQUET}")

    metrics = validate_splits(df, splits)
    print("\n=== Split Validation Summary ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
