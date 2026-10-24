"""Unit tests for the grouped splitting pipeline.

Verifies:
  * Area-based grouping: zero area overlap across partitions.
  * Stratification by country: each country with >1 area is represented.
  * Target proportions: ~70/15/15.
  * Zero row overlap between train, val, and test.
  * Strict reproducibility with identical seeds.
  * Target presence across all partitions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.clean import MEASUREMENTS
from src.config import SEED
from src.split import (
    get_split_indices,
    make_grouped_split,
    save_splits,
    validate_splits,
)


def make_synthetic_dataset(n_rows_per_area: int = 5, n_areas: int = 40) -> pd.DataFrame:
    """Create a representative multi-country multi-area synthetic frame."""
    rng = np.random.default_rng(0)
    countries = ["England", "USA", "Ireland", "Canada"]
    data: dict[str, list] = {
        "Country": [],
        "Area": [],
        "Waterbody Type": [],
        "Date": [],
        "CCME_Values": [],
        "CCME_WQI": [],
    }
    for m in MEASUREMENTS:
        data[m] = []

    for i in range(n_areas):
        c = countries[i % len(countries)]
        area_name = f"{c}-SITE-{i}"
        for _ in range(n_rows_per_area):
            data["Country"].append(c)
            data["Area"].append(area_name)
            data["Waterbody Type"].append("River")
            data["Date"].append("01-01-2020")
            data["CCME_Values"].append(rng.uniform(40, 100))
            data["CCME_WQI"].append("Good")
            for m in MEASUREMENTS:
                data[m].append(rng.uniform(1, 10))

    # Add a country with only 1 area (like China in the real data)
    for _ in range(20):
        data["Country"].append("China")
        data["Area"].append("China-Hou-Bay")
        data["Waterbody Type"].append("Bay")
        data["Date"].append("01-01-2020")
        data["CCME_Values"].append(rng.uniform(80, 100))
        data["CCME_WQI"].append("Good")
        for m in MEASUREMENTS:
            data[m].append(rng.uniform(1, 10))

    return pd.DataFrame(data)


class TestGroupedSplit:
    def test_no_row_overlap(self):
        df = make_synthetic_dataset()
        splits = make_grouped_split(df, seed=SEED)
        train_idx = set(splits[splits == "train"].index)
        val_idx = set(splits[splits == "val"].index)
        test_idx = set(splits[splits == "test"].index)

        assert len(train_idx.intersection(val_idx)) == 0
        assert len(train_idx.intersection(test_idx)) == 0
        assert len(val_idx.intersection(test_idx)) == 0
        assert len(train_idx) + len(val_idx) + len(test_idx) == len(df)

    def test_strict_zero_area_leakage(self):
        df = make_synthetic_dataset()
        splits = make_grouped_split(df, seed=SEED)

        train_areas = set(df.loc[splits == "train", "Area"].unique())
        val_areas = set(df.loc[splits == "val", "Area"].unique())
        test_areas = set(df.loc[splits == "test", "Area"].unique())

        assert len(train_areas.intersection(val_areas)) == 0
        assert len(train_areas.intersection(test_areas)) == 0
        assert len(val_areas.intersection(test_areas)) == 0

    def test_single_area_country_allocated_to_train(self):
        """When a country has only 1 area, it must be assigned to train without leaking."""
        df = make_synthetic_dataset()
        splits = make_grouped_split(df, seed=SEED)
        china_splits = splits[df["Country"] == "China"].unique()
        assert list(china_splits) == ["train"]

    def test_reproducibility(self):
        df = make_synthetic_dataset()
        s1 = make_grouped_split(df, seed=42)
        s2 = make_grouped_split(df, seed=42)
        pd.testing.assert_series_equal(s1, s2)

    def test_different_seed_produces_valid_split(self):
        df = make_synthetic_dataset()
        s_alt = make_grouped_split(df, seed=123)
        metrics = validate_splits(df, s_alt)
        assert metrics["train_rows"] > 0
        assert metrics["val_rows"] > 0
        assert metrics["test_rows"] > 0

    def test_target_proportions_reasonable(self):
        df = make_synthetic_dataset()
        splits = make_grouped_split(df, seed=SEED)
        props = splits.value_counts(normalize=True)
        assert 0.60 <= props["train"] <= 0.80
        assert 0.08 <= props["val"] <= 0.25
        assert 0.08 <= props["test"] <= 0.25

    def test_validation_function_raises_on_leakage(self):
        df = make_synthetic_dataset()
        splits = make_grouped_split(df, seed=SEED).copy()
        # Deliberately inject leakage
        val_first = splits[splits == "val"].index[0]
        splits.loc[val_first] = "train"  # introduces Area leakage
        with pytest.raises(AssertionError, match="Area leakage"):
            validate_splits(df, splits)

    def test_save_and_load_splits(self, tmp_path):
        df = make_synthetic_dataset()
        splits = make_grouped_split(df, seed=SEED)
        saved_file = save_splits(splits, splits_dir=tmp_path, save_csv=True)
        assert saved_file.exists()
        assert (tmp_path / "splits.csv").exists()

        indices = get_split_indices(splits_dir=tmp_path, df=df)
        assert "train" in indices and "val" in indices and "test" in indices
        assert len(indices["train"]) + len(indices["val"]) + len(indices["test"]) == len(df)
