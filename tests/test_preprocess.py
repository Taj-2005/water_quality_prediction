"""Unit tests for the train-only fitted preprocessing and partial-measurement masking.

Verifies:
  * Strict target separation: 'CCME_Values' is never in X.
  * Direct label leakage: 'CCME_WQI' is never in X.
  * Spatial leakage: 'Area' is never in X.
  * Train-only fitting: imputer and scaling parameters learned strictly on X_train.
  * Upstream imputation flags excluded from primary X by default.
  * Partial-measurement masking: 8/8, 7/8, 6/8, 4/8.
  * Dependency propagation: derived features dependent on masked inputs are cleanly dropped.
  * Immutability: masking does not alter the input DataFrame.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.clean import MEASUREMENTS
from src.preprocess import (
    PRIMARY_MEASUREMENTS,
    WaterQualityPreprocessor,
    extract_features_target,
    mask_features,
    validate_no_leakage,
)


def make_sample_features(n: int = 50) -> pd.DataFrame:
    """Generate sample dataframe containing raw measurements, metadata, flags, and target."""
    rng = np.random.default_rng(42)
    data: dict[str, Any] = {
        "Country": ["England"] * (n // 2) + ["USA"] * (n - n // 2),
        "Area": [f"SITE-{i}" for i in range(n)],
        "Waterbody Type": ["River"] * n,
        "Date": ["01-01-2020"] * n,
        "CCME_Values": rng.uniform(40, 100, n),
        "CCME_WQI": ["Good"] * n,
    }
    for m in MEASUREMENTS:
        data[m] = rng.uniform(1.0, 15.0, n)

    # Upstream imputation flags
    for m in MEASUREMENTS:
        data[f"{m}__imputed_upstream"] = [False] * n
    data["n_imputed_upstream"] = [0] * n
    data["flag_any_imputed_upstream"] = [False] * n

    return pd.DataFrame(data)


class TestTargetSeparationAndLeakage:
    def test_extract_features_target_excludes_target_and_labels(self):
        df = make_sample_features(20)
        X, y = extract_features_target(df)
        assert "CCME_Values" not in X.columns
        assert "CCME_WQI" not in X.columns
        assert len(y) == len(X)
        assert np.array_equal(y.values, df["CCME_Values"].values)

    def test_validate_no_leakage_raises_on_target(self):
        df = make_sample_features(10)
        with pytest.raises(ValueError, match="CRITICAL LEAKAGE: Target column"):
            validate_no_leakage(df)

    def test_validate_no_leakage_raises_on_label(self):
        df = make_sample_features(10).drop(columns=["CCME_Values"])
        with pytest.raises(ValueError, match="CRITICAL LEAKAGE: Target classification label"):
            validate_no_leakage(df)

    def test_validate_no_leakage_raises_on_area(self):
        df = make_sample_features(10).drop(columns=["CCME_Values", "CCME_WQI"])
        with pytest.raises(ValueError, match="CRITICAL LEAKAGE: Site identifier 'Area'"):
            validate_no_leakage(df)

    def test_upstream_flags_excluded_by_default(self):
        df = make_sample_features(10).drop(columns=["CCME_Values", "CCME_WQI", "Area"])
        with pytest.raises(ValueError, match="Upstream imputation flags found"):
            validate_no_leakage(df, allow_imputation_flags=False)


class TestTrainOnlyFitting:
    def test_preprocessor_raises_if_transform_called_before_fit(self):
        p = WaterQualityPreprocessor()
        df = make_sample_features(10)
        X, _ = extract_features_target(df)
        with pytest.raises(RuntimeError, match="must be fit"):
            p.transform(X)

    def test_preprocessor_fits_on_train_only(self):
        train_df = make_sample_features(30)
        test_df = make_sample_features(20)

        # Inject extreme outlier in test to verify it doesn't affect learned medians
        test_df.loc[0, "Ammonia (mg/l)"] = 9999.0

        X_train, _ = extract_features_target(train_df)
        X_test, _ = extract_features_target(test_df)

        p = WaterQualityPreprocessor(scale_numeric=True)
        p.fit(X_train)

        # Train medians must match train_df, not test_df
        expected_median = float(train_df["Ammonia (mg/l)"].median())
        assert np.isclose(p.numeric_medians_["Ammonia (mg/l)"], expected_median)

        # Transform both
        X_train_trans = p.transform(X_train)
        X_test_trans = p.transform(X_test)

        assert len(X_train_trans) == 30
        assert len(X_test_trans) == 20
        assert "CCME_Values" not in X_train_trans.columns
        assert "CCME_Values" not in X_test_trans.columns

    def test_imputation_flags_included_when_requested(self):
        train_df = make_sample_features(20)
        X_train, _ = extract_features_target(train_df, include_imputation_flags=True)
        p = WaterQualityPreprocessor(include_imputation_flags=True)
        p.fit(X_train)
        out = p.transform(X_train)
        assert "n_imputed_upstream" in out.columns


class TestPartialMeasurementMasking:
    def test_mask_8_of_8_keeps_all(self):
        df = make_sample_features(10)
        X, _ = extract_features_target(df)
        p = WaterQualityPreprocessor().fit(X)
        X_trans = p.transform(X)

        masked = mask_features(X_trans, available_measurements="8/8")
        for m in PRIMARY_MEASUREMENTS:
            assert m in masked.columns

    def test_mask_4_of_8_drops_unselected_measurements(self):
        df = make_sample_features(10)
        X, _ = extract_features_target(df)
        p = WaterQualityPreprocessor().fit(X)
        X_trans = p.transform(X)

        masked = mask_features(X_trans, available_measurements="4/8")
        assert "Orthophosphate (mg/l)" not in masked.columns
        assert "Biochemical Oxygen Demand (mg/l)" not in masked.columns
        assert "pH (ph units)" in masked.columns
        assert "Dissolved Oxygen (mg/l)" in masked.columns

    def test_derived_features_dropped_when_dependency_masked(self):
        """If BOD is masked, BOD_DO_ratio must also be removed to prevent indirect leakage."""
        df = make_sample_features(10)
        X, _ = extract_features_target(df)
        p = WaterQualityPreprocessor().fit(X)
        X_trans = p.transform(X).copy()

        # Simulate candidate engineered interaction feature
        X_trans["BOD_DO_ratio"] = (
            X_trans["Biochemical Oxygen Demand (mg/l)"] / X_trans["Dissolved Oxygen (mg/l)"]
        )
        X_trans["N_P_ratio"] = X_trans["Nitrogen (mg/l)"] / X_trans["Orthophosphate (mg/l)"]

        # Mask BOD
        available = [c for c in PRIMARY_MEASUREMENTS if c != "Biochemical Oxygen Demand (mg/l)"]
        masked = mask_features(X_trans, available_measurements=available)

        assert "Biochemical Oxygen Demand (mg/l)" not in masked.columns
        assert "BOD_DO_ratio" not in masked.columns, "BOD_DO_ratio leaked masked BOD information!"
        # N_P_ratio still has both Nitrogen and Orthophosphate available, so it stays
        assert "N_P_ratio" in masked.columns

    def test_masking_does_not_mutate_input_frame(self):
        df = make_sample_features(10)
        X, _ = extract_features_target(df)
        p = WaterQualityPreprocessor().fit(X)
        X_trans = p.transform(X)

        cols_before = list(X_trans.columns)
        _ = mask_features(X_trans, available_measurements="4/8")
        cols_after = list(X_trans.columns)
        assert cols_before == cols_after
