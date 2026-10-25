"""Train-only fitted preprocessing and partial-measurement masking.   Owner: Parthraj Singh Bhati.

Provides:
  * Strict target separation: 'CCME_Values' is isolated as y and never enters X.
  * Train-only fitting: any statistic (imputer medians, category encodings, scalers)
    is learned STRICTLY from X_train and applied to val/test without refitting.
  * Primary feature selection: the 8 Ireland standard measurements + approved non-leaky
    metadata (Country, Waterbody Type). Upstream imputation flags and site IDs are
    excluded by default.
  * Partial-measurement masking: controlled interface for 8/8, 7/8, 6/8, 4/8 scenarios
    that automatically drops any derived features dependent on masked inputs.
  * Leakage validation: programmatic checks guaranteeing zero target or group leakage.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from sklearn.base import BaseEstimator, TransformerMixin
except ImportError:

    class BaseEstimator:
        pass

    class TransformerMixin:
        def fit_transform(self, X, y=None, **fit_params):
            return self.fit(X, y, **fit_params).transform(X)


from src.clean import MEASUREMENTS
from src.config import TARGET_COLUMN
from src.split import get_split_data

# Approved feature sets
PRIMARY_MEASUREMENTS = list(MEASUREMENTS)

METADATA_FEATURES = ["Country", "Waterbody Type"]

EXCLUDED_COLUMNS = [
    TARGET_COLUMN,
    "CCME_WQI",  # Post-hoc binned target label (direct label leakage)
    "Area",  # Monitoring station ID (grouping key, prevents site memorization)
    "Date",  # Raw string date
    "Date_parsed",
]

# Standard Partial Measurement Scenarios for the project ablation
# 8/8: Full Ireland standard suite
# 7/8: Missing Orthophosphate (the dominant formula driver)
# 6/8: Missing Orthophosphate and BOD (heavy organics)
# 4/8: Core physical-chemical suite (pH, Dissolved Oxygen, Temperature, Nitrate)
PARTIAL_SCENARIOS: dict[str, list[str]] = {
    "8/8": PRIMARY_MEASUREMENTS,
    "7/8": [c for c in PRIMARY_MEASUREMENTS if c != "Orthophosphate (mg/l)"],
    "6/8": [
        c
        for c in PRIMARY_MEASUREMENTS
        if c not in ("Orthophosphate (mg/l)", "Biochemical Oxygen Demand (mg/l)")
    ],
    "4/8": [
        "pH (ph units)",
        "Dissolved Oxygen (mg/l)",
        "Temperature (cel)",
        "Nitrate (mg/l)",
    ],
}

# Default dependency registry for candidate engineered features (from Himanshu's PR #6 spec).
# If a measurement is masked, all dependent derived columns MUST be masked simultaneously.
DEFAULT_FEATURE_DEPENDENCIES: dict[str, list[str]] = {
    "N_P_ratio": ["Nitrogen (mg/l)", "Orthophosphate (mg/l)"],
    "Inorganic_N_ratio": ["Ammonia (mg/l)", "Nitrate (mg/l)"],
    "Organic_N_fraction": ["Nitrogen (mg/l)", "Nitrate (mg/l)", "Ammonia (mg/l)"],
    "BOD_DO_ratio": ["Biochemical Oxygen Demand (mg/l)", "Dissolved Oxygen (mg/l)"],
    "log_Ammonia": ["Ammonia (mg/l)"],
    "log_BOD": ["Biochemical Oxygen Demand (mg/l)"],
    "log_Orthophosphate": ["Orthophosphate (mg/l)"],
    "log_Nitrate": ["Nitrate (mg/l)"],
    "log_Nitrogen": ["Nitrogen (mg/l)"],
}


def validate_no_leakage(
    X: pd.DataFrame,
    y: pd.Series | None = None,
    allow_imputation_flags: bool = False,
) -> None:
    """Validate that feature matrix X contains zero direct or indirect target/site leakage.

    Raises
    ------
    ValueError
        If any leakage rule is violated.
    """
    cols = set(X.columns)

    if TARGET_COLUMN in cols:
        raise ValueError(
            f"CRITICAL LEAKAGE: Target column '{TARGET_COLUMN}' found inside feature matrix X!"
        )

    if "CCME_WQI" in cols:
        raise ValueError(
            "CRITICAL LEAKAGE: Target classification label 'CCME_WQI' found inside feature matrix X!"
        )

    if "Area" in cols:
        raise ValueError(
            "CRITICAL LEAKAGE: Site identifier 'Area' found inside feature matrix X! "
            "Area is the spatial grouping key and must not be used as an input predictor."
        )

    if not allow_imputation_flags:
        leak_flags = [
            c
            for c in cols
            if c.endswith("__imputed_upstream")
            or c == "n_imputed_upstream"
            or c == "flag_any_imputed_upstream"
        ]
        if leak_flags:
            raise ValueError(
                f"Upstream imputation flags found in primary feature matrix: {leak_flags}. "
                "These are confounded with Country (England) and must only be included in explicit ablations."
            )

    if y is not None:
        if len(X) != len(y):
            raise ValueError(f"Length mismatch: X has {len(X)} rows, y has {len(y)} rows.")
        if not np.array_equal(X.index, y.index):
            raise ValueError("Index mismatch between X and y.")


def mask_features(
    X: pd.DataFrame,
    available_measurements: str | Sequence[str] = "8/8",
    feature_dependencies: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Controlled partial-measurement masking for ML experiments.

    Simulates missing parameter availability at inference time without modifying the
    underlying clean dataset.

    Parameters
    ----------
    X : pd.DataFrame
        Input feature matrix.
    available_measurements : str or Sequence[str]
        Either a preset key ('8/8', '7/8', '6/8', '4/8') or an explicit list of
        measurement column names that remain available.
    feature_dependencies : dict[str, list[str]], optional
        Mapping of derived feature names to their required measurement inputs.
        Defaults to DEFAULT_FEATURE_DEPENDENCIES.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with unavailable measurements and any dependent derived
        features cleanly removed.
    """
    if isinstance(available_measurements, str):
        if available_measurements not in PARTIAL_SCENARIOS:
            raise ValueError(
                f"Unknown partial scenario '{available_measurements}'. "
                f"Valid presets: {list(PARTIAL_SCENARIOS.keys())}"
            )
        active_measurements = set(PARTIAL_SCENARIOS[available_measurements])
    else:
        active_measurements = set(available_measurements)

    deps = dict(DEFAULT_FEATURE_DEPENDENCIES)
    if feature_dependencies is not None:
        deps.update(feature_dependencies)

    # Determine which primary measurements are masked
    masked_measurements = set(PRIMARY_MEASUREMENTS) - active_measurements

    drop_cols: list[str] = []

    # 1. Drop masked primary measurements
    for col in masked_measurements:
        if col in X.columns:
            drop_cols.append(col)

    # 2. Drop any derived feature that depends on ANY masked measurement
    for derived_col, required_inputs in deps.items():
        if derived_col in X.columns:
            if any(req in masked_measurements for req in required_inputs):
                drop_cols.append(derived_col)

    return X.drop(columns=drop_cols, errors="ignore")


class WaterQualityPreprocessor(BaseEstimator, TransformerMixin):
    """Train-only fitted preprocessing transformer for water quality data.

    Learns imputation medians, categorical categories, and optional numeric scaling
    strictly from training partitions.
    """

    def __init__(
        self,
        include_metadata: bool = True,
        include_imputation_flags: bool = False,
        scale_numeric: bool = False,
    ) -> None:
        self.include_metadata = include_metadata
        self.include_imputation_flags = include_imputation_flags
        self.scale_numeric = scale_numeric

        # Learned state (train only)
        self.numeric_columns_: list[str] = []
        self.categorical_columns_: list[str] = []
        self.numeric_medians_: dict[str, float] = {}
        self.numeric_scales_: dict[str, tuple[float, float]] = {}  # (mean, std)
        self.categorical_categories_: dict[str, list[str]] = {}
        self.feature_names_out_: list[str] = []
        self.is_fitted_: bool = False

    def _select_raw_columns(self, df: pd.DataFrame) -> list[str]:
        cols = [c for c in PRIMARY_MEASUREMENTS if c in df.columns]

        if self.include_metadata:
            cols.extend([c for c in METADATA_FEATURES if c in df.columns])

        if self.include_imputation_flags:
            flag_cols = [
                c
                for c in df.columns
                if c.endswith("__imputed_upstream")
                or c in ("n_imputed_upstream", "flag_any_imputed_upstream")
            ]
            cols.extend(flag_cols)

        return cols

    def fit(self, X: pd.DataFrame, y: Any = None) -> WaterQualityPreprocessor:
        """Fit preprocessor strictly on training data."""
        validate_no_leakage(X, y=y, allow_imputation_flags=self.include_imputation_flags)

        selected_cols = self._select_raw_columns(X)
        sub_X = X[selected_cols]

        self.numeric_columns_ = [
            c for c in selected_cols if pd.api.types.is_numeric_dtype(sub_X[c])
        ]
        self.categorical_columns_ = [c for c in selected_cols if c not in self.numeric_columns_]

        # Learn train medians (for fallback imputation)
        for col in self.numeric_columns_:
            median_val = float(sub_X[col].median())
            self.numeric_medians_[col] = median_val
            if self.scale_numeric:
                mean_val = float(sub_X[col].mean())
                std_val = float(sub_X[col].std())
                if std_val == 0.0 or np.isnan(std_val):
                    std_val = 1.0
                self.numeric_scales_[col] = (mean_val, std_val)

        # Learn train categorical levels
        for col in self.categorical_columns_:
            cats = [str(x) for x in sub_X[col].dropna().unique()]
            self.categorical_categories_[col] = cats

        self.feature_names_out_ = list(selected_cols)
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply train-fitted transformations to val, test, or inference data."""
        if not self.is_fitted_:
            raise RuntimeError(
                "WaterQualityPreprocessor must be fit on X_train before transform()."
            )

        validate_no_leakage(X, allow_imputation_flags=self.include_imputation_flags)

        selected_cols = [c for c in self.feature_names_out_ if c in X.columns]
        out_df = X[selected_cols].copy()

        # Apply train medians for any missing values
        for col in self.numeric_columns_:
            if col in out_df.columns:
                median = self.numeric_medians_.get(col, 0.0)
                out_df[col] = out_df[col].fillna(median)
                if self.scale_numeric and col in self.numeric_scales_:
                    mean, std = self.numeric_scales_[col]
                    out_df[col] = (out_df[col] - mean) / std

        # Categorical handling: preserve categories learned on train
        for col in self.categorical_columns_:
            if col in out_df.columns:
                cats = self.categorical_categories_.get(col, [])
                out_df[col] = pd.Categorical(out_df[col].astype(str), categories=cats)

        return out_df

    def get_feature_names_out(self) -> list[str]:
        if not self.is_fitted_:
            raise RuntimeError("Preprocessor not fitted.")
        return list(self.feature_names_out_)


def extract_features_target(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    drop_excluded: bool = True,
    include_imputation_flags: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """Strict separation of raw features X and ground-truth target y.

    Guarantees that y is cleanly decoupled and excluded columns (target,
    labels, Area, Date, flags) are stripped from X by default.
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataframe.")

    y = df[target_col].copy()

    drop_cols = [target_col]
    if drop_excluded:
        for c in EXCLUDED_COLUMNS:
            if c in df.columns and c not in drop_cols:
                drop_cols.append(c)

        if not include_imputation_flags:
            flag_cols = [
                c
                for c in df.columns
                if c.endswith("__imputed_upstream")
                or c in ("n_imputed_upstream", "flag_any_imputed_upstream")
                or c.startswith("flag_")
            ]
            for c in flag_cols:
                if c not in drop_cols:
                    drop_cols.append(c)

    X = df.drop(columns=drop_cols).copy()
    validate_no_leakage(X, y, allow_imputation_flags=include_imputation_flags)
    return X, y


def get_preprocessed_splits(
    df: pd.DataFrame | None = None,
    splits_dir: Path | str | None = None,
    include_metadata: bool = True,
    include_imputation_flags: bool = False,
    scale_numeric: bool = False,
    partial_scenario: str | Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """End-to-end reproducible pipeline returning (X_train, X_val, X_test, y_train, y_val, y_test).

    Enforces the core scientific guarantee:
      1. Splits loaded from shared data/splits/ (Area-grouped).
      2. Preprocessor fitted STRICTLY on X_train.
      3. Validation and test transformed without refitting.
      4. Partial-measurement masking applied consistently if requested.
    """
    train_df, val_df, test_df = get_split_data(df=df, splits_dir=splits_dir)

    # 1. Target separation
    X_train_raw, y_train = extract_features_target(
        train_df, include_imputation_flags=include_imputation_flags
    )
    X_val_raw, y_val = extract_features_target(
        val_df, include_imputation_flags=include_imputation_flags
    )
    X_test_raw, y_test = extract_features_target(
        test_df, include_imputation_flags=include_imputation_flags
    )

    # 2. Fit preprocessor strictly on X_train
    preprocessor = WaterQualityPreprocessor(
        include_metadata=include_metadata,
        include_imputation_flags=include_imputation_flags,
        scale_numeric=scale_numeric,
    )
    preprocessor.fit(X_train_raw)

    X_train = preprocessor.transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)
    X_test = preprocessor.transform(X_test_raw)

    # 3. Partial measurement masking if requested
    if partial_scenario is not None:
        X_train = mask_features(X_train, available_measurements=partial_scenario)
        X_val = mask_features(X_val, available_measurements=partial_scenario)
        X_test = mask_features(X_test, available_measurements=partial_scenario)

    # 4. Final leakage validation on transformed frames
    validate_no_leakage(X_train, y_train, allow_imputation_flags=include_imputation_flags)
    validate_no_leakage(X_val, y_val, allow_imputation_flags=include_imputation_flags)
    validate_no_leakage(X_test, y_test, allow_imputation_flags=include_imputation_flags)

    return X_train, X_val, X_test, y_train, y_val, y_test
