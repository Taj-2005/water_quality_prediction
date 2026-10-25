"""Derived feature engineering for water quality prediction.   Owner: Himanshu Pandey.

Generates:
  * Stoichiometric and hydrochemical ratios:
      - N_P_ratio: Nitrogen / (max(0, Orthophosphate) + eps)
      - Inorganic_N_ratio: Ammonia / (max(0, Nitrate) + eps)
      - Organic_N_estimate: max(0, max(0, Nitrogen) - (max(0, Nitrate) + max(0, Ammonia))) [concentration difference in mg/l]
      - Organic_N_fraction: Organic_N_estimate / (max(0, Nitrogen) + eps) [dimensionless fraction in [0, 1]]
      - BOD_DO_ratio: max(0, BOD) / max(0.01, DO)
  * Monotonic log transforms for positively skewed heavy-tailed measurements:
      - log_Ammonia, log_BOD, log_Orthophosphate, log_Nitrate, log_Nitrogen
  * Cyclic temporal covariates (annual seasonality):
      - sin_day, cos_day, month

Prediction Regimes & Leakage Governance:
  * Regime 1 (Formula Emulation / Benchmark Surrogate):
      CRITICAL CAVEAT: Models trained in Regime 1 use the 8 chemical inputs that mathematically
      define the CCME index equation. Therefore, Regime 1 strictly benchmarks how accurately
      tree ensembles can approximate/emulate the deterministic CCME formula. It DOES NOT represent
      independent environmental forecasting or leakage-free predictive performance.
  * Regime 2 (Independent Environmental Prediction):
      Predicts water quality using strictly non-contributing metadata and spatiotemporal
      covariates (Country, Waterbody Type, sin_day, cos_day, month).
      STRICTLY PROHIBITED for Regime 2: Any raw chemical measurement or derived chemical feature.
  * Hard Negative Constraints (UNSAFE for ALL regimes):
      - Target formula components (F1, F2, F3, Scope, Frequency, Amplitude, excursions, nse):
        strictly forbidden (trivial equation reconstruction).
      - CCME_Values: forbidden (the regression target).
      - CCME_WQI: forbidden (direct post-hoc label leakage).
      - Area: forbidden (spatial grouping key; memorizes sites rather than general dynamics).
      - BOD:COD: excluded (COD absent from dataset schema).

Integration with Preprocessing:
  Feature column names match `DEFAULT_FEATURE_DEPENDENCIES` in `src.preprocess`, ensuring
  `src.preprocess.mask_features()` automatically drops derived features whenever their parent
  chemical measurements are masked under ablation scenarios (8/8, 7/8, 6/8, 4/8).
"""

from __future__ import annotations

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

EPSILON: float = 0.001
MIN_DO_FLOOR: float = 0.01

# Mapping of derived chemical features to their required raw measurement columns.
# Matches DEFAULT_FEATURE_DEPENDENCIES in src.preprocess for ablation compatibility.
CHEMICAL_FEATURE_DEPENDENCIES: dict[str, list[str]] = {
    "N_P_ratio": ["Nitrogen (mg/l)", "Orthophosphate (mg/l)"],
    "Inorganic_N_ratio": ["Ammonia (mg/l)", "Nitrate (mg/l)"],
    "Organic_N_estimate": ["Nitrogen (mg/l)", "Nitrate (mg/l)", "Ammonia (mg/l)"],
    "Organic_N_fraction": ["Nitrogen (mg/l)", "Nitrate (mg/l)", "Ammonia (mg/l)"],
    "BOD_DO_ratio": ["Biochemical Oxygen Demand (mg/l)", "Dissolved Oxygen (mg/l)"],
    "log_Ammonia": ["Ammonia (mg/l)"],
    "log_BOD": ["Biochemical Oxygen Demand (mg/l)"],
    "log_Orthophosphate": ["Orthophosphate (mg/l)"],
    "log_Nitrate": ["Nitrate (mg/l)"],
    "log_Nitrogen": ["Nitrogen (mg/l)"],
}

# Features classified by prediction regime
REGIME_1_CHEMICAL_INPUTS: list[str] = list(MEASUREMENTS)
REGIME_1_CHEMICAL_DERIVED: list[str] = list(CHEMICAL_FEATURE_DEPENDENCIES.keys())
REGIME_1_ALL_CHEMICALS: list[str] = REGIME_1_CHEMICAL_INPUTS + REGIME_1_CHEMICAL_DERIVED

REGIME_2_METADATA_FEATURES: list[str] = ["Country", "Waterbody Type"]
REGIME_2_TEMPORAL_FEATURES: list[str] = ["sin_day", "cos_day", "month"]
REGIME_2_ALLOWED_FEATURES: list[str] = REGIME_2_METADATA_FEATURES + REGIME_2_TEMPORAL_FEATURES

# Strict isolation: ALL 8 raw measurements and 10 derived chemical features are forbidden in Regime 2
REGIME_2_PROHIBITED_FEATURES: list[str] = list(REGIME_1_ALL_CHEMICALS)

REGIME_1_ALLOWED_FEATURES: list[str] = REGIME_1_ALL_CHEMICALS + REGIME_2_ALLOWED_FEATURES

# Forbidden across ALL regimes
FORBIDDEN_LEAKAGE_COLUMNS: list[str] = [
    TARGET_COLUMN,  # "CCME_Values"
    "CCME_WQI",     # Direct label leakage
    "Area",         # Spatial grouping key
]

# CCME WQI Target-Formula Components (strictly forbidden across ALL regimes)
FORMULA_COMPONENT_LEAKAGE_COLUMNS: list[str] = [
    "F1", "F2", "F3",
    "f1", "f2", "f3",
    "Scope", "Frequency", "Amplitude",
    "scope", "frequency", "amplitude",
    "excursion", "excursions", "nse",
    "failed_variables", "failed_tests",
]


def build_features(
    df: pd.DataFrame,
    include_ratios: bool = True,
    include_logs: bool = True,
    include_temporal: bool = True,
    date_col: str | None = None,
    inplace: bool = False,
) -> pd.DataFrame:
    """Generate derived features from hydrochemical measurements and date metadata.

    Mathematical and Data Handling Rules:
      - Missing values: Propagates NaN if any required input measurement is NaN.
        Does not fabricate or impute data; imputation belongs in src.preprocess.
      - Negative sensor drift: Non-temperature concentrations (< 0) are clipped at 0
        before computing ratios and logs, preventing complex logs or inverted signs.
      - Zero division: Denominators are protected by EPSILON (0.001) or MIN_DO_FLOOR (0.01).
      - Assay noise: Organic N estimate and fraction clip at 0 if DIN slightly exceeds TN.
      - Sub-zero temperature: Preserved as valid physical readings.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing raw measurements and/or date metadata.
    include_ratios : bool, default True
        Whether to generate stoichiometric and redox ratios (N_P_ratio, Inorganic_N_ratio,
        Organic_N_estimate, Organic_N_fraction, BOD_DO_ratio).
    include_logs : bool, default True
        Whether to generate log1p transforms for positively skewed columns (Ammonia, BOD,
        Orthophosphate, Nitrate, Nitrogen).
    include_temporal : bool, default True
        Whether to generate cyclic day-of-year and month features if date information is present.
    date_col : str | None, optional
        Explicit name of date column. Defaults to "Date_parsed" if present, else "Date".
    inplace : bool, default False
        Whether to add columns directly to df (False returns a shallow-copied new DataFrame).

    Returns
    -------
    pd.DataFrame
        DataFrame with derived features added. Existing columns are preserved untouched.
    """
    out = df if inplace else df.copy()

    # 1. Stoichiometric and Hydrochemical Ratios (Regime 1 / Chemical Suite)
    if include_ratios:
        # N:P ratio: Nitrogen / (max(0, Orthophosphate) + eps)
        if "Nitrogen (mg/l)" in out.columns and "Orthophosphate (mg/l)" in out.columns:
            n = out["Nitrogen (mg/l)"]
            p = out["Orthophosphate (mg/l)"]
            denom = np.maximum(0.0, p.fillna(0.0)) + EPSILON
            num = np.maximum(0.0, n.fillna(0.0))
            out["N_P_ratio"] = np.where(n.isna() | p.isna(), np.nan, num / denom)

        # Inorganic Nitrogen Speciation: Ammonia / (max(0, Nitrate) + eps)
        if "Ammonia (mg/l)" in out.columns and "Nitrate (mg/l)" in out.columns:
            nh4 = out["Ammonia (mg/l)"]
            no3 = out["Nitrate (mg/l)"]
            denom = np.maximum(0.0, no3.fillna(0.0)) + EPSILON
            num = np.maximum(0.0, nh4.fillna(0.0))
            out["Inorganic_N_ratio"] = np.where(nh4.isna() | no3.isna(), np.nan, num / denom)

        # Organic Nitrogen:
        #   - Organic_N_estimate: max(0, TN - (NO3 + NH4)) [concentration difference, mg/l]
        #   - Organic_N_fraction: Organic_N_estimate / (max(0, TN) + eps) [dimensionless fraction in [0, 1]]
        if (
            "Nitrogen (mg/l)" in out.columns
            and "Nitrate (mg/l)" in out.columns
            and "Ammonia (mg/l)" in out.columns
        ):
            n = out["Nitrogen (mg/l)"]
            no3 = out["Nitrate (mg/l)"]
            nh4 = out["Ammonia (mg/l)"]
            din = np.maximum(0.0, no3.fillna(0.0)) + np.maximum(0.0, nh4.fillna(0.0))
            tn = np.maximum(0.0, n.fillna(0.0))
            org_est = np.maximum(0.0, tn - din)
            org_frac = org_est / (tn + EPSILON)

            any_na = n.isna() | no3.isna() | nh4.isna()
            out["Organic_N_estimate"] = np.where(any_na, np.nan, org_est)
            out["Organic_N_fraction"] = np.where(any_na, np.nan, np.clip(org_frac, 0.0, 1.0))

        # Oxygen Stress / Organic Load Index: max(0, BOD) / max(0.01, DO)
        if "Biochemical Oxygen Demand (mg/l)" in out.columns and "Dissolved Oxygen (mg/l)" in out.columns:
            bod = out["Biochemical Oxygen Demand (mg/l)"]
            do = out["Dissolved Oxygen (mg/l)"]
            denom = np.maximum(MIN_DO_FLOOR, do.fillna(0.0))
            num = np.maximum(0.0, bod.fillna(0.0))
            out["BOD_DO_ratio"] = np.where(bod.isna() | do.isna(), np.nan, num / denom)

    # 2. Monotonic Log Transforms for Skewed Heavy-Tailed Columns
    if include_logs:
        log_candidates = [
            ("log_Ammonia", "Ammonia (mg/l)"),
            ("log_BOD", "Biochemical Oxygen Demand (mg/l)"),
            ("log_Orthophosphate", "Orthophosphate (mg/l)"),
            ("log_Nitrate", "Nitrate (mg/l)"),
            ("log_Nitrogen", "Nitrogen (mg/l)"),
        ]
        for derived_name, raw_col in log_candidates:
            if raw_col in out.columns:
                val = out[raw_col]
                clean_val = np.maximum(0.0, val.fillna(0.0))
                out[derived_name] = np.where(val.isna(), np.nan, np.log1p(clean_val))

    # 3. Cyclic Temporal Covariates (Independent / Regime 2 Context)
    if include_temporal:
        dt_series = None
        if date_col is not None and date_col in out.columns:
            target_series = out[date_col]
            if pd.api.types.is_datetime64_any_dtype(target_series):
                dt_series = target_series.dt
            else:
                dt_series = pd.to_datetime(target_series, format="%d-%m-%Y", errors="coerce").dt
        elif "Date_parsed" in out.columns:
            target_series = out["Date_parsed"]
            if pd.api.types.is_datetime64_any_dtype(target_series):
                dt_series = target_series.dt
            else:
                dt_series = pd.to_datetime(target_series, format="%d-%m-%Y", errors="coerce").dt
        elif "Date" in out.columns:
            dt_series = pd.to_datetime(out["Date"], format="%d-%m-%Y", errors="coerce").dt

        if dt_series is not None:
            doy = dt_series.dayofyear
            out["sin_day"] = np.sin(2.0 * np.pi * doy / 365.25)
            out["cos_day"] = np.cos(2.0 * np.pi * doy / 365.25)
            out["month"] = dt_series.month.astype("float32")

    return out


def attach_temporal_features(
    df: pd.DataFrame,
    date_source: pd.Series | pd.DataFrame | None = None,
    inplace: bool = False,
) -> pd.DataFrame:
    """Safely attach cyclic temporal features (sin_day, cos_day, month) to a feature matrix.

    This resolves the situation where get_splits() excludes 'Date' and 'Date_parsed'
    from X to prevent raw date leakage, allowing models to use seasonal features
    without requiring any modifications to shared upstream pipeline files.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (e.g., X_train, X_val, X_test from get_splits()).
    date_source : pd.Series or pd.DataFrame, optional
        Explicit source of dates aligned with df's index. If None and 'Date' or
        'Date_parsed' is not in df, dates are loaded from clean.parquet using df.index.
        To avoid reloading clean.parquet multiple times, callers should pass an already
        loaded date_source across multiple partitions.
    inplace : bool, default False
        Whether to modify df in place.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'sin_day', 'cos_day', and 'month' attached.
    """
    if {"sin_day", "cos_day", "month"}.issubset(df.columns):
        return df

    out = df if inplace else df.copy()

    # Case 1: Date column already present in df
    if "Date_parsed" in out.columns or "Date" in out.columns:
        return build_features(out, include_ratios=False, include_logs=False, include_temporal=True, inplace=True)

    # Case 2: Explicit date_source passed
    if date_source is not None:
        if isinstance(date_source, pd.DataFrame):
            col = "Date_parsed" if "Date_parsed" in date_source.columns else ("Date" if "Date" in date_source.columns else date_source.columns[0])
            dates = date_source.loc[out.index, col]
        else:
            dates = date_source.loc[out.index]
    else:
        # Case 3: Load clean dataset to look up dates by index
        from src.evaluate_api import load_data

        clean_dates = load_data()[["Date_parsed"]]
        dates = clean_dates.loc[out.index, "Date_parsed"]

    if pd.api.types.is_datetime64_any_dtype(dates):
        dt_series = dates.dt
    else:
        dt_series = pd.to_datetime(dates, format="%d-%m-%Y", errors="coerce").dt

    doy = dt_series.dayofyear
    out["sin_day"] = np.sin(2.0 * np.pi * doy / 365.25)
    out["cos_day"] = np.cos(2.0 * np.pi * doy / 365.25)
    out["month"] = dt_series.month.astype("float32")
    return out


def validate_regime(X: pd.DataFrame, regime: int = 1) -> None:
    """Validate that feature matrix X strictly adheres to the specified prediction regime.

    Parameters
    ----------
    X : pd.DataFrame
        Input feature matrix.
    regime : int in {1, 2}
        Prediction regime:
          - 1: Formula Emulation (Chemical inputs permitted; target, labels, site IDs,
               and intermediate formula components strictly forbidden).
          - 2: Independent Prediction (ALL chemical inputs strictly forbidden).

    Raises
    ------
    ValueError
        If regime is invalid, if target/label/site leakage is detected, if formula
        components (F1, F2, F3, excursions, nse) are present, or if chemical
        features are present in Regime 2.
    """
    if regime not in (1, 2):
        raise ValueError(
            f"Invalid regime {regime}. Must be 1 (Formula Emulation) or 2 (Independent Prediction)."
        )

    cols = set(X.columns)

    # 1. Hard negative constraints (forbidden across ALL regimes)
    for forbidden in FORBIDDEN_LEAKAGE_COLUMNS:
        if forbidden in cols:
            raise ValueError(
                f"CRITICAL LEAKAGE: Forbidden column '{forbidden}' found in feature matrix X! "
                "Target values, post-hoc classifications, and spatial grouping keys must never be features."
            )

    # 2. Formula component leakage (F1, F2, F3, Scope, Frequency, Amplitude, excursions, nse)
    for col in cols:
        col_lower = str(col).lower().strip()
        if (
            col in FORMULA_COMPONENT_LEAKAGE_COLUMNS
            or col_lower in {c.lower() for c in FORMULA_COMPONENT_LEAKAGE_COLUMNS}
            or "excursion" in col_lower
            or col_lower in ("f1", "f2", "f3", "nse", "scope", "amplitude")
        ):
            raise ValueError(
                f"CRITICAL LEAKAGE: Target-formula component '{col}' detected in feature matrix X! "
                "Intermediate equation variables (F1/Scope, F2/Frequency, F3/Amplitude, excursions, nse) "
                "constitute trivial mathematical reconstruction of CCME_Values and are strictly forbidden."
            )

    # 3. Strict Regime 2 isolation
    if regime == 2:
        chem_leaks = [c for c in cols if c in REGIME_2_PROHIBITED_FEATURES]
        if chem_leaks:
            raise ValueError(
                f"CRITICAL LEAKAGE: Chemical features {chem_leaks} detected in Regime 2 feature matrix! "
                "Regime 2 (Independent Environmental Prediction) strictly prohibits the 8 chemical inputs "
                "and any derived chemical features because CCME_Values is computed directly from them."
            )


def select_regime_features(
    X: pd.DataFrame,
    regime: int = 1,
    include_metadata: bool = True,
    include_temporal: bool = True,
) -> pd.DataFrame:
    """Explicitly filter and return feature matrix X for the specified prediction regime.

    Guarantees that chemical features cannot accidentally enter Regime 2 models.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (can contain raw measurements, derived features, metadata, temporal).
    regime : int in {1, 2}
        1 for Formula Emulation, 2 for Independent Prediction.
    include_metadata : bool, default True
        Whether to retain Country and Waterbody Type.
    include_temporal : bool, default True
        Whether to retain sin_day, cos_day, and month.

    Returns
    -------
    pd.DataFrame
        Filtered feature matrix compliant with the selected regime.
    """
    if regime not in (1, 2):
        raise ValueError(f"Invalid regime {regime}. Must be 1 or 2.")

    cols_to_keep: list[str] = []

    if regime == 1:
        # Include chemicals and derived chemicals (Formula Emulation)
        cols_to_keep.extend([c for c in REGIME_1_ALL_CHEMICALS if c in X.columns])
        if include_metadata:
            cols_to_keep.extend([c for c in REGIME_2_METADATA_FEATURES if c in X.columns])
        if include_temporal:
            cols_to_keep.extend([c for c in REGIME_2_TEMPORAL_FEATURES if c in X.columns])
    elif regime == 2:
        # Strictly non-chemical features (Independent Prediction)
        if include_metadata:
            cols_to_keep.extend([c for c in REGIME_2_METADATA_FEATURES if c in X.columns])
        if include_temporal:
            cols_to_keep.extend([c for c in REGIME_2_TEMPORAL_FEATURES if c in X.columns])

    out = X[cols_to_keep].copy()
    validate_regime(out, regime=regime)
    return out


def get_regime_splits(
    regime: int = 1,
    splits_dir: Path | str | None = None,
    partial_scenario: str | None = "8/8",
    include_ratios: bool = True,
    include_logs: bool = True,
    include_temporal: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """End-to-end split loader with derived feature engineering and regime isolation.

    Prediction Regimes:
      - Regime 1 (Formula Emulation / Benchmark Surrogate):
          Tree models trained on chemical measurements benchmark equation reconstruction
          capacity; results must not be characterized as un-leaked environmental prediction.
      - Regime 2 (Independent Environmental Prediction):
          Uses strictly non-chemical covariates (Country, Waterbody Type, sin_day, cos_day, month).

    Optimization:
      Loads the date series exactly once to attach temporal features across train,
      validation, and test partitions without redundant dataset disk I/O.
    """
    from src.evaluate_api import get_splits, load_data
    from src.preprocess import mask_features

    X_train, X_val, X_test, y_train, y_val, y_test = get_splits(
        splits_dir=splits_dir,
        include_metadata=True,
        include_imputation_flags=False,
    )

    # Load date series ONCE and reuse across all three partitions
    date_series = None
    if include_temporal:
        date_series = load_data()["Date_parsed"]

    if regime == 1:
        # Build derived hydrochemical features
        X_train = build_features(X_train, include_ratios=include_ratios, include_logs=include_logs, include_temporal=False)
        X_val = build_features(X_val, include_ratios=include_ratios, include_logs=include_logs, include_temporal=False)
        X_test = build_features(X_test, include_ratios=include_ratios, include_logs=include_logs, include_temporal=False)

        # Attach temporal features using the cached date series
        if include_temporal:
            X_train = attach_temporal_features(X_train, date_source=date_series)
            X_val = attach_temporal_features(X_val, date_source=date_series)
            X_test = attach_temporal_features(X_test, date_source=date_series)

        # Apply ablation masking if requested
        if partial_scenario is not None and partial_scenario != "8/8":
            deps = dict(CHEMICAL_FEATURE_DEPENDENCIES)
            X_train = mask_features(X_train, available_measurements=partial_scenario, feature_dependencies=deps)
            X_val = mask_features(X_val, available_measurements=partial_scenario, feature_dependencies=deps)
            X_test = mask_features(X_test, available_measurements=partial_scenario, feature_dependencies=deps)

        X_train = select_regime_features(X_train, regime=1, include_temporal=include_temporal)
        X_val = select_regime_features(X_val, regime=1, include_temporal=include_temporal)
        X_test = select_regime_features(X_test, regime=1, include_temporal=include_temporal)

    elif regime == 2:
        # Attach temporal features using the cached date series
        if include_temporal:
            X_train = attach_temporal_features(X_train, date_source=date_series)
            X_val = attach_temporal_features(X_val, date_source=date_series)
            X_test = attach_temporal_features(X_test, date_source=date_series)

        # Strictly filter out all chemical features
        X_train = select_regime_features(X_train, regime=2, include_temporal=include_temporal)
        X_val = select_regime_features(X_val, regime=2, include_temporal=include_temporal)
        X_test = select_regime_features(X_test, regime=2, include_temporal=include_temporal)

    validate_regime(X_train, regime=regime)
    validate_regime(X_val, regime=regime)
    validate_regime(X_test, regime=regime)

    return X_train, X_val, X_test, y_train, y_val, y_test


class WaterQualityFeatureEngineer(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible transformer for derived feature engineering.

    Stateless and train-safe: learns no statistics from X_train, operating on
    vectorized row-wise hydrochemical and temporal calculations.

    Input Contract & Temporal Handling:
      1. For hydrochemical features (Regime 1), input DataFrame must contain the 8
         chemical measurement columns.
      2. For temporal features (sin_day, cos_day, month):
         - If 'Date' or 'Date_parsed' is present in X, temporal features are derived directly.
         - If dates are already removed from X (e.g., following Parth's preprocessor),
           an explicit date_source can be provided at initialization via date_source=...
         - If date_source is None and dates are missing from X:
           * If X has an index matching clean.parquet, dates are looked up automatically.
           * If X has an unaligned or non-matching index, a clear ValueError is raised
             enforcing the input contract.
    """

    def __init__(
        self,
        regime: int = 1,
        include_ratios: bool = True,
        include_logs: bool = True,
        include_temporal: bool = True,
        date_col: str | None = None,
        date_source: pd.Series | pd.DataFrame | None = None,
    ) -> None:
        self.regime = regime
        self.include_ratios = include_ratios
        self.include_logs = include_logs
        self.include_temporal = include_temporal
        self.date_col = date_col
        self.date_source = date_source

    def fit(self, X: pd.DataFrame, y: Any = None) -> WaterQualityFeatureEngineer:
        """Stateless fit returning self."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform input DataFrame by computing derived features and filtering by regime."""
        out = X.copy()

        # Handle temporal features if requested but dates are not in X
        if self.include_temporal and not {"sin_day", "cos_day", "month"}.issubset(out.columns):
            has_date_col = (
                "Date_parsed" in out.columns
                or "Date" in out.columns
                or (self.date_col is not None and self.date_col in out.columns)
            )
            if not has_date_col:
                try:
                    out = attach_temporal_features(out, date_source=self.date_source, inplace=True)
                except Exception as exc:
                    raise ValueError(
                        "WaterQualityFeatureEngineer contract violation: 'include_temporal=True' was requested, "
                        "but input DataFrame lacks 'Date'/'Date_parsed' columns and date lookup failed. "
                        "Please pass a DataFrame containing date columns, supply 'date_source=...' at initialization, "
                        "or call attach_temporal_features() beforehand."
                    ) from exc

        if self.regime == 1:
            out = build_features(
                out,
                include_ratios=self.include_ratios,
                include_logs=self.include_logs,
                include_temporal=self.include_temporal,
                date_col=self.date_col,
                inplace=True,
            )
            return select_regime_features(
                out,
                regime=1,
                include_metadata=True,
                include_temporal=self.include_temporal,
            )
        elif self.regime == 2:
            out = build_features(
                out,
                include_ratios=False,
                include_logs=False,
                include_temporal=self.include_temporal,
                date_col=self.date_col,
                inplace=True,
            )
            return select_regime_features(
                out,
                regime=2,
                include_metadata=True,
                include_temporal=self.include_temporal,
            )
        else:
            raise ValueError(f"Invalid regime {self.regime}. Must be 1 or 2.")
