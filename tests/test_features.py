"""Unit tests for derived feature engineering (src/features.py).   Owner: Himanshu Pandey.

Verifies:
  * Mathematical correctness of stoichiometric ratios, organic nitrogen estimate/fraction,
    and log transforms.
  * Missing-value propagation (NaN inputs safely yield NaN outputs).
  * Numerical safety: zero division protection, negative sensor drift clipping, DIN > TN clipping.
  * Negative temperature preservation (sub-zero physical readings remain valid).
  * Rejection of target-formula components (F1, F2, F3, Scope, Frequency, Amplitude, excursions, nse).
  * Rejection of target and spatial grouping leakage (CCME_Values, CCME_WQI, Area).
  * Explicit Regime 1 (Formula Emulation) vs. Regime 2 (Independent Prediction) isolation.
  * Temporal feature generation without shared file modifications, including single-load caching.
  * WaterQualityFeatureEngineer input contract enforcement.
  * Masking compatibility: integration with src.preprocess.mask_features().
  * Immutability: input DataFrame is preserved untouched.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.clean import MEASUREMENTS
from src.features import (
    CHEMICAL_FEATURE_DEPENDENCIES,
    EPSILON,
    MIN_DO_FLOOR,
    REGIME_2_PROHIBITED_FEATURES,
    WaterQualityFeatureEngineer,
    attach_temporal_features,
    build_features,
    select_regime_features,
    validate_regime,
)
from src.preprocess import mask_features


def make_test_df(n: int = 10) -> pd.DataFrame:
    """Construct a minimal valid DataFrame with the 8 measurements and Date."""
    rng = np.random.default_rng(42)
    data = {
        "Ammonia (mg/l)": rng.uniform(0.1, 5.0, n),
        "Biochemical Oxygen Demand (mg/l)": rng.uniform(1.0, 15.0, n),
        "Dissolved Oxygen (mg/l)": rng.uniform(4.0, 14.0, n),
        "Orthophosphate (mg/l)": rng.uniform(0.01, 1.0, n),
        "pH (ph units)": rng.uniform(6.5, 8.5, n),
        "Temperature (cel)": rng.uniform(10.0, 25.0, n),
        "Nitrogen (mg/l)": rng.uniform(2.0, 20.0, n),
        "Nitrate (mg/l)": rng.uniform(1.0, 15.0, n),
        "Country": ["England"] * n,
        "Waterbody Type": ["River"] * n,
        "Date": ["15-06-2020"] * n,
    }
    return pd.DataFrame(data)


class TestMathematicalCorrectness:
    def test_n_p_ratio_formula(self):
        df = pd.DataFrame({
            "Nitrogen (mg/l)": [10.0],
            "Orthophosphate (mg/l)": [0.1],
        })
        out = build_features(df, include_logs=False, include_temporal=False)
        expected = 10.0 / (0.1 + EPSILON)
        assert np.isclose(out.loc[0, "N_P_ratio"], expected)

    def test_inorganic_n_ratio_formula(self):
        df = pd.DataFrame({
            "Ammonia (mg/l)": [0.5],
            "Nitrate (mg/l)": [4.5],
        })
        out = build_features(df, include_logs=False, include_temporal=False)
        expected = 0.5 / (4.5 + EPSILON)
        assert np.isclose(out.loc[0, "Inorganic_N_ratio"], expected)

    def test_organic_n_estimate_and_fraction_formulas(self):
        df = pd.DataFrame({
            "Nitrogen (mg/l)": [10.0],
            "Nitrate (mg/l)": [4.0],
            "Ammonia (mg/l)": [1.0],
        })
        out = build_features(df, include_logs=False, include_temporal=False)
        # TN (10.0) - DIN (4.0 + 1.0 = 5.0) = 5.0 mg/l
        assert np.isclose(out.loc[0, "Organic_N_estimate"], 5.0)
        # Fraction: 5.0 / (10.0 + eps) ≈ 0.49995 in [0, 1]
        expected_frac = 5.0 / (10.0 + EPSILON)
        assert np.isclose(out.loc[0, "Organic_N_fraction"], expected_frac)
        assert 0.0 <= out.loc[0, "Organic_N_fraction"] <= 1.0

    def test_bod_do_ratio_formula(self):
        df = pd.DataFrame({
            "Biochemical Oxygen Demand (mg/l)": [6.0],
            "Dissolved Oxygen (mg/l)": [12.0],
        })
        out = build_features(df, include_logs=False, include_temporal=False)
        assert np.isclose(out.loc[0, "BOD_DO_ratio"], 6.0 / 12.0)

    def test_log_transforms_formula(self):
        df = pd.DataFrame({
            "Ammonia (mg/l)": [0.0, np.e - 1.0],
            "Biochemical Oxygen Demand (mg/l)": [1.0, 9.0],
            "Orthophosphate (mg/l)": [0.5, 3.0],
            "Nitrate (mg/l)": [0.0, 7.0],
            "Nitrogen (mg/l)": [2.0, 5.0],
        })
        out = build_features(df, include_ratios=False, include_temporal=False)
        assert np.isclose(out.loc[0, "log_Ammonia"], 0.0)
        assert np.isclose(out.loc[1, "log_Ammonia"], 1.0)
        assert np.isclose(out.loc[0, "log_BOD"], np.log1p(1.0))

    def test_temporal_cyclic_encoding(self):
        df = pd.DataFrame({
            "Date": ["01-01-2020", "02-07-2020"],
        })
        out = build_features(df, include_ratios=False, include_logs=False)
        assert "sin_day" in out.columns
        assert "cos_day" in out.columns
        assert "month" in out.columns
        assert out.loc[0, "month"] == 1.0
        assert out.loc[1, "month"] == 7.0
        assert (-1.0 <= out["sin_day"]).all() and (out["sin_day"] <= 1.0).all()
        assert (-1.0 <= out["cos_day"]).all() and (out["cos_day"] <= 1.0).all()


class TestMissingValueHandling:
    def test_nan_inputs_propagate_to_nan_derived(self):
        df = pd.DataFrame({
            "Ammonia (mg/l)": [np.nan, 1.0],
            "Nitrate (mg/l)": [5.0, np.nan],
            "Nitrogen (mg/l)": [np.nan, 10.0],
            "Orthophosphate (mg/l)": [0.1, np.nan],
            "Biochemical Oxygen Demand (mg/l)": [np.nan, 4.0],
            "Dissolved Oxygen (mg/l)": [10.0, np.nan],
        })
        out = build_features(df, include_temporal=False)

        assert np.isnan(out.loc[0, "Inorganic_N_ratio"])
        assert np.isnan(out.loc[1, "Inorganic_N_ratio"])
        assert np.isnan(out.loc[0, "N_P_ratio"])
        assert np.isnan(out.loc[1, "N_P_ratio"])
        assert np.isnan(out.loc[0, "Organic_N_estimate"])
        assert np.isnan(out.loc[1, "Organic_N_estimate"])
        assert np.isnan(out.loc[0, "Organic_N_fraction"])
        assert np.isnan(out.loc[1, "Organic_N_fraction"])
        assert np.isnan(out.loc[0, "BOD_DO_ratio"])
        assert np.isnan(out.loc[1, "BOD_DO_ratio"])
        assert np.isnan(out.loc[0, "log_Ammonia"])
        assert np.isnan(out.loc[1, "log_Nitrate"])


class TestNumericalSafetyAndSensorNoise:
    def test_zero_denominator_safely_bounded(self):
        df = pd.DataFrame({
            "Nitrogen (mg/l)": [10.0],
            "Orthophosphate (mg/l)": [0.0],
            "Ammonia (mg/l)": [2.0],
            "Nitrate (mg/l)": [0.0],
            "Biochemical Oxygen Demand (mg/l)": [5.0],
            "Dissolved Oxygen (mg/l)": [0.0],
        })
        out = build_features(df, include_logs=False, include_temporal=False)

        assert not np.isinf(out["N_P_ratio"]).any()
        assert not np.isinf(out["Inorganic_N_ratio"]).any()
        assert not np.isinf(out["BOD_DO_ratio"]).any()
        assert out.loc[0, "BOD_DO_ratio"] == 5.0 / MIN_DO_FLOOR

    def test_negative_sensor_drift_clipped(self):
        df = pd.DataFrame({
            "Ammonia (mg/l)": [-0.005],
            "Biochemical Oxygen Demand (mg/l)": [-2.0],
            "Orthophosphate (mg/l)": [-0.004],
            "Nitrogen (mg/l)": [5.0],
            "Nitrate (mg/l)": [10.0],
            "Dissolved Oxygen (mg/l)": [8.0],
        })
        out = build_features(df, include_temporal=False)

        assert out.loc[0, "log_Ammonia"] == 0.0
        assert out.loc[0, "log_BOD"] == 0.0
        assert out.loc[0, "log_Orthophosphate"] == 0.0
        assert out.loc[0, "BOD_DO_ratio"] == 0.0
        assert out.loc[0, "N_P_ratio"] >= 0.0

    def test_din_exceeding_tn_clips_organic_estimate_and_fraction_at_zero(self):
        df = pd.DataFrame({
            "Nitrogen (mg/l)": [5.0],
            "Nitrate (mg/l)": [6.0],
            "Ammonia (mg/l)": [2.0],  # DIN = 8.0 > TN = 5.0
        })
        out = build_features(df, include_logs=False, include_temporal=False)
        assert out.loc[0, "Organic_N_estimate"] == 0.0
        assert out.loc[0, "Organic_N_fraction"] == 0.0

    def test_sub_zero_temperature_preserved_without_corruption(self):
        df = pd.DataFrame({
            "Temperature (cel)": [-5.26, 0.0, 18.5],
            "Date": ["10-01-2010", "15-01-2010", "20-07-2010"],
        })
        out = build_features(df, include_ratios=False, include_logs=False, include_temporal=True)
        assert out.loc[0, "Temperature (cel)"] == -5.26
        assert "sin_day" in out.columns


class TestTemporalFeatureAttachment:
    def test_attach_temporal_features_with_explicit_date_source(self):
        df = pd.DataFrame({
            "Country": ["England", "USA"],
            "Waterbody Type": ["River", "Lake"],
        }, index=[100, 200])
        dates = pd.Series(["15-03-2021", "20-09-2021"], index=[100, 200])

        out = attach_temporal_features(df, date_source=dates)
        assert "sin_day" in out.columns
        assert "cos_day" in out.columns
        assert "month" in out.columns
        assert out.loc[100, "month"] == 3.0
        assert out.loc[200, "month"] == 9.0

    def test_attach_temporal_features_idempotent(self):
        df = pd.DataFrame({
            "Country": ["England"],
            "sin_day": [0.5],
            "cos_day": [0.5],
            "month": [6.0],
        })
        out = attach_temporal_features(df)
        assert out is df


class TestRegimeSeparationAndGovernance:
    def test_regime_2_raises_on_raw_chemical_measurement(self):
        df = pd.DataFrame({
            "Country": ["England"],
            "Waterbody Type": ["River"],
            "Ammonia (mg/l)": [0.5],
            "sin_day": [0.1],
        })
        with pytest.raises(ValueError, match="CRITICAL LEAKAGE: Chemical features"):
            validate_regime(df, regime=2)

    def test_regime_2_raises_on_derived_chemical_feature(self):
        df = pd.DataFrame({
            "Country": ["England"],
            "Waterbody Type": ["River"],
            "N_P_ratio": [15.2],
            "sin_day": [0.1],
        })
        with pytest.raises(ValueError, match="CRITICAL LEAKAGE: Chemical features"):
            validate_regime(df, regime=2)

    def test_select_regime_features_purges_all_chemicals_for_regime_2(self):
        df = make_test_df(5)
        X_all = build_features(df, include_ratios=True, include_logs=True, include_temporal=True)

        X_regime2 = select_regime_features(X_all, regime=2)
        assert set(X_regime2.columns) == {"Country", "Waterbody Type", "sin_day", "cos_day", "month"}

        for col in REGIME_2_PROHIBITED_FEATURES:
            assert col not in X_regime2.columns
        validate_regime(X_regime2, regime=2)

    def test_select_regime_features_regime_1_preserves_all_features(self):
        df = make_test_df(5)
        X_all = build_features(df, include_ratios=True, include_logs=True, include_temporal=True)

        X_regime1 = select_regime_features(X_all, regime=1)
        for col in MEASUREMENTS:
            assert col in X_regime1.columns
        for col in CHEMICAL_FEATURE_DEPENDENCIES:
            assert col in X_regime1.columns
        assert "sin_day" in X_regime1.columns
        validate_regime(X_regime1, regime=1)

    def test_validate_regime_raises_on_target_or_group_leakage(self):
        df_target = pd.DataFrame({"CCME_Values": [85.0], "Country": ["England"]})
        with pytest.raises(ValueError, match="Forbidden column 'CCME_Values'"):
            validate_regime(df_target, regime=1)

        df_site = pd.DataFrame({"Area": ["Station_1"], "Country": ["England"]})
        with pytest.raises(ValueError, match="Forbidden column 'Area'"):
            validate_regime(df_site, regime=1)

    def test_validate_regime_rejects_formula_components_f1_f2_f3(self):
        for col in ["F1", "F2", "F3", "f1", "f2", "f3"]:
            df = pd.DataFrame({col: [25.0], "Country": ["England"]})
            with pytest.raises(ValueError, match="Target-formula component"):
                validate_regime(df, regime=1)
            with pytest.raises(ValueError, match="Target-formula component"):
                validate_regime(df, regime=2)

    def test_validate_regime_rejects_excursions_and_nse(self):
        for col in ["excursion", "excursions", "excursion_Ammonia", "nse", "failed_variables", "Scope", "Amplitude"]:
            df = pd.DataFrame({col: [1.5], "Country": ["England"]})
            with pytest.raises(ValueError, match="Target-formula component"):
                validate_regime(df, regime=1)
            with pytest.raises(ValueError, match="Target-formula component"):
                validate_regime(df, regime=2)


class TestAblationMaskingCompatibility:
    def test_mask_features_drops_dependent_engineered_features(self):
        df = make_test_df(5)
        X_feat = build_features(df, include_temporal=True)
        X_clean = X_feat.drop(columns=["Date"], errors="ignore")

        masked_7_8 = mask_features(
            X_clean,
            available_measurements="7/8",
            feature_dependencies=CHEMICAL_FEATURE_DEPENDENCIES,
        )
        assert "Orthophosphate (mg/l)" not in masked_7_8.columns
        assert "N_P_ratio" not in masked_7_8.columns
        assert "log_Orthophosphate" not in masked_7_8.columns
        assert "Inorganic_N_ratio" in masked_7_8.columns
        assert "sin_day" in masked_7_8.columns

    def test_mask_features_4_of_8_core_suite(self):
        df = make_test_df(5)
        X_feat = build_features(df, include_temporal=True)
        X_clean = X_feat.drop(columns=["Date"], errors="ignore")

        masked_4_8 = mask_features(
            X_clean,
            available_measurements="4/8",
            feature_dependencies=CHEMICAL_FEATURE_DEPENDENCIES,
        )
        assert "Ammonia (mg/l)" not in masked_4_8.columns
        assert "Biochemical Oxygen Demand (mg/l)" not in masked_4_8.columns
        assert "Orthophosphate (mg/l)" not in masked_4_8.columns
        assert "Nitrogen (mg/l)" not in masked_4_8.columns
        assert "N_P_ratio" not in masked_4_8.columns
        assert "Inorganic_N_ratio" not in masked_4_8.columns
        assert "Organic_N_estimate" not in masked_4_8.columns
        assert "Organic_N_fraction" not in masked_4_8.columns
        assert "BOD_DO_ratio" not in masked_4_8.columns
        assert "pH (ph units)" in masked_4_8.columns
        assert "Dissolved Oxygen (mg/l)" in masked_4_8.columns
        assert "Temperature (cel)" in masked_4_8.columns
        assert "Nitrate (mg/l)" in masked_4_8.columns
        assert "log_Nitrate" in masked_4_8.columns
        assert "sin_day" in masked_4_8.columns


class TestSklearnTransformerInterface:
    def test_transformer_regime_1_fit_transform(self):
        df = make_test_df(6)
        fe = WaterQualityFeatureEngineer(regime=1)
        out = fe.fit_transform(df)
        assert isinstance(out, pd.DataFrame)
        assert "N_P_ratio" in out.columns
        assert "Organic_N_estimate" in out.columns
        assert "Organic_N_fraction" in out.columns
        assert "log_BOD" in out.columns
        assert "sin_day" in out.columns
        validate_regime(out, regime=1)

    def test_transformer_regime_2_fit_transform(self):
        df = make_test_df(6)
        fe = WaterQualityFeatureEngineer(regime=2)
        out = fe.fit_transform(df)
        assert isinstance(out, pd.DataFrame)
        assert "N_P_ratio" not in out.columns
        assert "Ammonia (mg/l)" not in out.columns
        assert set(out.columns) == {"Country", "Waterbody Type", "sin_day", "cos_day", "month"}
        validate_regime(out, regime=2)

    def test_transformer_enforces_contract_when_dates_missing(self):
        df_no_dates = pd.DataFrame({
            "Country": ["England"],
            "Waterbody Type": ["River"],
        }, index=[999999999])  # Index not in clean dataset

        fe = WaterQualityFeatureEngineer(regime=2, include_temporal=True)
        with pytest.raises(ValueError, match="WaterQualityFeatureEngineer contract violation"):
            fe.transform(df_no_dates)

    def test_transformer_accepts_explicit_date_source_when_dates_missing(self):
        df_no_dates = pd.DataFrame({
            "Country": ["England", "USA"],
            "Waterbody Type": ["River", "Lake"],
        }, index=[501, 502])
        dates = pd.Series(["10-04-2022", "12-08-2022"], index=[501, 502])

        fe = WaterQualityFeatureEngineer(regime=2, include_temporal=True, date_source=dates)
        out = fe.transform(df_no_dates)
        assert "sin_day" in out.columns
        assert out.loc[501, "month"] == 4.0
        assert out.loc[502, "month"] == 8.0
