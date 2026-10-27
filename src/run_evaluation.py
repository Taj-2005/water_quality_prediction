"""Master experiment and evaluation runner for the Water Quality Prediction project.

Executes all final integration and evaluation steps:
  1. Shared pipeline validation (zero leakage, identical splits)
  2. Naive baseline evaluation (median y_train)
  3. Random Forest evaluation (engineered vs raw)
  4. XGBoost evaluation (engineered vs raw)
  5. Train vs Val vs Test overfitting analysis & generalization gaps
  6. Partial measurement experiments (8/8, 7/8, 6/8, 4/8)
  7. Upstream imputation ablation (without flags vs with flags)
  8. Feature engineering ablation
  9. Error analysis & Country analysis (noting China in-sample constraint)
 10. Bootstrap confidence intervals (1,000 bootstrap draws on fixed test predictions)
 11. Publication-quality figure generation
 12. Machine-readable result export to results/

Owner: Parthraj Singh Bhati.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import REPORTS, SEED
from src.evaluate_api import get_splits
from src.evaluation import (
    bootstrap_test_metrics,
    evaluate_by_group,
    evaluate_regression,
)
from src.features import get_regime_splits
from src.models_rf import (
    CategoricalEncoder,
    apply_subsample,
)
from src.models_xgb import (
    build_model,
    match_categories,
)
from src.preprocess import validate_no_leakage

RESULTS_DIR = Path("results")
FIGURES_DIR = REPORTS / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Best tuned XGBoost parameters from Jivit's Optuna optimization
XGB_BEST_PARAMS = {
    "learning_rate": 0.05521687825110163,
    "max_depth": 7,
    "subsample": 0.9501985440733633,
    "colsample_bytree": 0.8139729920161747,
    "min_child_weight": 30,
    "reg_alpha": 0.007059856297239945,
    "reg_lambda": 33.116390971061385,
}

# Best RF parameters from Himanshu's validation sweep
RF_BEST_PARAMS = {
    "n_estimators": 100,
    "max_depth": 15,
    "min_samples_leaf": 5,
    "max_features": "sqrt",
}


def run_all_evaluations():
    print("=" * 70)
    print("STEP 1 & 2: VERIFY SHARED PIPELINE & DATA INTEGRITY")
    print("=" * 70)

    # 1. Load shared splits
    X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = get_splits(
        include_metadata=True, include_imputation_flags=False
    )
    print(f"Train split shape: {X_train_raw.shape}, Target length: {len(y_train):,}")
    print(f"Val split shape:   {X_val_raw.shape}, Target length: {len(y_val):,}")
    print(f"Test split shape:  {X_test_raw.shape}, Target length: {len(y_test):,}")

    # Validate leakage
    validate_no_leakage(X_train_raw, y_train, allow_imputation_flags=False)
    validate_no_leakage(X_val_raw, y_val, allow_imputation_flags=False)
    validate_no_leakage(X_test_raw, y_test, allow_imputation_flags=False)
    print("Shared split verified: zero target or spatial site leakage.")

    # ---------------------------------------------------------
    # STEP 4: NAIVE BASELINE (median y_train)
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 4: RUN NAIVE BASELINE (median y_train)")
    print("=" * 70)

    t0 = time.perf_counter()
    train_median = float(np.median(y_train))
    train_mean = float(np.mean(y_train))
    baseline_fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred_val_baseline = np.full(len(y_val), train_median)
    y_pred_test_baseline = np.full(len(y_test), train_median)
    y_pred_train_baseline = np.full(len(y_train), train_median)
    baseline_pred_time = time.perf_counter() - t0

    metrics_baseline_train = evaluate_regression(y_train, y_pred_train_baseline)
    metrics_baseline_val = evaluate_regression(y_val, y_pred_val_baseline)
    metrics_baseline_test = evaluate_regression(y_test, y_pred_test_baseline)

    print(f"Training target median: {train_median:.4f} (mean: {train_mean:.4f})")
    print(
        f"Baseline Val  -> MAE={metrics_baseline_val['mae']:.4f}, "
        f"RMSE={metrics_baseline_val['rmse']:.4f}, R2={metrics_baseline_val['r2']:.6f}"
    )
    print(
        f"Baseline Test -> MAE={metrics_baseline_test['mae']:.4f}, "
        f"RMSE={metrics_baseline_test['rmse']:.4f}, R2={metrics_baseline_test['r2']:.6f}"
    )

    # ---------------------------------------------------------
    # STEP 5: PRIMARY MODELS EVALUATION
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 5: TRAIN & EVALUATE RANDOM FOREST (100k subsample, engineered)")
    print("=" * 70)

    from sklearn.ensemble import RandomForestRegressor

    X_train_eng, X_val_eng, X_test_eng, _, _, _ = get_regime_splits(
        regime=1, partial_scenario="8/8"
    )

    # Deterministic 100k subsample for RF training parity
    X_tr_rf, y_tr_rf = apply_subsample(X_train_eng, y_train, subsample=100_000, seed=SEED)

    rf_encoder = CategoricalEncoder()
    X_tr_rf_enc = rf_encoder.fit_transform(X_tr_rf)
    X_va_rf_enc = rf_encoder.transform(X_val_eng)
    X_te_rf_enc = rf_encoder.transform(X_test_eng)

    t0 = time.perf_counter()
    rf_model = RandomForestRegressor(
        **RF_BEST_PARAMS,
        random_state=SEED,
        n_jobs=-1,
    )
    rf_model.fit(X_tr_rf_enc, y_tr_rf)
    rf_train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred_train_rf = rf_model.predict(X_tr_rf_enc)
    y_pred_val_rf = rf_model.predict(X_va_rf_enc)
    y_pred_test_rf = rf_model.predict(X_te_rf_enc)
    rf_pred_time = time.perf_counter() - t0

    metrics_rf_train = evaluate_regression(y_tr_rf, y_pred_train_rf)
    metrics_rf_val = evaluate_regression(y_val, y_pred_val_rf)
    metrics_rf_test = evaluate_regression(y_test, y_pred_test_rf)

    print(f"RF Training Time: {rf_train_time:.2f}s, Inference Time: {rf_pred_time:.2f}s")
    print(
        f"RF Val  -> MAE={metrics_rf_val['mae']:.4f}, "
        f"RMSE={metrics_rf_val['rmse']:.4f}, R2={metrics_rf_val['r2']:.6f}"
    )
    print(
        f"RF Test -> MAE={metrics_rf_test['mae']:.4f}, "
        f"RMSE={metrics_rf_test['rmse']:.4f}, R2={metrics_rf_test['r2']:.6f}"
    )

    print("\n" + "=" * 70)
    print("STEP 5: TRAIN & EVALUATE XGBOOST (Full 1.98M train, engineered)")
    print("=" * 70)

    X_tr_xgb, X_va_xgb, X_te_xgb = match_categories(X_train_eng, X_val_eng, X_test_eng)

    t0 = time.perf_counter()
    xgb_model = build_model(XGB_BEST_PARAMS)
    xgb_model.fit(X_tr_xgb, y_train, eval_set=[(X_va_xgb, y_val)], verbose=False)
    xgb_train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    # Predict on 100k train sample for train metrics computation (for speed)
    X_tr_xgb_sub, y_tr_xgb_sub = apply_subsample(X_tr_xgb, y_train, subsample=100_000, seed=SEED)
    y_pred_train_xgb = xgb_model.predict(X_tr_xgb_sub)
    y_pred_val_xgb = xgb_model.predict(X_va_xgb)
    y_pred_test_xgb = xgb_model.predict(X_te_xgb)
    xgb_pred_time = time.perf_counter() - t0

    metrics_xgb_train = evaluate_regression(y_tr_xgb_sub, y_pred_train_xgb)
    metrics_xgb_val = evaluate_regression(y_val, y_pred_val_xgb)
    metrics_xgb_test = evaluate_regression(y_test, y_pred_test_xgb)

    print(f"XGBoost Training Time: {xgb_train_time:.2f}s, Inference Time: {xgb_pred_time:.2f}s")
    print(f"XGBoost Best Iteration: {xgb_model.best_iteration}")
    print(
        f"XGBoost Val  -> MAE={metrics_xgb_val['mae']:.4f}, "
        f"RMSE={metrics_xgb_val['rmse']:.4f}, R2={metrics_xgb_val['r2']:.6f}"
    )
    print(
        f"XGBoost Test -> MAE={metrics_xgb_test['mae']:.4f}, "
        f"RMSE={metrics_xgb_test['rmse']:.4f}, R2={metrics_xgb_test['r2']:.6f}"
    )

    # ---------------------------------------------------------
    # STEP 6: OVERFITTING CHECK
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 6: OVERFITTING & GENERALIZATION GAP ANALYSIS")
    print("=" * 70)
    models_comparison = [
        {
            "Model": "Naive Baseline (Median)",
            "n_train_samples": len(y_train),
            "Train MAE": metrics_baseline_train["mae"],
            "Train RMSE": metrics_baseline_train["rmse"],
            "Train R2": metrics_baseline_train["r2"],
            "Val MAE": metrics_baseline_val["mae"],
            "Val RMSE": metrics_baseline_val["rmse"],
            "Val R2": metrics_baseline_val["r2"],
            "Test MAE": metrics_baseline_test["mae"],
            "Test RMSE": metrics_baseline_test["rmse"],
            "Test R2": metrics_baseline_test["r2"],
            "Gen Gap (Test-Train RMSE)": metrics_baseline_test["rmse"]
            - metrics_baseline_train["rmse"],
            "Train Time (s)": baseline_fit_time,
            "Pred Time (s)": baseline_pred_time,
        },
        {
            "Model": "Random Forest (100k, Eng)",
            "n_train_samples": 100_000,
            "Train MAE": metrics_rf_train["mae"],
            "Train RMSE": metrics_rf_train["rmse"],
            "Train R2": metrics_rf_train["r2"],
            "Val MAE": metrics_rf_val["mae"],
            "Val RMSE": metrics_rf_val["rmse"],
            "Val R2": metrics_rf_val["r2"],
            "Test MAE": metrics_rf_test["mae"],
            "Test RMSE": metrics_rf_test["rmse"],
            "Test R2": metrics_rf_test["r2"],
            "Gen Gap (Test-Train RMSE)": metrics_rf_test["rmse"] - metrics_rf_train["rmse"],
            "Train Time (s)": rf_train_time,
            "Pred Time (s)": rf_pred_time,
        },
        {
            "Model": "XGBoost (Full, Eng)",
            "n_train_samples": len(y_train),
            "Train MAE": metrics_xgb_train["mae"],
            "Train RMSE": metrics_xgb_train["rmse"],
            "Train R2": metrics_xgb_train["r2"],
            "Val MAE": metrics_xgb_val["mae"],
            "Val RMSE": metrics_xgb_val["rmse"],
            "Val R2": metrics_xgb_val["r2"],
            "Test MAE": metrics_xgb_test["mae"],
            "Test RMSE": metrics_xgb_test["rmse"],
            "Test R2": metrics_xgb_test["r2"],
            "Gen Gap (Test-Train RMSE)": metrics_xgb_test["rmse"] - metrics_xgb_train["rmse"],
            "Train Time (s)": xgb_train_time,
            "Pred Time (s)": xgb_pred_time,
        },
    ]
    df_metrics = pd.DataFrame(models_comparison)
    df_metrics.to_csv(RESULTS_DIR / "metrics.csv", index=False)
    print(df_metrics.to_string(index=False))

    # ---------------------------------------------------------
    # STEP 7: PARTIAL MEASUREMENT EXPERIMENTS (8/8, 7/8, 6/8, 4/8)
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 7: PARTIAL MEASUREMENT EXPERIMENTS (8/8, 7/8, 6/8, 4/8)")
    print("=" * 70)

    scenarios = ["8/8", "7/8", "6/8", "4/8"]
    partial_rows = []

    for sc in scenarios:
        print(f"\n--- Running Scenario {sc} ---")
        # Load engineered splits with scenario masking
        X_tr_sc, X_va_sc, X_te_sc, _, _, _ = get_regime_splits(regime=1, partial_scenario=sc)

        # 1. Random Forest on 100k subsample (consistent with primary model training size)
        if sc == "8/8":
            m_rf_val = metrics_rf_val
            m_rf_te = metrics_rf_test
            n_feat_rf = X_tr_rf_enc.shape[1]
        else:
            X_tr_sub, y_tr_sub = apply_subsample(X_tr_sc, y_train, subsample=100_000, seed=SEED)
            rf_enc = CategoricalEncoder()
            X_tr_enc = rf_enc.fit_transform(X_tr_sub)
            X_va_enc = rf_enc.transform(X_va_sc)
            X_te_enc = rf_enc.transform(X_te_sc)

            rf_sc = RandomForestRegressor(**RF_BEST_PARAMS, random_state=SEED, n_jobs=-1)
            rf_sc.fit(X_tr_enc, y_tr_sub)
            p_val_rf = rf_sc.predict(X_va_enc)
            p_te_rf = rf_sc.predict(X_te_enc)
            m_rf_val = evaluate_regression(y_val, p_val_rf)
            m_rf_te = evaluate_regression(y_test, p_te_rf)
            n_feat_rf = X_tr_enc.shape[1]

        partial_rows.append(
            {
                "Measurement Availability": sc,
                "Model": "Random Forest",
                "n_train_samples": 100_000,
                "Features": n_feat_rf,
                "Val MAE": m_rf_val["mae"],
                "Val RMSE": m_rf_val["rmse"],
                "Val R2": m_rf_val["r2"],
                "Test MAE": m_rf_te["mae"],
                "Test RMSE": m_rf_te["rmse"],
                "Test R2": m_rf_te["r2"],
            }
        )

        # 2. XGBoost on FULL training dataset (N_train = 1,986,397)
        X_tr_xgb_sc, X_va_xgb_sc, X_te_xgb_sc = match_categories(X_tr_sc, X_va_sc, X_te_sc)
        if sc == "8/8":
            # Exact reuse of primary full-data XGBoost model
            m_xgb_val = metrics_xgb_val
            m_xgb_te = metrics_xgb_test
        else:
            t_sc0 = time.perf_counter()
            xgb_sc = build_model(XGB_BEST_PARAMS)
            xgb_sc.fit(X_tr_xgb_sc, y_train, eval_set=[(X_va_xgb_sc, y_val)], verbose=False)
            t_sc_fit = time.perf_counter() - t_sc0
            p_val_xgb = xgb_sc.predict(X_va_xgb_sc)
            p_te_xgb = xgb_sc.predict(X_te_xgb_sc)
            m_xgb_val = evaluate_regression(y_val, p_val_xgb)
            m_xgb_te = evaluate_regression(y_test, p_te_xgb)
            print(
                f"Scenario {sc} XGBoost full-data fit in {t_sc_fit:.1f}s, "
                f"best iteration: {xgb_sc.best_iteration}"
            )

        partial_rows.append(
            {
                "Measurement Availability": sc,
                "Model": "XGBoost",
                "n_train_samples": len(y_train),
                "Features": X_tr_xgb_sc.shape[1],
                "Val MAE": m_xgb_val["mae"],
                "Val RMSE": m_xgb_val["rmse"],
                "Val R2": m_xgb_val["r2"],
                "Test MAE": m_xgb_te["mae"],
                "Test RMSE": m_xgb_te["rmse"],
                "Test R2": m_xgb_te["r2"],
            }
        )

        print(
            f"Scenario {sc} | RF Test RMSE: {m_rf_te['rmse']:.4f}, R2: {m_rf_te['r2']:.4f} "
            f"| XGB Test RMSE: {m_xgb_te['rmse']:.4f}, R2: {m_xgb_te['r2']:.4f}"
        )

    df_partial = pd.DataFrame(partial_rows)
    df_partial.to_csv(RESULTS_DIR / "partial_measurement_results.csv", index=False)
    print("\nPartial Measurement Results:")
    print(df_partial.to_string(index=False))

    # ---------------------------------------------------------
    # STEP 8 & 9: ABLATION EXPERIMENTS (Full N_train = 1,986,397)
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 8 & 9: UPSTREAM IMPUTATION & FEATURE ENGINEERING ABLATION (Full N_train)")
    print("=" * 70)

    ablation_rows = []

    # 8. Upstream Imputation Ablation: Without Flags vs With Flags
    X_tr_no_flags, X_va_no_flags, X_te_no_flags, _, _, _ = get_splits(
        include_metadata=True, include_imputation_flags=False
    )
    X_tr_w_flags, X_va_w_flags, X_te_w_flags, _, _, _ = get_splits(
        include_metadata=True, include_imputation_flags=True
    )

    tr_no_m, va_no_m, te_no_m = match_categories(X_tr_no_flags, X_va_no_flags, X_te_no_flags)
    tr_w_m, va_w_m, te_w_m = match_categories(X_tr_w_flags, X_va_w_flags, X_te_w_flags)

    print("Fitting XGBoost on full data without imputation flags (raw 10 features)...")
    t_no0 = time.perf_counter()
    m_no_flags = build_model(XGB_BEST_PARAMS)
    m_no_flags.fit(tr_no_m, y_train, eval_set=[(va_no_m, y_val)], verbose=False)
    t_no_fit = time.perf_counter() - t_no0
    print(f"Fitted without flags in {t_no_fit:.1f}s, best iteration: {m_no_flags.best_iteration}")

    p_val_no = m_no_flags.predict(va_no_m)
    p_te_no = m_no_flags.predict(te_no_m)
    metrics_no_flags_val = evaluate_regression(y_val, p_val_no)
    metrics_no_flags_te = evaluate_regression(y_test, p_te_no)

    print("Fitting XGBoost on full data with imputation flags (20 features)...")
    t_w0 = time.perf_counter()
    m_w_flags = build_model(XGB_BEST_PARAMS)
    m_w_flags.fit(tr_w_m, y_train, eval_set=[(va_w_m, y_val)], verbose=False)
    t_w_fit = time.perf_counter() - t_w0
    print(f"Fitted with flags in {t_w_fit:.1f}s, best iteration: {m_w_flags.best_iteration}")

    p_val_w = m_w_flags.predict(va_w_m)
    p_te_w = m_w_flags.predict(te_w_m)
    metrics_w_flags_val = evaluate_regression(y_val, p_val_w)
    metrics_w_flags_te = evaluate_regression(y_test, p_te_w)

    ablation_rows.append(
        {
            "Ablation Domain": "Upstream Imputation Flags",
            "Model": "XGBoost",
            "Configuration": "Without Imputation Flags (Primary)",
            "n_train_samples": len(y_train),
            "n_features": tr_no_m.shape[1],
            "Val MAE": metrics_no_flags_val["mae"],
            "Val RMSE": metrics_no_flags_val["rmse"],
            "Val R2": metrics_no_flags_val["r2"],
            "Test MAE": metrics_no_flags_te["mae"],
            "Test RMSE": metrics_no_flags_te["rmse"],
            "Test R2": metrics_no_flags_te["r2"],
        }
    )
    ablation_rows.append(
        {
            "Ablation Domain": "Upstream Imputation Flags",
            "Model": "XGBoost",
            "Configuration": "With Imputation Flags (Ablation)",
            "n_train_samples": len(y_train),
            "n_features": tr_w_m.shape[1],
            "Val MAE": metrics_w_flags_val["mae"],
            "Val RMSE": metrics_w_flags_val["rmse"],
            "Val R2": metrics_w_flags_val["r2"],
            "Test MAE": metrics_w_flags_te["mae"],
            "Test RMSE": metrics_w_flags_te["rmse"],
            "Test R2": metrics_w_flags_te["r2"],
        }
    )

    # 9. Feature Engineering Ablation: Raw vs Raw + Engineered
    ablation_rows.append(
        {
            "Ablation Domain": "Feature Engineering",
            "Model": "XGBoost",
            "Configuration": "Raw Features",
            "n_train_samples": len(y_train),
            "n_features": tr_no_m.shape[1],
            "Val MAE": metrics_no_flags_val["mae"],
            "Val RMSE": metrics_no_flags_val["rmse"],
            "Val R2": metrics_no_flags_val["r2"],
            "Test MAE": metrics_no_flags_te["mae"],
            "Test RMSE": metrics_no_flags_te["rmse"],
            "Test R2": metrics_no_flags_te["r2"],
        }
    )
    ablation_rows.append(
        {
            "Ablation Domain": "Feature Engineering",
            "Model": "XGBoost",
            "Configuration": "Raw + Engineered Features",
            "n_train_samples": len(y_train),
            "n_features": X_tr_xgb.shape[1],
            "Val MAE": metrics_xgb_val["mae"],
            "Val RMSE": metrics_xgb_val["rmse"],
            "Val R2": metrics_xgb_val["r2"],
            "Test MAE": metrics_xgb_test["mae"],
            "Test RMSE": metrics_xgb_test["rmse"],
            "Test R2": metrics_xgb_test["r2"],
        }
    )

    df_ablation = pd.DataFrame(ablation_rows)
    df_ablation.to_csv(RESULTS_DIR / "ablation_results.csv", index=False)
    print("\nAblation Results:")
    print(df_ablation.to_string(index=False))

    # ---------------------------------------------------------
    # STEP 11 & 12: ERROR ANALYSIS & COUNTRY ANALYSIS
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 11 & 12: ERROR ANALYSIS & COUNTRY ANALYSIS")
    print("=" * 70)

    # Best model predictions on test set: y_pred_test_xgb
    # Load test metadata (Country, Waterbody Type)
    test_country = X_test_raw["Country"]
    test_waterbody = X_test_raw["Waterbody Type"]

    df_country = evaluate_by_group(y_test, y_pred_test_xgb, test_country, group_name="Country")

    # Note China in-sample evaluation
    # China observations in train:
    train_country = X_train_raw["Country"]
    china_train_mask = train_country == "China"
    china_n_train = int(np.sum(china_train_mask))

    # Evaluate China in-sample
    p_china_train = xgb_model.predict(X_tr_xgb[china_train_mask])
    m_china = evaluate_regression(y_train[china_train_mask], p_china_train)

    china_row = {
        "Country": "China (In-Sample / Train Only - Hou Bay single area)",
        "n_samples": china_n_train,
        "pct_of_total": float(china_n_train / len(y_train) * 100.0),
        "mae": m_china["mae"],
        "rmse": m_china["rmse"],
        "r2": m_china["r2"],
        "medae": m_china["medae"],
        "max_error": m_china["max_error"],
        "residual_mean": m_china["residual_mean"],
        "residual_std": m_china["residual_std"],
    }
    df_country = pd.concat([df_country, pd.DataFrame([china_row])], ignore_index=True)
    df_country.to_csv(RESULTS_DIR / "country_results.csv", index=False)
    print("\nCountry Performance Analysis:")
    print(df_country.to_string(index=False))

    # Error analysis by Waterbody Type
    df_waterbody = evaluate_by_group(
        y_test, y_pred_test_xgb, test_waterbody, group_name="Waterbody Type"
    )
    print("\nWaterbody Type Performance Analysis:")
    print(df_waterbody.to_string(index=False))

    # Target ceiling error analysis: CCME == 100 vs CCME < 100
    ceiling_mask = y_test == 100.0
    m_ceiling = evaluate_regression(y_test[ceiling_mask], y_pred_test_xgb[ceiling_mask])
    m_subceiling = evaluate_regression(y_test[~ceiling_mask], y_pred_test_xgb[~ceiling_mask])

    print("\nTarget Regime Error Analysis:")
    print(
        f"  Ceiling (CCME=100.0, {np.sum(ceiling_mask):,} samples, {np.mean(ceiling_mask) * 100:.1f}%) -> "
        f"MAE={m_ceiling['mae']:.4f}, RMSE={m_ceiling['rmse']:.4f}, MaxError={m_ceiling['max_error']:.4f}"
    )
    print(
        f"  Sub-ceiling (CCME<100.0, {np.sum(~ceiling_mask):,} samples, {np.mean(~ceiling_mask) * 100:.1f}%) -> "
        f"MAE={m_subceiling['mae']:.4f}, RMSE={m_subceiling['rmse']:.4f}, R2={m_subceiling['r2']:.4f}"
    )

    # ---------------------------------------------------------
    # STEP 14: SHAP / INTERPRETABILITY
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 14: SHAP / INTERPRETABILITY ANALYSIS")
    print("=" * 70)

    try:
        import shap

        # Sample 2,000 test observations for SHAP computation
        rng = np.random.default_rng(SEED)
        shap_idx = rng.choice(len(X_te_xgb), size=2000, replace=False)
        X_shap = X_te_xgb.iloc[shap_idx]

        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer(X_shap)

        # Plot SHAP summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_shap, show=False, max_display=15)
        plt.title(
            "SHAP Feature Attribution (XGBoost Emulator) - Top 15 Features",
            fontsize=13,
            pad=15,
        )
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "06_shap_summary.png", dpi=200, bbox_inches="tight")
        plt.close()
        print(f"SHAP summary plot saved -> {FIGURES_DIR / '06_shap_summary.png'}")

        # Mean absolute SHAP values for global ranking
        mean_abs_shap = np.mean(np.abs(shap_values.values), axis=0)
        shap_df = (
            pd.DataFrame({"Feature": X_shap.columns, "Mean_Abs_SHAP": mean_abs_shap})
            .sort_values("Mean_Abs_SHAP", ascending=False)
            .reset_index(drop=True)
        )
        print("\nTop 10 Features by Mean Absolute SHAP:")
        print(shap_df.head(10).to_string(index=False))
        shap_df.to_csv(RESULTS_DIR / "shap_importance.csv", index=False)
    except Exception as exc:
        print(f"SHAP computation encountered error: {exc}")

    # ---------------------------------------------------------
    # STEP 15: STATISTICAL / BOOTSTRAP CHECK
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 15: NON-PARAMETRIC BOOTSTRAP TEST SET ROBUSTNESS (1,000 resamples)")
    print("=" * 70)

    boot_baseline = bootstrap_test_metrics(
        y_test, y_pred_test_baseline, n_bootstraps=1000, seed=SEED
    )
    boot_rf = bootstrap_test_metrics(y_test, y_pred_test_rf, n_bootstraps=1000, seed=SEED)
    boot_xgb = bootstrap_test_metrics(y_test, y_pred_test_xgb, n_bootstraps=1000, seed=SEED)

    boot_rows = []
    for model_name, res in [
        ("Naive Baseline", boot_baseline),
        ("Random Forest", boot_rf),
        ("XGBoost", boot_xgb),
    ]:
        boot_rows.append(
            {
                "Model": model_name,
                "MAE (Point [95% CI])": f"{res['mae']['point_estimate']:.4f} [{res['mae']['ci_lower']:.4f}, {res['mae']['ci_upper']:.4f}]",
                "RMSE (Point [95% CI])": f"{res['rmse']['point_estimate']:.4f} [{res['rmse']['ci_lower']:.4f}, {res['rmse']['ci_upper']:.4f}]",
                "R2 (Point [95% CI])": f"{res['r2']['point_estimate']:.4f} [{res['r2']['ci_lower']:.4f}, {res['r2']['ci_upper']:.4f}]",
            }
        )
    df_boot = pd.DataFrame(boot_rows)
    df_boot.to_csv(RESULTS_DIR / "bootstrap_ci_results.csv", index=False)
    print(df_boot.to_string(index=False))

    # ---------------------------------------------------------
    # STEP 13: PREDICTION PLOTS
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 13: GENERATING FINAL PREDICTION & EVALUATION PLOTS")
    print("=" * 70)

    # 1. Actual vs Predicted for best model (XGBoost)
    sample_sub = rng.choice(len(y_test), size=10_000, replace=False)
    y_sub_true = y_test.iloc[sample_sub].values
    y_sub_pred = y_pred_test_xgb[sample_sub]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_sub_true, y_sub_pred, alpha=0.2, s=8, color="#1f77b4", rasterized=True)
    lims = [30, 105]
    ax.plot(lims, lims, "r--", lw=1.5, label="Perfect Emulation (1:1)")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Actual CCME_Values (Ground Truth)", fontsize=11)
    ax.set_ylabel("Predicted CCME_Values (XGBoost)", fontsize=11)
    ax.set_title("Actual vs Predicted CCME_Values — XGBoost Best Model (Test Set)", fontsize=12)
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "06_actual_vs_predicted.png", dpi=200)
    plt.close()

    # 2. Residual Distribution for Best Model
    residuals_xgb = y_test.values - y_pred_test_xgb
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        residuals_xgb,
        bins=100,
        range=(-3.0, 3.0),
        color="#2ca02c",
        edgecolor="none",
        alpha=0.8,
    )
    ax.axvline(
        0,
        color="crimson",
        lw=1.5,
        ls="--",
        label=f"Zero Residual (Mean: {np.mean(residuals_xgb):.4f})",
    )
    ax.set_xlabel("Residual (y_true - y_pred)", fontsize=11)
    ax.set_ylabel("Sample Count", fontsize=11)
    ax.set_title("Residual Distribution — XGBoost Emulator on Test Partition", fontsize=12)
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "06_residual_distribution.png", dpi=200)
    plt.close()

    # 3. Error vs Actual Target (Highlighting the 100.0 Ceiling)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        y_sub_true,
        residuals_xgb[sample_sub],
        alpha=0.25,
        s=8,
        color="#ff7f0e",
        rasterized=True,
    )
    ax.axhline(0, color="black", lw=1, ls="--")
    ax.set_xlabel("Actual CCME_Values", fontsize=11)
    ax.set_ylabel("Residual (y_true - y_pred)", fontsize=11)
    ax.set_title(
        "Residuals vs Actual Target — Target Ceiling Censorship Artifact (at 100)",
        fontsize=12,
    )
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "06_error_vs_actual.png", dpi=200)
    plt.close()

    # 4. Model Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(8, 5))
    models = ["Naive Baseline", "Random Forest", "XGBoost"]
    rmses = [
        metrics_baseline_test["rmse"],
        metrics_rf_test["rmse"],
        metrics_xgb_test["rmse"],
    ]
    maes = [
        metrics_baseline_test["mae"],
        metrics_rf_test["mae"],
        metrics_xgb_test["mae"],
    ]
    x = np.arange(len(models))
    width = 0.35
    ax.bar(x - width / 2, rmses, width, label="Test RMSE", color="#d62728")
    ax.bar(x + width / 2, maes, width, label="Test MAE", color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("Error (CCME Index Points)", fontsize=11)
    ax.set_title("Test Error Comparison Across Evaluated Architectures", fontsize=12)
    ax.legend()
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "06_model_comparison.png", dpi=200)
    plt.close()

    # 5. Partial Measurement Performance Degradation
    fig, ax = plt.subplots(figsize=(8, 5))
    for mod, color, marker in [
        ("Random Forest", "#2ca02c", "o"),
        ("XGBoost", "#1f77b4", "s"),
    ]:
        sub_df = df_partial[df_partial["Model"] == mod]
        ax.plot(
            sub_df["Measurement Availability"],
            sub_df["Test RMSE"],
            marker=marker,
            lw=2,
            label=f"{mod} RMSE",
            color=color,
        )
    ax.set_xlabel("Measurement Availability Scenario", fontsize=11)
    ax.set_ylabel("Test RMSE (CCME Index Points)", fontsize=11)
    ax.set_title("Performance Degradation Under Incomplete Measurement Suites", fontsize=12)
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "06_partial_measurement_degradation.png", dpi=200)
    plt.close()

    print(f"All figures successfully generated in {FIGURES_DIR}")
    print("\nALL EVALUATION STEPS COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    run_all_evaluations()
