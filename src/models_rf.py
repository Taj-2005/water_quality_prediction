"""Random Forest model implementation.   Owner: Himanshu Pandey.

Theoretical foundation:
  - Bagging (Bootstrap Aggregation) & Variance Reduction (Breiman, 2001).
  - Out-of-Bag (OOB) error monitoring on in-sample bagging draws.
  - Hyperparameter tuning conducted EXCLUSIVELY on the validation partition.
  - Holdout test evaluation performed strictly once on the final selected model.

Experimental Regimes & Scenarios:
  - 8/8 Scenario (Regime 1): ML surrogate / formula emulator benchmark approximating
    the Canadian Council of Ministers of the Environment (CCME) deterministic formula.
    CRITICAL CAVEAT: Must NOT be described as independent environmental prediction.
  - Partial Measurement Scenarios (7/8, 6/8, 4/8): Testing model resilience under
    partial parameter availability with dependent derived features cleanly masked.
  - Regime 2: Independent environmental prediction using strictly non-chemical
    metadata and spatiotemporal covariates.
  - Feature ablation: Raw features vs. Engineered stoichiometric & hydrochemical features.

Logging & Evaluation:
  - Employs shared evaluate() API with documented fallback to scikit-learn metrics.
  - Appends experiment configuration and validation/test metrics to reports/rf_results.csv.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.preprocessing import OneHotEncoder

    HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    RandomForestRegressor = None  # type: ignore[assignment, misc]
    mean_absolute_error = None  # type: ignore[assignment]
    mean_squared_error = None  # type: ignore[assignment]
    r2_score = None  # type: ignore[assignment]
    OneHotEncoder = None  # type: ignore[assignment, misc]
    HAS_SKLEARN = False

from src.config import REPORTS, SEED
from src.evaluate_api import evaluate, get_splits
from src.features import get_regime_splits

REPORT_PATH = REPORTS / "rf_results.csv"

# Default hyperparameter grid for validation tuning
DEFAULT_PARAM_GRID: list[dict[str, Any]] = [
    {"n_estimators": 50, "max_depth": 10, "min_samples_leaf": 5, "max_features": "sqrt"},
    {"n_estimators": 100, "max_depth": 15, "min_samples_leaf": 5, "max_features": "sqrt"},
    {"n_estimators": 100, "max_depth": 20, "min_samples_leaf": 2, "max_features": "sqrt"},
    {"n_estimators": 150, "max_depth": 15, "min_samples_leaf": 1, "max_features": 0.5},
]


def compute_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Compute MAE, RMSE, and R2 using shared evaluate() API with fallback.

    If Satya's src.metrics.regression_metrics raises NotImplementedError, falls
    back to standard scikit-learn regression calculations.
    """
    try:
        return evaluate(y_true, y_pred)
    except (NotImplementedError, ImportError):
        y_t = np.asarray(y_true, dtype=float)
        y_p = np.asarray(y_pred, dtype=float)
        if HAS_SKLEARN and mean_absolute_error is not None:
            mae = float(mean_absolute_error(y_t, y_p))
            rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
            r2 = float(r2_score(y_t, y_p))
        else:
            mae = float(np.mean(np.abs(y_t - y_p)))
            mse = float(np.mean((y_t - y_p) ** 2))
            rmse = float(np.sqrt(mse))
            ss_res = float(np.sum((y_t - y_p) ** 2))
            ss_tot = float(np.sum((y_t - np.mean(y_t)) ** 2))
            if ss_tot > 0:
                r2 = float(1.0 - (ss_res / ss_tot))
            else:
                r2 = 1.0 if ss_res == 0.0 else 0.0
        return {"mae": mae, "rmse": rmse, "r2": r2}


class CategoricalEncoder:
    """Train-only fitted one-hot encoder for categorical metadata columns.

    Guarantees:
      - Learns categories strictly from X_train.
      - Encodes X_val and X_test without refitting.
      - Unseen categories at inference time are safely ignored (handle_unknown='ignore').
      - Returns a pandas DataFrame with aligned row indices and float64 dtypes.
    """

    def __init__(self) -> None:
        self.cat_cols_: list[str] = []
        self.num_cols_: list[str] = []
        self.encoder_: OneHotEncoder | None = None
        self.encoded_feature_names_: list[str] = []
        self.is_fitted_: bool = False

    def fit(self, X: pd.DataFrame) -> CategoricalEncoder:
        self.cat_cols_ = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
        self.num_cols_ = [c for c in X.columns if c not in self.cat_cols_]

        if self.cat_cols_:
            if not HAS_SKLEARN or OneHotEncoder is None:
                raise ImportError(
                    "CategoricalEncoder requires scikit-learn. "
                    "Install dependencies with: pip install -r requirements.txt"
                )
            self.encoder_ = OneHotEncoder(
                sparse_output=False,
                handle_unknown="ignore",
            )
            self.encoder_.fit(X[self.cat_cols_].astype(str))
            self.encoded_feature_names_ = list(self.encoder_.get_feature_names_out(self.cat_cols_))
        else:
            self.encoder_ = None
            self.encoded_feature_names_ = []

        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError(
                "CategoricalEncoder must be fitted on training data before transform."
            )

        num_part = X[self.num_cols_].copy()

        if self.cat_cols_ and self.encoder_ is not None:
            cat_data = X[self.cat_cols_].astype(str)
            encoded_array = self.encoder_.transform(cat_data)
            encoded_df = pd.DataFrame(
                encoded_array,
                columns=self.encoded_feature_names_,
                index=X.index,
                dtype=np.float64,
            )
            return pd.concat([num_part, encoded_df], axis=1)

        return num_part

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)


def load_rf_splits(
    regime: int = 1,
    feature_set: str = "engineered",
    partial_scenario: str = "8/8",
    splits_dir: Path | str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Load train, val, and test splits configured for Random Forest experiments.

    Parameters
    ----------
    regime : int, default 1
        1 = Formula Emulation / Benchmark Surrogate.
        2 = Independent Environmental Prediction.
    feature_set : str, default 'engineered'
        'raw' or 'engineered'.
    partial_scenario : str, default '8/8'
        '8/8', '7/8', '6/8', or '4/8'.
    splits_dir : Path | str | None, optional
        Custom directory holding splits.parquet.
    """
    if feature_set == "raw":
        if regime == 2:
            X_tr, X_va, X_te, y_tr, y_va, y_te = get_splits(
                splits_dir=splits_dir,
                include_metadata=True,
                include_imputation_flags=False,
                partial_scenario=partial_scenario,
            )
            cols = [c for c in ["Country", "Waterbody Type"] if c in X_tr.columns]
            return X_tr[cols], X_va[cols], X_te[cols], y_tr, y_va, y_te
        else:
            return get_splits(
                splits_dir=splits_dir,
                include_metadata=True,
                include_imputation_flags=False,
                partial_scenario=partial_scenario,
            )
    elif feature_set == "engineered":
        return get_regime_splits(
            regime=regime,
            splits_dir=splits_dir,
            partial_scenario=partial_scenario,
            include_ratios=True,
            include_logs=True,
            include_temporal=True,
        )
    else:
        raise ValueError(f"Unknown feature_set '{feature_set}'. Must be 'raw' or 'engineered'.")


def apply_subsample(
    X: pd.DataFrame,
    y: pd.Series,
    subsample: int | float | None,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.Series]:
    """Deterministically subsample training partitions for rapid experimentation."""
    if subsample is None:
        return X, y

    n_total = len(X)
    if isinstance(subsample, float):
        if not (0.0 < subsample <= 1.0):
            raise ValueError(f"Float subsample must be in (0, 1], got {subsample}")
        n_sample = int(n_total * subsample)
    elif isinstance(subsample, int):
        if subsample <= 0:
            raise ValueError(f"Integer subsample must be positive, got {subsample}")
        n_sample = min(subsample, n_total)
    else:
        raise TypeError(f"subsample must be int, float, or None, got {type(subsample)}")

    if n_sample >= n_total:
        return X, y

    rng = np.random.default_rng(seed)
    indices = rng.choice(n_total, size=n_sample, replace=False)
    return X.iloc[indices].copy(), y.iloc[indices].copy()


def get_experiment_label(regime: int, partial_scenario: str) -> str:
    """Generate transparent, academically honest experiment labels.

    Distinguishes formula emulation / benchmark surrogate from independent prediction.
    """
    if regime == 1:
        if partial_scenario == "8/8":
            return (
                "surrogate_emulator_benchmark (formula approximation, NOT independent prediction)"
            )
        return f"partial_measurement_emulation_{partial_scenario}"
    elif regime == 2:
        return "independent_environmental_prediction (non-chemical covariates)"
    return f"regime_{regime}_{partial_scenario}"


def record_results(
    results: dict[str, Any],
    output_path: Path | str | None = None,
) -> Path:
    """Append experiment configuration and evaluation metrics to CSV report."""
    target = Path(output_path) if output_path else REPORT_PATH
    target.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment_label": results["config"]["experiment_label"],
        "regime": results["config"]["regime"],
        "feature_set": results["config"]["feature_set"],
        "partial_scenario": results["config"]["partial_scenario"],
        "n_train_samples": results["config"]["n_train_samples"],
        "n_features": results["config"]["n_features"],
        "n_estimators": results["best_params"]["n_estimators"],
        "max_depth": results["best_params"]["max_depth"],
        "min_samples_leaf": results["best_params"]["min_samples_leaf"],
        "max_features": results["best_params"]["max_features"],
        "val_mae": results["val_metrics"]["mae"],
        "val_rmse": results["val_metrics"]["rmse"],
        "val_r2": results["val_metrics"]["r2"],
        "test_mae": results["test_metrics"]["mae"],
        "test_rmse": results["test_metrics"]["rmse"],
        "test_r2": results["test_metrics"]["r2"],
        "oob_score": results["oob_score"],
    }

    df_row = pd.DataFrame([row])
    if target.exists() and target.stat().st_size > 0:
        df_row.to_csv(target, mode="a", header=False, index=False)
    else:
        df_row.to_csv(target, mode="w", header=True, index=False)

    return target


def train(
    regime: int = 1,
    feature_set: str = "engineered",
    partial_scenario: str = "8/8",
    n_estimators: int = 100,
    max_depth: int | None = 15,
    min_samples_leaf: int = 5,
    max_features: str | float | int = "sqrt",
    random_state: int = SEED,
    tune: bool = False,
    param_grid: list[dict[str, Any]] | None = None,
    subsample: int | float | None = None,
    oob_score: bool = False,
    save_reports: bool = True,
    output_path: Path | str | None = None,
    splits_dir: Path | str | None = None,
    n_jobs: int = -1,
) -> dict[str, Any]:
    """Train and evaluate Random Forest model with validation-based model selection.

    Parameters
    ----------
    regime : int, default 1
        1 = Formula Emulation / Benchmark Surrogate.
        2 = Independent Environmental Prediction.
    feature_set : str, default 'engineered'
        'raw' or 'engineered'.
    partial_scenario : str, default '8/8'
        Ablation setting ('8/8', '7/8', '6/8', '4/8').
    n_estimators : int, default 100
        Number of decision trees in forest.
    max_depth : int | None, default 15
        Maximum tree depth. None allows full growth until min_samples_leaf.
    min_samples_leaf : int, default 5
        Minimum samples required at leaf node (regularization).
    max_features : str | float | int, default 'sqrt'
        Number of features considered per split.
    random_state : int, default SEED (42)
        Reproducibility seed.
    tune : bool, default False
        If True, executes hyperparameter search on VALIDATION set only.
    param_grid : list[dict[str, Any]] | None, optional
        Custom list of parameter dictionaries to tune over on validation data.
    subsample : int | float | None, default None
        Optional deterministic subsample of X_train for fast verification.
    oob_score : bool, default False
        Whether to compute out-of-bag generalization estimate on training bootstrap.
    save_reports : bool, default True
        Whether to append results to reports/rf_results.csv.
    output_path : Path | str | None, optional
        Custom path for CSV report.
    splits_dir : Path | str | None, optional
        Path to custom splits directory.
    n_jobs : int, default -1
        Number of parallel CPU jobs.

    Returns
    -------
    dict[str, Any]
        Dictionary containing fitted model, encoder, best params, validation metrics,
        final test metrics, feature names, feature importances, and run configuration.
    """
    if not HAS_SKLEARN or RandomForestRegressor is None:
        raise ImportError(
            "Random Forest modeling requires scikit-learn. "
            "Install dependencies with: pip install -r requirements.txt"
        )

    # 1. Load data via shared pipeline APIs
    X_train, X_val, X_test, y_train, y_val, y_test = load_rf_splits(
        regime=regime,
        feature_set=feature_set,
        partial_scenario=partial_scenario,
        splits_dir=splits_dir,
    )

    # 2. Apply deterministic subsample if requested (train only)
    X_train_sub, y_train_sub = apply_subsample(
        X_train, y_train, subsample=subsample, seed=random_state
    )

    # 3. Fit categorical encoder STRICTLY on training split
    encoder = CategoricalEncoder()
    X_train_enc = encoder.fit_transform(X_train_sub)
    X_val_enc = encoder.transform(X_val)
    X_test_enc = encoder.transform(X_test)

    feature_names = list(X_train_enc.columns)

    # 4. Hyperparameter tuning on VALIDATION SET ONLY (Test set strictly untouched)
    if tune:
        candidates = param_grid if param_grid is not None else DEFAULT_PARAM_GRID
        best_model = None
        best_params = None
        best_val_rmse = float("inf")
        best_val_metrics = None

        for params in candidates:
            rf = RandomForestRegressor(
                **params,
                random_state=random_state,
                n_jobs=n_jobs,
                oob_score=oob_score,
            )
            rf.fit(X_train_enc, y_train_sub)
            val_preds = rf.predict(X_val_enc)
            val_metrics = compute_metrics(y_val, val_preds)

            if val_metrics["rmse"] < best_val_rmse:
                best_val_rmse = val_metrics["rmse"]
                best_model = rf
                best_params = params
                best_val_metrics = val_metrics

        model = best_model
        final_params = best_params
        final_val_metrics = best_val_metrics
    else:
        final_params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
        }
        model = RandomForestRegressor(
            **final_params,
            random_state=random_state,
            n_jobs=n_jobs,
            oob_score=oob_score,
        )
        model.fit(X_train_enc, y_train_sub)
        val_preds = model.predict(X_val_enc)
        final_val_metrics = compute_metrics(y_val, val_preds)

    # 5. Out-of-Bag (OOB) score if enabled
    oob_r2 = float(model.oob_score_) if oob_score and hasattr(model, "oob_score_") else None

    # 6. Evaluate on TEST SET EXACTLY ONCE
    test_preds = model.predict(X_test_enc)
    final_test_metrics = compute_metrics(y_test, test_preds)

    # 7. Package results
    label = get_experiment_label(regime, partial_scenario)
    feature_importances = dict(
        zip(feature_names, [float(x) for x in model.feature_importances_], strict=True)
    )

    results: dict[str, Any] = {
        "model": model,
        "encoder": encoder,
        "best_params": final_params,
        "val_metrics": final_val_metrics,
        "test_metrics": final_test_metrics,
        "oob_score": oob_r2,
        "feature_names": feature_names,
        "feature_importances": feature_importances,
        "config": {
            "regime": regime,
            "feature_set": feature_set,
            "partial_scenario": partial_scenario,
            "n_train_samples": len(X_train_sub),
            "n_features": len(feature_names),
            "experiment_label": label,
        },
    }

    # 8. Append to reports/rf_results.csv if requested
    if save_reports:
        record_results(results, output_path=output_path)

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Random Forest on Water Quality Prediction dataset."
    )
    parser.add_argument(
        "--regime",
        type=int,
        default=1,
        choices=[1, 2],
        help="Regime 1 (emulation) or Regime 2 (independent)",
    )
    parser.add_argument(
        "--feature-set", type=str, default="engineered", choices=["engineered", "raw"]
    )
    parser.add_argument(
        "--partial-scenario", type=str, default="8/8", choices=["8/8", "7/8", "6/8", "4/8"]
    )
    parser.add_argument(
        "--subsample", type=int, default=None, help="Subsample count for X_train (e.g. 50000)"
    )
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter search on val")
    parser.add_argument("--oob", action="store_true", help="Compute Out-of-Bag score")
    parser.add_argument("--no-report", action="store_true", help="Skip saving to CSV")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    res = train(
        regime=args.regime,
        feature_set=args.feature_set,
        partial_scenario=args.partial_scenario,
        subsample=args.subsample,
        tune=args.tune,
        oob_score=args.oob,
        save_reports=not args.no_report,
    )
    print(f"Random Forest Training Completed: {res['config']['experiment_label']}")
    print(f"Features ({res['config']['n_features']}): {res['feature_names']}")
    print(
        f"Val Metrics:  MAE={res['val_metrics']['mae']:.4f}, RMSE={res['val_metrics']['rmse']:.4f}, R2={res['val_metrics']['r2']:.4f}"
    )
    print(
        f"Test Metrics: MAE={res['test_metrics']['mae']:.4f}, RMSE={res['test_metrics']['rmse']:.4f}, R2={res['test_metrics']['r2']:.4f}"
    )
    if res["oob_score"] is not None:
        print(f"OOB R2: {res['oob_score']:.4f}")
