"""XGBoost model.   Owner: Jivit Rana.

What this file does, in order:
  1. Loads the shared train / validation / test split (same data as everyone else).
  2. Optionally searches for good hyperparameters with Optuna (validation set only).
  3. Trains XGBoost with early stopping (validation set only).
  4. Scores the test set exactly once and saves the results to reports/xgb_results.csv.

It can also run the ablation: XGBoost with raw features vs engineered features.

Quick start:
    python -m src.models_xgb                      # train with default settings
    python -m src.models_xgb --tune               # search hyperparameters first
    python -m src.models_xgb --ablation           # raw vs engineered features

Note: in Regime 1 the 8 chemical inputs define the CCME target, so a high score means
"XGBoost can imitate the CCME formula", NOT "XGBoost predicts water quality independently".
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

import pandas as pd

from src.config import REPORTS, SEED
from src.models_rf import apply_subsample, compute_metrics, get_experiment_label, load_rf_splits

try:
    import xgboost as xgb

    HAS_XGB = True
except Exception:  # xgboost missing, or its OpenMP library (libomp) is missing on Mac
    xgb = None
    HAS_XGB = False

REPORT_PATH = REPORTS / "xgb_results.csv"
ABLATION_PATH = REPORTS / "xgb_ablation.csv"

MAX_TREES = 2000  # upper limit; early stopping usually stops far sooner
EARLY_STOP_AFTER = 50  # stop if validation error has not improved for 50 trees
TUNE_ROWS = 300_000  # each Optuna trial trains on this many rows, to keep it fast

# Sensible starting settings, used when we are not tuning.
DEFAULT_PARAMS = {
    "learning_rate": 0.1,  # shrinkage: how much each new tree is allowed to change the answer
    "max_depth": 8,  # how deep each tree can grow
    "subsample": 0.8,  # fraction of rows used for each tree
    "colsample_bytree": 0.8,  # fraction of columns used for each tree
    "min_child_weight": 5,  # minimum amount of data in a leaf
    "reg_alpha": 0.0,  # L1 regularization
    "reg_lambda": 1.0,  # L2 regularization
}


def match_categories(X_train, X_val, X_test):
    """Make Country / Waterbody Type use the same categories in all three splits.

    XGBoost needs the category codes to match. We learn the categories from the
    training set only; anything new in val/test becomes missing (NaN).
    """
    category_columns = [c for c in X_train.columns if not pd.api.types.is_numeric_dtype(X_train[c])]
    fixed = []
    for df in (X_train, X_val, X_test):
        df = df.copy()
        for col in category_columns:
            train_categories = pd.Categorical(X_train[col]).categories
            df[col] = pd.Categorical(df[col].astype(object), categories=train_categories)
        fixed.append(df)
    return fixed


def build_model(params):
    """Create an XGBoost regressor with early stopping turned on."""
    return xgb.XGBRegressor(
        n_estimators=MAX_TREES,
        early_stopping_rounds=EARLY_STOP_AFTER,
        tree_method="hist",  # fast histogram-based training
        enable_categorical=True,  # use Country / Waterbody Type directly
        objective="reg:squarederror",  # the loss we minimise
        eval_metric="rmse",
        random_state=SEED,
        n_jobs=-1,
        **params,
    )


def tune(X_train, y_train, X_val, y_val, n_trials):
    """Try n_trials random-but-smart settings with Optuna and return the best one.

    Each trial is judged on the VALIDATION set. The test set is never passed in.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    X_small, y_small = apply_subsample(X_train, y_train, TUNE_ROWS, seed=SEED)

    def score_one_trial(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 50, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 50.0, log=True),
        }
        model = build_model(params)
        model.fit(X_small, y_small, eval_set=[(X_val, y_val)], verbose=False)
        return model.best_score  # validation RMSE, lower is better

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(score_one_trial, n_trials=n_trials)
    return study.best_params


def save_results(results, output_path=None):
    """Append one row of settings + scores to the results CSV."""
    path = output_path or REPORT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **results["config"],
        "best_iteration": results["best_iteration"],
        **results["best_params"],
        "val_mae": results["val_metrics"]["mae"],
        "val_rmse": results["val_metrics"]["rmse"],
        "val_r2": results["val_metrics"]["r2"],
        "test_mae": results["test_metrics"]["mae"],
        "test_rmse": results["test_metrics"]["rmse"],
        "test_r2": results["test_metrics"]["r2"],
    }
    file_exists = path.exists() and path.stat().st_size > 0
    pd.DataFrame([row]).to_csv(
        path, mode="a" if file_exists else "w", header=not file_exists, index=False
    )


def train(
    regime=1,
    feature_set="engineered",
    partial_scenario="8/8",
    params=None,
    do_tune=False,
    n_trials=20,
    subsample=None,
    save_reports=True,
    output_path=None,
):
    """Train XGBoost and return its scores.

    Parameters
    ----------
    regime : 1 = imitate the CCME formula, 2 = use only non-chemical columns.
    feature_set : "raw" or "engineered" (Himanshu's extra features).
    partial_scenario : "8/8", "7/8", "6/8" or "4/8" measurements available.
    params : XGBoost settings to use when not tuning (defaults to DEFAULT_PARAMS).
    do_tune : search for better settings with Optuna first.
    n_trials : how many settings Optuna tries.
    subsample : use only this many training rows (handy for quick tests).
    save_reports : append the result to reports/xgb_results.csv.
    """
    if not HAS_XGB:
        raise ImportError(
            "XGBoost is not available. Run: pip install -r requirements.txt (Mac: brew install libomp)"
        )

    # 1. Load the shared split.
    X_train, X_val, X_test, y_train, y_val, y_test = load_rf_splits(
        regime=regime, feature_set=feature_set, partial_scenario=partial_scenario
    )
    X_train, X_val, X_test = match_categories(X_train, X_val, X_test)
    X_train, y_train = apply_subsample(X_train, y_train, subsample, seed=SEED)

    # 2. Pick hyperparameters (validation set only).
    if do_tune:
        best_params = tune(X_train, y_train, X_val, y_val, n_trials)
    else:
        best_params = {**DEFAULT_PARAMS, **(params or {})}

    # 3. Train. Early stopping watches the validation set.
    model = build_model(best_params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # 4. Score. The test set is used only here, once.
    val_metrics = compute_metrics(y_val, model.predict(X_val))
    test_metrics = compute_metrics(y_test, model.predict(X_test))

    feature_names = list(X_train.columns)
    results = {
        "model": model,
        "best_params": {**DEFAULT_PARAMS, **best_params},
        "best_iteration": int(model.best_iteration),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "feature_names": feature_names,
        "feature_importances": dict(
            zip(feature_names, model.feature_importances_.tolist(), strict=True)
        ),
        "config": {
            "experiment_label": get_experiment_label(regime, partial_scenario),
            "regime": regime,
            "feature_set": feature_set,
            "partial_scenario": partial_scenario,
            "n_train_samples": len(X_train),
            "n_features": len(feature_names),
            "tuned": do_tune,
            "n_trials": n_trials if do_tune else 0,
        },
    }
    if save_reports:
        save_results(results, output_path)
    return results


def run_ablation(regime=1, partial_scenario="8/8", output_path=None, **train_kwargs):
    """Train XGBoost twice, with raw and with engineered features, on the same split.

    Everything else (settings, seed, split) is identical, so any difference in the scores
    comes from the features. Returns a small table and saves it to reports/xgb_ablation.csv.
    """
    rows = []
    for feature_set in ("raw", "engineered"):
        res = train(
            regime=regime,
            feature_set=feature_set,
            partial_scenario=partial_scenario,
            **train_kwargs,
        )
        rows.append(
            {
                "regime": regime,
                "partial_scenario": partial_scenario,
                "feature_set": feature_set,
                "n_features": res["config"]["n_features"],
                "best_iteration": res["best_iteration"],
                "val_rmse": res["val_metrics"]["rmse"],
                "test_mae": res["test_metrics"]["mae"],
                "test_rmse": res["test_metrics"]["rmse"],
                "test_r2": res["test_metrics"]["r2"],
            }
        )
    table = pd.DataFrame(rows)
    path = output_path or ABLATION_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return table


def parse_args():
    parser = argparse.ArgumentParser(description="Train XGBoost on the Water Quality dataset.")
    parser.add_argument("--regime", type=int, default=1, choices=[1, 2])
    parser.add_argument("--feature-set", default="engineered", choices=["engineered", "raw"])
    parser.add_argument("--partial-scenario", default="8/8", choices=["8/8", "7/8", "6/8", "4/8"])
    parser.add_argument(
        "--subsample", type=int, default=None, help="use only this many training rows"
    )
    parser.add_argument("--tune", action="store_true", help="search hyperparameters with Optuna")
    parser.add_argument("--n-trials", type=int, default=20, help="number of Optuna trials")
    parser.add_argument(
        "--ablation", action="store_true", help="compare raw vs engineered features"
    )
    parser.add_argument("--no-report", action="store_true", help="do not write to the CSV")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    shared = {
        "do_tune": args.tune,
        "n_trials": args.n_trials,
        "subsample": args.subsample,
        "save_reports": not args.no_report,
    }
    if args.ablation:
        table = run_ablation(regime=args.regime, partial_scenario=args.partial_scenario, **shared)
        print(table.to_string(index=False))
    else:
        res = train(
            regime=args.regime,
            feature_set=args.feature_set,
            partial_scenario=args.partial_scenario,
            **shared,
        )
        val, test = res["val_metrics"], res["test_metrics"]
        print(f"XGBoost finished: {res['config']['experiment_label']}")
        print(f"Features: {res['config']['n_features']}, trees used: {res['best_iteration']}")
        print(f"Validation: MAE={val['mae']:.4f}  RMSE={val['rmse']:.4f}  R2={val['r2']:.4f}")
        print(f"Test:       MAE={test['mae']:.4f}  RMSE={test['rmse']:.4f}  R2={test['r2']:.4f}")
