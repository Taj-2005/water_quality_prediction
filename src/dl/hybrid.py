"""XGBoost + MLP residual hybrid.  Owner: Jivit Rana.

Rationale (half-page, as required by the Phase 2 work plan)
------------------------------------------------------------
XGBoost excels at capturing global, monotone relationships between
water-quality parameters and the CCME index — relationships that the formula
itself encodes almost perfectly.  Its weakness is systematic, structured
residuals: observations near the CCME ceiling (100) or at abrupt
country/waterbody transitions where the tree splits are coarse.

The MLP, on the other hand, processes the full feature vector through
dense non-linear layers and can learn smooth correction surfaces that
individual XGBoost leaves cannot represent.  By training the MLP only
on the residual (target − XGBoost_prediction), we give it a narrower,
well-defined task (predict the error, not the value) and avoid the risk
of the MLP independently re-learning what XGBoost already captured well.

This design is called a *residual hybrid* (also known as gradient-based
stacking in the literature):

    ŷ_hybrid = ŷ_xgb + ŷ_mlp_residual

CRITICAL ANTI-LEAKAGE RULE
---------------------------
XGBoost predictions used for TRAINING the MLP must be out-of-fold (OOF).
That is: for every training row, the XGBoost prediction must have been
produced by a fold of XGBoost that did NOT see that row.

If this rule is violated the MLP is taught to correct errors that XGBoost
only appears to make — but won't make on new data — and the validation and
test gains will not generalise.

Implementation
--------------
1. Split X_train / y_train into K folds (default 5).
2. For each fold k:
   a. Train XGBoost on the other K-1 folds.
   b. Predict the held-out fold → oof_xgb_pred[fold_k_indices].
3. Residuals on train = y_train − oof_xgb_pred  (no leakage).
4. Train MLP on X_train, target = residuals.
5. For validation/test:
   a. Retrain XGBoost on all of X_train (the "final XGBoost").
   b. Residual ≈ y_true − final_xgb_pred (used for MLP target on val/test).
   c. Hybrid prediction = final_xgb_pred + mlp_residual_pred.

Quick start
-----------
    python -m src.dl.hybrid                       # train & evaluate hybrid
    python -m src.dl.hybrid --ablation            # full ablation table
    python -m src.dl.hybrid --seeds 5             # 5-seed multi-run

Dependencies
------------
- Parth's src/dl/train.py  (get_loaders, train) must be merged before
  full MLP integration; until then the stub MLP in this file is used.
- Himanshu's src/dl/model.py provides the MLP architecture once merged.
"""

from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.config import REPORTS, SEED
from src.evaluate_api import evaluate, get_splits
from src.models_rf import compute_metrics
from src.models_xgb import build_model as build_xgb, match_categories

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HYBRID_RESULTS_PATH = REPORTS / "hybrid_results.csv"

# ---------------------------------------------------------------------------
# XGBoost hyperparameters (reuse Phase 1 tuned defaults from models_xgb.py)
# ---------------------------------------------------------------------------
XGB_PARAMS = {
    "learning_rate": 0.1,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
}

# ---------------------------------------------------------------------------
# MLP (stub until Himanshu's src/dl/model.py and Parth's src/dl/train.py land)
# ---------------------------------------------------------------------------


def _try_import_mlp():
    """Return (MLP class, train_fn) if Himanshu/Parth's code is merged, else None."""
    try:
        from src.dl.model import MLP  # noqa: PLC0415
        from src.dl.train import train as mlp_train  # noqa: PLC0415

        return MLP, mlp_train
    except ImportError:
        return None, None


class _StubMLP:
    """Minimal sklearn-compatible MLP used until src/dl/model.py is merged.

    Uses scikit-learn's MLPRegressor with sensible defaults so the hybrid
    pipeline runs end-to-end before the PyTorch MLP is available.
    The stub is replaced transparently once the import above succeeds.
    """

    def __init__(self, seed: int = SEED):
        from sklearn.neural_network import MLPRegressor  # noqa: PLC0415

        self._model = MLPRegressor(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            solver="adam",
            learning_rate_init=1e-3,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=seed,
            verbose=False,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_StubMLP":
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)


# ---------------------------------------------------------------------------
# Out-of-fold XGBoost predictions
# ---------------------------------------------------------------------------


def _oof_xgb_predictions(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_folds: int = 5,
    seed: int = SEED,
) -> tuple[np.ndarray, object]:
    """Compute out-of-fold XGBoost predictions on the training set.

    Returns
    -------
    oof_preds : ndarray of shape (n_train,)
        OOF predictions — each row was predicted by a model that never saw it.
    final_xgb : fitted XGBRegressor
        XGBoost trained on the full training set, used for val/test inference.
    """
    from sklearn.model_selection import KFold  # noqa: PLC0415

    oof_preds = np.zeros(len(y_train))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for fold_idx, (trn_idx, oof_idx) in enumerate(kf.split(X_train)):
        X_trn = X_train.iloc[trn_idx]
        y_trn = y_train.iloc[trn_idx]
        X_oof = X_train.iloc[oof_idx]
        X_v = X_val  # monitor overfitting during fold training

        # Match categories inside each fold so XGBoost sees consistent codes.
        X_trn_m, X_v_m, X_oof_m = match_categories(X_trn, X_v, X_oof)

        model = build_xgb(XGB_PARAMS)
        model.fit(X_trn_m, y_trn, eval_set=[(X_v_m, y_val)], verbose=False)
        oof_preds[oof_idx] = model.predict(X_oof_m)

    # Final XGBoost trained on all training data.
    X_tr_m, X_v_m, _ = match_categories(X_train, X_val, X_val)
    final_xgb = build_xgb(XGB_PARAMS)
    final_xgb.fit(X_tr_m, y_train, eval_set=[(X_v_m, y_val)], verbose=False)

    return oof_preds, final_xgb


# ---------------------------------------------------------------------------
# Numeric feature extraction (MLP needs float arrays)
# ---------------------------------------------------------------------------


def _to_numeric_array(df: pd.DataFrame) -> np.ndarray:
    """Convert a mixed-type DataFrame to a float32 NumPy array for the MLP.

    Categorical columns (Country, Waterbody Type) are label-encoded.
    Boolean flag columns are cast to 0/1.
    Missing values are filled with column medians (computed inside this call,
    so train/val/test must each be transformed separately after fitting on train).
    """
    out = df.copy()
    for col in out.columns:
        if hasattr(out[col], "cat"):
            out[col] = out[col].cat.codes.astype(float)
        elif out[col].dtype == bool:
            out[col] = out[col].astype(float)
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    # Fill NaN with 0 (data has already been cleaned; this is a safety net).
    out = out.fillna(0.0)
    return out.to_numpy(dtype=np.float32)


def _fit_numeric_scaler(X_train_arr: np.ndarray):
    """Fit a StandardScaler on the training array and return it."""
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

    scaler = StandardScaler()
    scaler.fit(X_train_arr)
    return scaler


# ---------------------------------------------------------------------------
# Core hybrid training
# ---------------------------------------------------------------------------


def train_hybrid(
    regime: int = 1,
    partial_scenario: str = "8/8",
    n_folds: int = 5,
    seed: int = SEED,
    subsample: int | None = None,
    save_reports: bool = True,
    output_path: Path | None = None,
) -> dict:
    """Train the residual XGBoost + MLP hybrid and return a results dict.

    Parameters
    ----------
    regime : {1, 2}
        Feature regime (1 = all chemical + metadata, 2 = metadata only).
    partial_scenario : str
        Which chemical features are available ('8/8', '7/8', '6/8', '4/8').
    n_folds : int
        Number of folds for OOF XGBoost predictions.
    seed : int
        Global random seed.
    subsample : int | None
        If set, subsample training rows for quick experiments.
    save_reports : bool
        Write results to reports/hybrid_results.csv.
    output_path : Path | None
        Override the default output path.

    Returns
    -------
    dict with keys: hybrid_val_metrics, hybrid_test_metrics, xgb_test_metrics,
                    mlp_residual_test_metrics, config, timing_seconds
    """
    # 1. Load shared splits (same as every other model in the project).
    include_meta = regime == 1
    X_train, X_val, X_test, y_train, y_val, y_test = get_splits(
        include_metadata=include_meta,
        partial_scenario=partial_scenario if regime == 1 else None,
    )

    if subsample is not None and subsample < len(X_train):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X_train), size=subsample, replace=False)
        X_train = X_train.iloc[idx]
        y_train = y_train.iloc[idx]

    t0 = time.perf_counter()

    # 2. OOF XGBoost predictions on train (no leakage).
    print(f"[hybrid] Running {n_folds}-fold OOF XGBoost …")
    oof_xgb_pred, final_xgb = _oof_xgb_predictions(
        X_train, y_train, X_val, y_val, n_folds=n_folds, seed=seed
    )

    # 3. XGBoost predictions on val and test.
    X_tr_m, X_v_m, X_te_m = match_categories(X_train, X_val, X_test)
    xgb_val_pred = final_xgb.predict(X_v_m)
    xgb_test_pred = final_xgb.predict(X_te_m)

    # 4. Residuals on train (this is what the MLP learns to predict).
    train_residuals = np.asarray(y_train) - oof_xgb_pred
    val_residuals = np.asarray(y_val) - xgb_val_pred

    # 5. Convert to numeric arrays for MLP.
    X_train_arr = _to_numeric_array(X_train)
    X_val_arr = _to_numeric_array(X_val)
    X_test_arr = _to_numeric_array(X_test)

    scaler = _fit_numeric_scaler(X_train_arr)
    X_train_scaled = scaler.transform(X_train_arr)
    X_val_scaled = scaler.transform(X_val_arr)
    X_test_scaled = scaler.transform(X_test_arr)

    # 6. Train MLP on residuals.
    MLP, mlp_train_fn = _try_import_mlp()
    if MLP is not None:
        # Himanshu/Parth's code is merged — use the real PyTorch MLP.
        print("[hybrid] Using src/dl/model.py MLP …")
        mlp = MLP(input_dim=X_train_scaled.shape[1], seed=seed)
        mlp_train_fn(mlp, X_train_scaled, train_residuals,
                     X_val_scaled, val_residuals)
        mlp_val_pred = mlp.predict(X_val_scaled)
        mlp_test_pred = mlp.predict(X_test_scaled)
    else:
        # Fallback stub (sklearn MLPRegressor).
        print("[hybrid] src/dl/model.py not yet merged — using sklearn stub MLP …")
        mlp = _StubMLP(seed=seed)
        mlp.fit(X_train_scaled, train_residuals)
        mlp_val_pred = mlp.predict(X_val_scaled)
        mlp_test_pred = mlp.predict(X_test_scaled)

    t1 = time.perf_counter()
    elapsed = round(t1 - t0, 2)

    # 7. Hybrid predictions = XGBoost + MLP residual correction.
    hybrid_val_pred = xgb_val_pred + mlp_val_pred
    hybrid_test_pred = xgb_test_pred + mlp_test_pred

    # 8. Evaluate using the shared evaluate() function.
    xgb_val_m = evaluate(y_val, xgb_val_pred)
    xgb_test_m = evaluate(y_test, xgb_test_pred)
    hybrid_val_m = evaluate(y_val, hybrid_val_pred)
    hybrid_test_m = evaluate(y_test, hybrid_test_pred)
    mlp_res_test_m = evaluate(np.asarray(y_test) - xgb_test_pred, mlp_test_pred)

    result = {
        "xgb_val_metrics": xgb_val_m,
        "xgb_test_metrics": xgb_test_m,
        "hybrid_val_metrics": hybrid_val_m,
        "hybrid_test_metrics": hybrid_test_m,
        "mlp_residual_test_metrics": mlp_res_test_m,
        "xgb_val_pred": xgb_val_pred,
        "xgb_test_pred": xgb_test_pred,
        "hybrid_val_pred": hybrid_val_pred,
        "hybrid_test_pred": hybrid_test_pred,
        "y_val": np.asarray(y_val),
        "y_test": np.asarray(y_test),
        "timing_seconds": elapsed,
        "config": {
            "regime": regime,
            "partial_scenario": partial_scenario,
            "n_folds": n_folds,
            "seed": seed,
            "n_train": len(X_train),
            "n_features": X_train.shape[1],
            "mlp_backend": "pytorch" if MLP is not None else "sklearn_stub",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }

    if save_reports:
        _save_results(result, output_path or HYBRID_RESULTS_PATH)

    return result


# ---------------------------------------------------------------------------
# Ablation table
# ---------------------------------------------------------------------------


def run_ablation(
    regime: int = 1,
    partial_scenario: str = "8/8",
    n_seeds: int = 5,
    subsample: int | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Run the four-way ablation required by the Phase 2 work plan.

    Conditions compared:
      1. XGBoost alone            (Phase 1 model, re-run here for fair seed parity)
      2. MLP alone                (requires src/dl/model.py; stub used otherwise)
      3. Plain average ensemble   (shows whether hybrid beats naive blending)
      4. Residual hybrid          (Jivit's approach)

    Reports mean ± std over n_seeds seeds.
    """
    seeds = [SEED + i for i in range(n_seeds)]

    rows = []
    for seed in seeds:
        # --- condition 1: XGBoost alone ---
        X_train, X_val, X_test, y_train, y_val, y_test = get_splits(
            include_metadata=(regime == 1),
            partial_scenario=partial_scenario if regime == 1 else None,
        )
        if subsample:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(X_train), size=subsample, replace=False)
            X_train_s = X_train.iloc[idx]; y_train_s = y_train.iloc[idx]
        else:
            X_train_s, y_train_s = X_train, y_train

        X_tr_m, X_v_m, X_te_m = match_categories(X_train_s, X_val, X_test)
        xgb = build_xgb(XGB_PARAMS)
        xgb.set_params(random_state=seed)
        xgb.fit(X_tr_m, y_train_s, eval_set=[(X_v_m, y_val)], verbose=False)
        xgb_pred = xgb.predict(X_te_m)
        xgb_m = evaluate(y_test, xgb_pred)

        # --- condition 2: MLP alone ---
        X_train_arr = _to_numeric_array(X_train_s)
        X_test_arr = _to_numeric_array(X_test)
        scaler = _fit_numeric_scaler(X_train_arr)
        X_tr_sc = scaler.transform(X_train_arr)
        X_te_sc = scaler.transform(X_test_arr)

        MLP_cls, mlp_train_fn = _try_import_mlp()
        if MLP_cls is not None:
            mlp_alone = MLP_cls(input_dim=X_tr_sc.shape[1], seed=seed)
            mlp_train_fn(mlp_alone, X_tr_sc, np.asarray(y_train_s),
                         scaler.transform(_to_numeric_array(X_val)),
                         np.asarray(y_val))
            mlp_pred = mlp_alone.predict(X_te_sc)
        else:
            mlp_alone = _StubMLP(seed=seed)
            mlp_alone.fit(X_tr_sc, np.asarray(y_train_s))
            mlp_pred = mlp_alone.predict(X_te_sc)
        mlp_m = evaluate(y_test, mlp_pred)

        # --- condition 3: plain average ---
        avg_pred = (xgb_pred + mlp_pred) / 2.0
        avg_m = evaluate(y_test, avg_pred)

        # --- condition 4: residual hybrid ---
        h_result = train_hybrid(
            regime=regime,
            partial_scenario=partial_scenario,
            n_folds=5,
            seed=seed,
            subsample=subsample,
            save_reports=False,
        )
        hyb_m = h_result["hybrid_test_metrics"]

        for name, m in [
            ("xgboost_alone", xgb_m),
            ("mlp_alone", mlp_m),
            ("plain_average", avg_m),
            ("residual_hybrid", hyb_m),
        ]:
            rows.append({
                "condition": name,
                "seed": seed,
                "regime": regime,
                "partial_scenario": partial_scenario,
                **{k: round(v, 6) for k, v in m.items()},
            })

    df = pd.DataFrame(rows)
    agg = (
        df.groupby("condition")[["mae", "rmse", "r2"]]
        .agg(["mean", "std"])
        .round(4)
    )
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg = agg.reset_index()

    path = output_path or (REPORTS / "hybrid_ablation.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path.with_name("hybrid_ablation_raw.csv"), index=False)
    agg.to_csv(path, index=False)
    print(f"\nAblation table saved to {path}")
    print(agg.to_string(index=False))
    return agg


# ---------------------------------------------------------------------------
# CSV result logger
# ---------------------------------------------------------------------------


def _save_results(result: dict, path: Path) -> None:
    """Append one row to the hybrid results CSV (append-only, like Phase 1)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = result["config"]
    row = {
        "timestamp": cfg["timestamp"],
        "regime": cfg["regime"],
        "partial_scenario": cfg["partial_scenario"],
        "n_folds": cfg["n_folds"],
        "seed": cfg["seed"],
        "n_train": cfg["n_train"],
        "n_features": cfg["n_features"],
        "mlp_backend": cfg["mlp_backend"],
        "timing_s": result["timing_seconds"],
        # XGBoost-alone metrics on test
        **{f"xgb_test_{k}": v for k, v in result["xgb_test_metrics"].items()},
        # Hybrid metrics on test
        **{f"hybrid_test_{k}": v for k, v in result["hybrid_test_metrics"].items()},
        # Val metrics for sanity
        **{f"hybrid_val_{k}": v for k, v in result["hybrid_val_metrics"].items()},
    }
    write_header = not path.exists()
    with open(path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"[hybrid] Results saved → {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args():
    p = argparse.ArgumentParser(description="Jivit's XGBoost + MLP residual hybrid.")
    p.add_argument("--regime", type=int, default=1, choices=[1, 2])
    p.add_argument("--partial-scenario", default="8/8")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seeds", type=int, default=1,
                   help="Number of seeds for multi-seed runs (ablation uses this)")
    p.add_argument("--subsample", type=int, default=None)
    p.add_argument("--ablation", action="store_true",
                   help="Run the full 4-condition ablation table")
    p.add_argument("--no-report", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.ablation:
        table = run_ablation(
            regime=args.regime,
            partial_scenario=args.partial_scenario,
            n_seeds=args.seeds,
            subsample=args.subsample,
        )
    else:
        res = train_hybrid(
            regime=args.regime,
            partial_scenario=args.partial_scenario,
            n_folds=args.folds,
            seed=SEED,
            subsample=args.subsample,
            save_reports=not args.no_report,
        )
        xm = res["xgb_test_metrics"]
        hm = res["hybrid_test_metrics"]
        print(f"\nXGBoost alone — MAE={xm['mae']:.4f}  RMSE={xm['rmse']:.4f}  R²={xm['r2']:.4f}")
        print(f"Residual hybrid — MAE={hm['mae']:.4f}  RMSE={hm['rmse']:.4f}  R²={hm['r2']:.4f}")
        delta_rmse = xm["rmse"] - hm["rmse"]
        print(f"RMSE improvement from hybrid: {delta_rmse:+.4f}")
        print(f"Training time: {res['timing_seconds']:.1f}s")
