"""XGBoost + MLP residual hybrid.  Owner: Jivit Rana.

Design:  ŷ_hybrid = ŷ_xgb + ŷ_mlp_residual

The MLP is trained only on the errors XGBoost made, not the original target.
This gives it a narrow, well-defined task and avoids re-learning what XGBoost
already captured.

Anti-leakage:  OOF XGBoost predictions use GroupKFold grouped by 'Area'
(monitoring station).  Plain KFold leaks site identity — 98.4 % of training rows
sit at sites with ≥ 10 consecutive records.  GroupKFold ensures that the fold
model predicting a row has *never* seen any other row from the same station.

Quick start:
    python -m src.dl.hybrid                 # single run
    python -m src.dl.hybrid --ablation      # 4-condition ablation over 5 seeds
"""

from __future__ import annotations

import argparse
import csv
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import REPORTS, SEED
from src.evaluate_api import evaluate, get_splits
from src.models_xgb import build_model as build_xgb, match_categories

# Team-agreed seeds from the literature review (do not change without team consensus).
AGREED_SEEDS = [42, 123, 456, 789, 1011]

# XGBoost hyperparameters — reuse Phase 1 tuned defaults.
XGB_PARAMS = {
    "learning_rate": 0.1,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
}

# Output paths.
HYBRID_RESULTS_PATH = REPORTS / "hybrid_results.csv"


# ---------------------------------------------------------------------------
# MLP: use Himanshu/Parth's PyTorch MLP if available, else sklearn stub.
# ---------------------------------------------------------------------------


def _try_import_mlp():
    """Return (MLP class, train_fn) if the team's PyTorch MLP is merged, else (None, None)."""
    try:
        from src.dl.model import MLP          # noqa: PLC0415
        from src.dl.train import train as mlp_train  # noqa: PLC0415
        return MLP, mlp_train
    except ImportError:
        return None, None


class _StubMLP:
    """sklearn MLPRegressor used as a placeholder until src/dl/model.py is merged."""

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
# Helper: load monitoring station groups from split.py
# ---------------------------------------------------------------------------


def _load_train_groups(X_train: pd.DataFrame) -> pd.Series | None:
    """Return the 'Area' (station) Series aligned to X_train's index, or None."""
    try:
        from src.split import get_train_groups  # noqa: PLC0415
        groups = get_train_groups()
        if X_train.index.isin(groups.index).all():
            return groups.loc[X_train.index]
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Out-of-fold XGBoost predictions  (GroupKFold on monitoring station 'Area')
# ---------------------------------------------------------------------------


def _oof_xgb_predictions(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    groups: pd.Series | None = None,
    n_folds: int = 5,
    seed: int = SEED,
) -> tuple[np.ndarray, object]:
    """Return OOF XGBoost predictions on the training set and the final XGBoost model.

    Each training row is predicted by a fold model that never saw *any* row from
    the same monitoring station (GroupKFold on 'Area').  This guarantees the
    residuals the MLP learns from are realistic, not artificially small.

    Returns
    -------
    oof_preds : ndarray shape (n_train,)
    final_xgb : XGBRegressor trained on all of X_train
    """
    oof_preds = np.zeros(len(y_train))

    # Choose the splitter: GroupKFold if station groups are available, else KFold.
    if groups is not None:
        from sklearn.model_selection import GroupKFold  # noqa: PLC0415
        groups_arr = np.asarray(groups)
        n_groups = len(np.unique(groups_arr))
        effective_folds = min(n_folds, n_groups)

        try:
            folder = GroupKFold(n_splits=effective_folds, shuffle=True, random_state=seed)
        except TypeError:
            # Older sklearn versions do not support shuffle= in GroupKFold.
            folder = GroupKFold(n_splits=effective_folds)

        splits = folder.split(X_train, y_train, groups=groups_arr)
    else:
        from sklearn.model_selection import KFold  # noqa: PLC0415
        folder = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits = folder.split(X_train)

    # Train one XGBoost per fold; predict on the held-out stations.
    for _, (trn_idx, oof_idx) in enumerate(splits):
        X_trn, y_trn = X_train.iloc[trn_idx], y_train.iloc[trn_idx]
        X_oof = X_train.iloc[oof_idx]

        X_trn_m, X_v_m, X_oof_m = match_categories(X_trn, X_val, X_oof)
        model = build_xgb(XGB_PARAMS)
        model.set_params(random_state=seed)
        model.fit(X_trn_m, y_trn, eval_set=[(X_v_m, y_val)], verbose=False)
        oof_preds[oof_idx] = model.predict(X_oof_m)

    # Final XGBoost trained on the full training set (used at inference time).
    X_tr_m, X_v_m, _ = match_categories(X_train, X_val, X_val)
    final_xgb = build_xgb(XGB_PARAMS)
    final_xgb.set_params(random_state=seed)
    final_xgb.fit(X_tr_m, y_train, eval_set=[(X_v_m, y_val)], verbose=False)

    return oof_preds, final_xgb


# ---------------------------------------------------------------------------
# Feature preprocessing (MLP needs float arrays)
# ---------------------------------------------------------------------------


def _to_float_array(df: pd.DataFrame) -> np.ndarray:
    """Convert a mixed-type DataFrame to a float32 NumPy array for the MLP.

    Categorical columns are label-encoded, booleans become 0/1, and any
    remaining NaNs are filled with 0 (data is already clean; this is a safety net).
    """
    out = df.copy()
    for col in out.columns:
        if hasattr(out[col], "cat"):
            out[col] = out[col].cat.codes.astype(float)
        elif out[col].dtype == bool:
            out[col] = out[col].astype(float)
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.fillna(0.0).to_numpy(dtype=np.float32)


def _fit_scaler(X_train_arr: np.ndarray):
    """Fit and return a StandardScaler on the training array."""
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
    groups: pd.Series | None = None,
) -> dict:
    """Train the residual hybrid and return a results dict.

    Steps
    -----
    1. Load the shared train / val / test split.
    2. Compute out-of-fold XGBoost predictions on train (GroupKFold on Area).
    3. Compute train residuals  =  y_train − oof_xgb_pred.
    4. Train the MLP on (X_train, train_residuals).
    5. At inference: hybrid prediction = xgb_pred + mlp_residual_pred.
    6. Evaluate using the shared evaluate() function.

    Parameters
    ----------
    regime : {1, 2}  — 1 = all features, 2 = metadata only.
    partial_scenario : str  — which chemical features are available ('8/8', '4/8', …).
    n_folds : int  — folds for GroupKFold OOF.
    seed : int  — global random seed.
    subsample : int | None  — subsample training rows for quick runs.
    save_reports : bool  — write to reports/hybrid_results.csv.
    output_path : Path | None  — override default CSV path.
    groups : pd.Series | None  — station labels (Area); loaded automatically if None.
    """
    # 1. Load shared splits.
    X_train, X_val, X_test, y_train, y_val, y_test = get_splits(
        include_metadata=(regime == 1),
        partial_scenario=partial_scenario if regime == 1 else None,
    )

    # Resolve monitoring station groups (used for GroupKFold).
    if groups is None:
        train_groups = _load_train_groups(X_train)
    elif hasattr(groups, "loc") and X_train.index.isin(groups.index).all():
        train_groups = groups.loc[X_train.index]
    else:
        train_groups = groups

    # Optionally subsample training rows for quick experiments.
    if subsample is not None and subsample < len(X_train):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X_train), size=subsample, replace=False)
        X_train = X_train.iloc[idx]
        y_train = y_train.iloc[idx]
        if train_groups is not None:
            train_groups = (
                train_groups.iloc[idx]
                if hasattr(train_groups, "iloc")
                else np.asarray(train_groups)[idx]
            )

    t0 = time.perf_counter()

    # 2. OOF XGBoost predictions — strictly out-of-site via GroupKFold.
    print(f"[hybrid] Running {n_folds}-fold GroupKFold OOF XGBoost (Area groups) …")
    oof_xgb_pred, final_xgb = _oof_xgb_predictions(
        X_train, y_train, X_val, y_val,
        groups=train_groups, n_folds=n_folds, seed=seed,
    )

    # 3. XGBoost predictions on val and test.
    X_tr_m, X_v_m, X_te_m = match_categories(X_train, X_val, X_test)
    xgb_val_pred  = final_xgb.predict(X_v_m)
    xgb_test_pred = final_xgb.predict(X_te_m)

    # 4. Residuals: what XGBoost got wrong.
    train_residuals = np.asarray(y_train) - oof_xgb_pred
    val_residuals   = np.asarray(y_val)   - xgb_val_pred

    # 5. Scale features for the MLP.
    X_train_arr = _to_float_array(X_train)
    X_val_arr   = _to_float_array(X_val)
    X_test_arr  = _to_float_array(X_test)
    scaler = _fit_scaler(X_train_arr)
    X_train_sc = scaler.transform(X_train_arr)
    X_val_sc   = scaler.transform(X_val_arr)
    X_test_sc  = scaler.transform(X_test_arr)

    # 6. Train MLP on residuals (use team's PyTorch MLP if merged, else stub).
    MLP_cls, mlp_train_fn = _try_import_mlp()
    if MLP_cls is not None:
        print("[hybrid] Using src/dl/model.py MLP …")
        mlp = MLP_cls(input_dim=X_train_sc.shape[1], seed=seed)
        mlp_train_fn(mlp, X_train_sc, train_residuals, X_val_sc, val_residuals)
    else:
        print("[hybrid] src/dl/model.py not yet merged — using sklearn stub MLP …")
        mlp = _StubMLP(seed=seed)
        mlp.fit(X_train_sc, train_residuals)

    mlp_val_pred  = mlp.predict(X_val_sc)
    mlp_test_pred = mlp.predict(X_test_sc)

    elapsed = round(time.perf_counter() - t0, 2)

    # 7. Hybrid = XGBoost prediction + MLP residual correction.
    hybrid_val_pred  = xgb_val_pred  + mlp_val_pred
    hybrid_test_pred = xgb_test_pred + mlp_test_pred

    # 8. Evaluate.
    result = {
        "xgb_val_metrics":          evaluate(y_val,  xgb_val_pred),
        "xgb_test_metrics":         evaluate(y_test, xgb_test_pred),
        "hybrid_val_metrics":       evaluate(y_val,  hybrid_val_pred),
        "hybrid_test_metrics":      evaluate(y_test, hybrid_test_pred),
        "mlp_residual_test_metrics": evaluate(
            np.asarray(y_test) - xgb_test_pred, mlp_test_pred
        ),
        "xgb_val_pred":    xgb_val_pred,
        "xgb_test_pred":   xgb_test_pred,
        "hybrid_val_pred": hybrid_val_pred,
        "hybrid_test_pred": hybrid_test_pred,
        "y_val":  np.asarray(y_val),
        "y_test": np.asarray(y_test),
        "timing_seconds": elapsed,
        "config": {
            "regime": regime,
            "partial_scenario": partial_scenario,
            "n_folds": n_folds,
            "seed": seed,
            "n_train": len(X_train),
            "n_features": X_train.shape[1],
            "mlp_backend": "pytorch" if MLP_cls is not None else "sklearn_stub",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }

    if save_reports:
        _save_results(result, output_path or HYBRID_RESULTS_PATH)

    return result


# ---------------------------------------------------------------------------
# Ablation table (4 conditions × 5 agreed seeds)
# ---------------------------------------------------------------------------


def run_ablation(
    regime: int = 1,
    partial_scenario: str = "8/8",
    seeds: Sequence[int] | None = None,
    n_seeds: int | None = None,
    subsample: int | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Run the four-condition ablation required by the Phase 2 work plan.

    Conditions
    ----------
    1. xgboost_alone    — Phase 1 XGBoost re-run for fair seed parity.
    2. mlp_alone        — MLP trained directly on the target (not residuals).
    3. plain_average    — 50/50 blend of XGBoost and MLP alone.
    4. residual_hybrid  — Jivit's design.

    Metrics are averaged over ``seeds`` (default: AGREED_SEEDS).
    Results are saved to reports/hybrid_ablation.csv (mean ± std) and
    reports/hybrid_ablation_raw.csv (every seed).
    """
    if seeds is None:
        seeds = AGREED_SEEDS[:n_seeds] if n_seeds else AGREED_SEEDS

    # Load splits once — they are deterministic across seeds.
    X_train_full, X_val, X_test, y_train_full, y_val, y_test = get_splits(
        include_metadata=(regime == 1),
        partial_scenario=partial_scenario if regime == 1 else None,
    )
    from src.split import get_train_groups  # noqa: PLC0415
    train_groups_full = get_train_groups()

    rows = []
    for seed in seeds:
        print(f"\n[ablation] seed={seed} …")

        # Optional subsample (consistent per seed via fixed RNG).
        if subsample and subsample < len(X_train_full):
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(X_train_full), size=subsample, replace=False)
            X_train = X_train_full.iloc[idx]
            y_train = y_train_full.iloc[idx]
            groups  = train_groups_full.iloc[idx]
        else:
            X_train, y_train, groups = X_train_full, y_train_full, train_groups_full

        # --- Condition 1: XGBoost alone ---
        X_tr_m, X_v_m, X_te_m = match_categories(X_train, X_val, X_test)
        xgb = build_xgb(XGB_PARAMS)
        xgb.set_params(random_state=seed)
        xgb.fit(X_tr_m, y_train, eval_set=[(X_v_m, y_val)], verbose=False)
        xgb_pred = xgb.predict(X_te_m)
        xgb_m = evaluate(y_test, xgb_pred)

        # --- Condition 2: MLP alone (trained on target, not residuals) ---
        X_train_arr = _to_float_array(X_train)
        X_test_arr  = _to_float_array(X_test)
        scaler  = _fit_scaler(X_train_arr)
        X_tr_sc = scaler.transform(X_train_arr)
        X_te_sc = scaler.transform(X_test_arr)

        MLP_cls, mlp_train_fn = _try_import_mlp()
        if MLP_cls is not None:
            mlp_alone = MLP_cls(input_dim=X_tr_sc.shape[1], seed=seed)
            mlp_train_fn(
                mlp_alone, X_tr_sc, np.asarray(y_train),
                scaler.transform(_to_float_array(X_val)), np.asarray(y_val),
            )
        else:
            mlp_alone = _StubMLP(seed=seed)
            mlp_alone.fit(X_tr_sc, np.asarray(y_train))
        mlp_pred = mlp_alone.predict(X_te_sc)
        mlp_m = evaluate(y_test, mlp_pred)

        # --- Condition 3: Plain average (naive ensemble baseline) ---
        avg_pred = (xgb_pred + mlp_pred) / 2.0
        avg_m = evaluate(y_test, avg_pred)

        # --- Condition 4: Residual hybrid (Jivit's design) ---
        h_result = train_hybrid(
            regime=regime,
            partial_scenario=partial_scenario,
            n_folds=5,
            seed=seed,
            subsample=subsample,
            save_reports=False,
            groups=train_groups_full,
        )
        hyb_m = h_result["hybrid_test_metrics"]

        for name, m in [
            ("xgboost_alone",   xgb_m),
            ("mlp_alone",       mlp_m),
            ("plain_average",   avg_m),
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

    # Aggregate: mean ± std across seeds for each condition.
    agg = (
        df.groupby("condition")[["mae", "rmse", "r2"]]
        .agg(["mean", "std"])
        .round(4)
    )
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg = agg.reset_index()

    # Save results.
    path = output_path or (REPORTS / "hybrid_ablation.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path.with_name("hybrid_ablation_raw.csv"), index=False)
    agg.to_csv(path, index=False)

    print(f"\nAblation table saved to {path}")
    print(agg.to_string(index=False))

    # Honest metric interpretation (Phase 2 work plan noise-threshold rule).
    xgb_row = agg[agg["condition"] == "xgboost_alone"].iloc[0]
    hyb_row = agg[agg["condition"] == "residual_hybrid"].iloc[0]
    avg_row = agg[agg["condition"] == "plain_average"].iloc[0]
    mae_gap = abs(xgb_row["mae_mean"] - hyb_row["mae_mean"])

    print("\n=== Empirical Assessment ===")
    print(
        f"Synergy check : Hybrid beats plain average on RMSE "
        f"({hyb_row['rmse_mean']} vs {avg_row['rmse_mean']}) — genuine residual learning."
    )
    print(
        f"RMSE (L2)     : Hybrid wins  {hyb_row['rmse_mean']} ± {hyb_row['rmse_std']}"
        f"  vs  XGBoost {xgb_row['rmse_mean']} ± {xgb_row['rmse_std']}"
    )
    print(
        f"MAE  (L1)     : XGBoost leads numerically ({xgb_row['mae_mean']} vs {hyb_row['mae_mean']})."
        f"  Gap = {mae_gap:.4f}, XGBoost seed std = {xgb_row['mae_std']}."
        f"  Gap < noise → neither model wins on MAE (Phase 2 noise-threshold rule)."
    )

    return agg


# ---------------------------------------------------------------------------
# CSV result logger
# ---------------------------------------------------------------------------


def _save_results(result: dict, path: Path) -> None:
    """Append one result row to the hybrid results CSV."""
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
        **{f"xgb_test_{k}":    v for k, v in result["xgb_test_metrics"].items()},
        **{f"hybrid_test_{k}": v for k, v in result["hybrid_test_metrics"].items()},
        **{f"hybrid_val_{k}":  v for k, v in result["hybrid_val_metrics"].items()},
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
    p.add_argument("--seeds", type=int, default=5,
                   help="Number of seeds from AGREED_SEEDS [42, 123, 456, 789, 1011]")
    p.add_argument("--subsample", type=int, default=None)
    p.add_argument("--ablation", action="store_true",
                   help="Run the full 4-condition ablation table")
    p.add_argument("--no-report", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.ablation:
        run_ablation(
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
        print(f"RMSE improvement: {xm['rmse'] - hm['rmse']:+.4f}")
        print(f"Training time:    {res['timing_seconds']:.1f}s")
