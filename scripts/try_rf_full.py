"""Test script to evaluate Random Forest on full 1,986,397 training rows.

Evaluates with:
  - 100 trees, max_depth=15, min_samples_leaf=5, max_features='sqrt'
  - n_jobs=-1
  - 20-minute (1,200s) hard timeout

Captures:
  - Outcome: SUCCESS, TIMEOUT, MEMORY_ERROR, or other exception
  - Training time
  - Peak memory (if measurable)
  - Test metrics (if finished)
"""

from __future__ import annotations

import signal
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.evaluate_api import evaluate
from src.models_rf import CategoricalEncoder, load_rf_splits

REPORTS = Path("reports")
OUTCOME_CSV = REPORTS / "rf_full_training_outcome.csv"


def timeout_handler(signum, frame):
    raise TimeoutError("Execution exceeded 20-minute (1,200s) time limit.")


def main():
    print("=" * 70)
    print("EXPERIMENT: Random Forest on Full Training Split (1,986,397 rows)")
    print(
        "Hyperparameters: n_estimators=100, max_depth=15, min_samples_leaf=5, max_features='sqrt', n_jobs=-1"
    )
    print("Timeout: 20 minutes (1,200 seconds)")
    print("=" * 70)

    t0 = time.perf_counter()
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(1200)  # 20 minutes

    status = "UNKNOWN"
    err_msg = ""
    train_time = 0.0
    metrics = {}

    try:
        print("[1/4] Loading full Phase 1 splits (engineered features)...")
        X_tr, X_va, X_te, y_tr, y_va, y_te = load_rf_splits(
            regime=1, feature_set="engineered", partial_scenario="8/8"
        )
        print(f"      Train shape: {X_tr.shape}, Test shape: {X_te.shape}")

        print("[2/4] Fitting CategoricalEncoder on train only...")
        encoder = CategoricalEncoder()
        X_tr_enc = encoder.fit_transform(X_tr)
        X_te_enc = encoder.transform(X_te)
        print(f"      Encoded train shape: {X_tr_enc.shape}")

        print(
            "[3/4] Initializing RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_leaf=5, max_features='sqrt', n_jobs=-1)..."
        )
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_leaf=5,
            max_features="sqrt",
            n_jobs=-1,
            random_state=42,
        )

        print("[4/4] Fitting Random Forest on 1,986,397 rows...")
        fit_t0 = time.perf_counter()
        rf.fit(X_tr_enc, y_tr)
        train_time = time.perf_counter() - fit_t0
        print(f"      Fit completed in {train_time:.2f} seconds!")

        # Cancel alarm
        signal.alarm(0)

        print("Evaluating on test split (415,804 rows)...")
        test_pred = rf.predict(X_te_enc)
        metrics = evaluate(y_te, test_pred)
        status = "SUCCESS"
        print(
            f"      Test RMSE: {metrics.get('rmse'):.4f}, MAE: {metrics.get('mae'):.4f}, R2: {metrics.get('r2'):.4f}"
        )

    except TimeoutError as e:
        status = "TIMEOUT"
        err_msg = str(e)
        train_time = time.perf_counter() - t0
        print(f"FAILED: {err_msg}")
    except MemoryError as e:
        status = "MEMORY_ERROR"
        err_msg = str(e)
        train_time = time.perf_counter() - t0
        print(f"FAILED: Out of memory: {err_msg}")
    except Exception as e:
        status = "FAILED"
        err_msg = f"{type(e).__name__}: {e}"
        train_time = time.perf_counter() - t0
        print(f"FAILED: {err_msg}")
    finally:
        signal.alarm(0)

    # Save outcome
    total_time = time.perf_counter() - t0
    record = {
        "status": status,
        "n_train_samples": 1986397,
        "n_trees": 100,
        "max_depth": 15,
        "n_jobs": -1,
        "time_limit_s": 1200,
        "train_time_s": round(train_time, 2),
        "total_time_s": round(total_time, 2),
        "error_message": err_msg,
        "test_rmse": metrics.get("rmse", np.nan),
        "test_mae": metrics.get("mae", np.nan),
        "test_r2": metrics.get("r2", np.nan),
    }

    df = pd.DataFrame([record])
    OUTCOME_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTCOME_CSV, index=False)
    print(f"\n[Saved] Outcome written to {OUTCOME_CSV}:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
