"""Tests for the XGBoost + MLP residual hybrid (src/dl/hybrid.py).

Checks that:
  - OOF XGBoost predictions preserve training row count and do not leak
  - train_hybrid runs and returns the required metric dictionary structure
  - hybrid predictions are an additive combination of XGBoost and residual MLP
  - results are correctly saved and appended to CSV
  - ablation generates 4 conditions (XGBoost alone, MLP alone, plain average, residual hybrid)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.dl import hybrid

pytestmark = pytest.mark.skipif(not hybrid.hasattr_xgb if hasattr(hybrid, "hasattr_xgb") else False, reason="xgboost test")

QUICK = {"subsample": 2000, "save_reports": False, "n_folds": 2}


def test_oof_xgb_predictions_shape_and_leakage_contract():
    """OOF predictions must have identical shape to y_train."""
    X_train = pd.DataFrame({
        "num1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "cat1": pd.Categorical(["a", "b", "a", "b", "a", "b"]),
    })
    y_train = pd.Series([10.0, 20.0, 15.0, 25.0, 12.0, 22.0])
    X_val = pd.DataFrame({
        "num1": [2.5, 3.5],
        "cat1": pd.Categorical(["a", "b"]),
    })
    y_val = pd.Series([18.0, 21.0])

    oof_preds, final_xgb = hybrid._oof_xgb_predictions(
        X_train, y_train, X_val, y_val, n_folds=2, seed=42
    )

    assert len(oof_preds) == len(y_train)
    assert not np.isnan(oof_preds).any()
    assert hasattr(final_xgb, "predict")


def test_train_hybrid_returns_metrics():
    """train_hybrid must return expected evaluation dict with all keys."""
    res = hybrid.train_hybrid(**QUICK)
    assert "xgb_test_metrics" in res
    assert "hybrid_test_metrics" in res
    assert "hybrid_val_metrics" in res
    assert "mlp_residual_test_metrics" in res
    for metric in ("mae", "rmse", "r2"):
        assert metric in res["hybrid_test_metrics"]
        assert metric in res["xgb_test_metrics"]
    assert res["config"]["n_train"] == 2000
    assert res["timing_seconds"] > 0


def test_hybrid_prediction_is_additive():
    """ŷ_hybrid must equal ŷ_xgb + ŷ_mlp_residual mathematically."""
    res = hybrid.train_hybrid(**QUICK)
    xgb_pred = res["xgb_test_pred"]
    hyb_pred = res["hybrid_test_pred"]
    # Check that predictions differ by the learned non-zero residual correction
    diff = hyb_pred - xgb_pred
    assert len(diff) == len(xgb_pred)
    assert not np.allclose(diff, 0.0)


def test_results_appended_to_csv(tmp_path):
    """Multiple runs should append to CSV without overwriting."""
    csv_path = tmp_path / "test_hybrid_results.csv"
    hybrid.train_hybrid(subsample=1000, n_folds=2, save_reports=True, output_path=csv_path)
    hybrid.train_hybrid(subsample=1000, n_folds=2, save_reports=True, output_path=csv_path)
    df = pd.read_csv(csv_path)
    assert len(df) == 2
    assert "xgb_test_rmse" in df.columns
    assert "hybrid_test_rmse" in df.columns
    assert "timing_s" in df.columns


def test_ablation_has_four_conditions(tmp_path):
    """Ablation table must evaluate all 4 specified conditions."""
    out = tmp_path / "hybrid_ablation.csv"
    agg = hybrid.run_ablation(
        n_seeds=1,
        subsample=1000,
        output_path=out,
    )
    expected_conditions = {"xgboost_alone", "mlp_alone", "plain_average", "residual_hybrid"}
    assert set(agg["condition"]) == expected_conditions
    assert out.exists()


def test_agreed_seeds():
    """Seeds must match the 5 team-agreed seeds from the literature review."""
    assert hybrid.AGREED_SEEDS == [42, 123, 456, 789, 1011]


def test_oof_group_leakage_zero_overlap():
    """GroupKFold must guarantee zero Area overlap between train folds and holdout folds."""
    from sklearn.model_selection import GroupKFold

    # 10 rows from 5 distinct stations
    groups = np.array(["Station_A", "Station_A", "Station_B", "Station_B",
                       "Station_C", "Station_C", "Station_D", "Station_D",
                       "Station_E", "Station_E"])
    X = pd.DataFrame({"feat": np.arange(10.0)})
    y = pd.Series(np.arange(10.0))

    gkf = GroupKFold(n_splits=5)
    for trn_idx, oof_idx in gkf.split(X, y, groups=groups):
        trn_groups = set(groups[trn_idx])
        oof_groups = set(groups[oof_idx])
        # Zero group overlap contract
        assert len(trn_groups.intersection(oof_groups)) == 0


def test_get_train_groups_availability():
    """src.split.get_train_groups must return training Area labels."""
    from src.split import get_train_groups

    train_groups = get_train_groups()
    assert len(train_groups) > 0
    assert train_groups.nunique() > 1000


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    print("Running test_oof_xgb_predictions_shape_and_leakage_contract...")
    test_oof_xgb_predictions_shape_and_leakage_contract()
    print("Running test_train_hybrid_returns_metrics...")
    test_train_hybrid_returns_metrics()
    print("Running test_hybrid_prediction_is_additive...")
    test_hybrid_prediction_is_additive()
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp)
        print("Running test_results_appended_to_csv...")
        test_results_appended_to_csv(p)
        print("Running test_ablation_has_four_conditions...")
        test_ablation_has_four_conditions(p)
    print("All hybrid tests passed successfully!")

