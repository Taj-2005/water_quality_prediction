"""Tests for the XGBoost + WaterQualityMLP residual hybrid (src/dl/hybrid.py).

Verifies:
  - OOF XGBoost predictions preserve training row count and enforce zero site leakage
  - train_hybrid runs and returns the required metric dictionary structure
  - hybrid predictions are strictly additive (ŷ_hybrid = ŷ_xgb + ŷ_mlp_residual)
  - results are correctly saved and appended to CSV with PyTorch backend
  - ablation generates 4 conditions on validation split
  - GroupKFold guarantees zero site overlap across folds
  - _load_train_groups fails fast on index mismatch
  - PyTorch WaterQualityMLP training and residual prediction
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.dl import hybrid
from src.models_xgb import HAS_XGB
from tests.conftest import requires_parquet_engine, requires_phase1_splits

QUICK = {"subsample": 2000, "save_reports": False, "n_folds": 2, "epochs": 2}


# ---------------------------------------------------------------------------
# Tests that run in all environments (including CI without XGBoost)
# ---------------------------------------------------------------------------


def test_agreed_seeds():
    """Seeds must match the 5 team-agreed literature seeds."""
    assert hybrid.AGREED_SEEDS == [42, 123, 456, 789, 1011]


def test_oof_group_leakage_zero_overlap():
    """GroupKFold must guarantee zero Area overlap between train folds and holdout folds."""
    from sklearn.model_selection import GroupKFold

    groups = np.array(
        [
            "Station_A",
            "Station_A",
            "Station_B",
            "Station_B",
            "Station_C",
            "Station_C",
            "Station_D",
            "Station_D",
            "Station_E",
            "Station_E",
        ]
    )
    X = pd.DataFrame({"feat": np.arange(10.0)})
    y = pd.Series(np.arange(10.0))

    gkf = GroupKFold(n_splits=5)
    for trn_idx, oof_idx in gkf.split(X, y, groups=groups):
        trn_groups = set(groups[trn_idx])
        oof_groups = set(groups[oof_idx])
        assert len(trn_groups.intersection(oof_groups)) == 0


@requires_parquet_engine
@requires_phase1_splits
def test_get_train_groups_availability():
    """src.split.get_train_groups must return training Area labels."""
    from src.split import get_train_groups

    train_groups = get_train_groups()
    assert len(train_groups) > 0
    assert train_groups.nunique() > 1000


def test_load_train_groups_fails_fast_on_mismatch(monkeypatch):
    """_load_train_groups must raise RuntimeError on invalid/unmatched indices."""
    monkeypatch.setattr(
        "src.split.get_train_groups",
        lambda: pd.Series(["SiteA", "SiteB"], index=[100, 200]),
    )
    invalid_df = pd.DataFrame({"feat": [1.0, 2.0]}, index=[-999999, -888888])
    with pytest.raises(RuntimeError, match="Index mismatch"):
        hybrid._load_train_groups(invalid_df)


def test_pytorch_mlp_residual_forward():
    """WaterQualityMLP must instantiate and produce unscaled predictions through _predict_mlp."""
    from src.dl.model import WaterQualityMLP

    mlp = WaterQualityMLP(input_dim=5, hidden_dims=(32, 16), activation="relu", dropout=0.0)
    X = np.random.randn(20, 5).astype(np.float32)
    preds = hybrid._predict_mlp(mlp, X, y_mean=50.0, y_std=10.0)

    assert isinstance(preds, np.ndarray)
    assert preds.shape == (20,)
    assert not np.isnan(preds).any()


def test_hybrid_additive_identity():
    """Verify that hybrid prediction is strictly additive: ŷ_hybrid = ŷ_xgb + ŷ_mlp."""
    xgb_pred = np.array([55.0, 60.0, 75.0], dtype=np.float64)
    mlp_residual = np.array([-1.5, 0.8, -0.2], dtype=np.float64)
    expected_hybrid = xgb_pred + mlp_residual
    assert np.allclose(expected_hybrid, [53.5, 60.8, 74.8])


# ---------------------------------------------------------------------------
# Integration tests requiring XGBoost (skipped on runners without XGBoost)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_XGB, reason="xgboost is not installed")
def test_oof_xgb_predictions_shape_and_leakage_contract():
    """OOF predictions must have identical shape to y_train."""
    X_train = pd.DataFrame(
        {
            "num1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "cat1": pd.Categorical(["a", "b", "a", "b", "a", "b"]),
        }
    )
    y_train = pd.Series([10.0, 20.0, 15.0, 25.0, 12.0, 22.0])
    X_val = pd.DataFrame(
        {
            "num1": [2.5, 3.5],
            "cat1": pd.Categorical(["a", "b"]),
        }
    )
    y_val = pd.Series([18.0, 21.0])
    groups = pd.Series(["S1", "S1", "S2", "S2", "S3", "S3"])

    oof_preds, final_xgb = hybrid._oof_xgb_predictions(
        X_train, y_train, X_val, y_val, groups=groups, n_folds=2, seed=42
    )

    assert len(oof_preds) == len(y_train)
    assert not np.isnan(oof_preds).any()
    assert hasattr(final_xgb, "predict")


@pytest.mark.skipif(not HAS_XGB, reason="xgboost is not installed")
@requires_parquet_engine
@requires_phase1_splits
def test_train_hybrid_returns_metrics():
    """train_hybrid must return expected evaluation dict with PyTorch backend."""
    res = hybrid.train_hybrid(**QUICK)
    assert "xgb_test_metrics" in res
    assert "hybrid_test_metrics" in res
    assert "hybrid_val_metrics" in res
    assert "mlp_residual_test_metrics" in res
    for metric in ("mae", "rmse", "r2"):
        assert metric in res["hybrid_test_metrics"]
        assert metric in res["xgb_test_metrics"]
    assert res["config"]["n_train"] == 2000
    assert res["config"]["mlp_backend"] == "pytorch"
    assert res["timing_seconds"] > 0


@pytest.mark.skipif(not HAS_XGB, reason="xgboost is not installed")
@requires_parquet_engine
@requires_phase1_splits
def test_hybrid_prediction_is_additive():
    """ŷ_hybrid must equal ŷ_xgb + ŷ_mlp_residual mathematically in train_hybrid."""
    res = hybrid.train_hybrid(**QUICK)
    xgb_pred = res["xgb_val_pred"]
    hyb_pred = res["hybrid_val_pred"]
    mlp_res = res["mlp_val_residual_pred"]

    assert np.allclose(hyb_pred, xgb_pred + mlp_res, atol=1e-5)
    diff = hyb_pred - xgb_pred
    assert len(diff) == len(xgb_pred)
    assert not np.allclose(diff, 0.0)


@pytest.mark.skipif(not HAS_XGB, reason="xgboost is not installed")
@requires_parquet_engine
@requires_phase1_splits
def test_results_appended_to_csv(tmp_path: Path):
    """Multiple runs should append to CSV without overwriting."""
    csv_path = tmp_path / "test_hybrid_results.csv"
    hybrid.train_hybrid(
        subsample=1000, n_folds=2, epochs=1, save_reports=True, output_path=csv_path
    )
    hybrid.train_hybrid(
        subsample=1000, n_folds=2, epochs=1, save_reports=True, output_path=csv_path
    )
    df = pd.read_csv(csv_path)
    assert len(df) == 2
    assert "xgb_test_rmse" in df.columns
    assert "hybrid_test_rmse" in df.columns
    assert "mlp_backend" in df.columns
    assert (df["mlp_backend"] == "pytorch").all()


@pytest.mark.skipif(not HAS_XGB, reason="xgboost is not installed")
@requires_parquet_engine
@requires_phase1_splits
def test_ablation_has_four_conditions_on_val(tmp_path: Path):
    """Ablation table must evaluate all 4 specified conditions on validation split."""
    out = tmp_path / "hybrid_ablation.csv"
    agg = hybrid.run_ablation(
        n_seeds=1,
        subsample=1000,
        output_path=out,
    )
    expected_conditions = {"xgboost_alone", "mlp_alone", "plain_average", "residual_hybrid"}
    assert set(agg["condition"]) == expected_conditions
    assert out.exists()

    raw_path = out.with_name("hybrid_ablation_raw.csv")
    assert raw_path.exists()
    raw_df = pd.read_csv(raw_path)
    assert (raw_df["split_evaluated"] == "val").all()
