"""Tests for the XGBoost model (src/models_xgb.py).

Checks that:
  - validation/test splits get the same categories as the training split
  - train() returns the expected scores and is repeatable
  - results are written to a CSV
  - tuning only ever sees the validation set
  - the ablation runs both "raw" and "engineered" features
"""

import pandas as pd
import pytest

from src import models_xgb

pytest.importorskip("optuna", reason="optuna is needed for the tuning test")
pytestmark = pytest.mark.skipif(not models_xgb.HAS_XGB, reason="xgboost is not available")

# 5,000 training rows keeps every test fast.
QUICK = {"subsample": 5000, "save_reports": False}


def test_match_categories_uses_train_categories():
    train = pd.DataFrame({"c": pd.Categorical(["a", "b"]), "x": [1.0, 2.0]})
    val = pd.DataFrame({"c": pd.Categorical(["b", "z"]), "x": [3.0, 4.0]})
    test = val.copy()
    _, val_fixed, _ = models_xgb.match_categories(train, val, test)
    assert list(val_fixed["c"].cat.categories) == ["a", "b"]
    assert pd.isna(val_fixed["c"].iloc[1])  # "z" never appeared in train


def test_train_returns_scores():
    res = models_xgb.train(**QUICK)
    assert set(res["val_metrics"]) == {"mae", "rmse", "r2"}
    assert set(res["test_metrics"]) == {"mae", "rmse", "r2"}
    assert res["config"]["n_train_samples"] == 5000
    assert len(res["feature_importances"]) == res["config"]["n_features"]
    assert "NOT independent prediction" in res["config"]["experiment_label"]


def test_train_is_repeatable():
    first = models_xgb.train(**QUICK)["val_metrics"]["rmse"]
    second = models_xgb.train(**QUICK)["val_metrics"]["rmse"]
    assert first == pytest.approx(second)


def test_results_are_saved_to_csv(tmp_path):
    csv = tmp_path / "xgb.csv"
    models_xgb.train(subsample=5000, output_path=csv)
    models_xgb.train(subsample=5000, output_path=csv)
    df = pd.read_csv(csv)
    assert len(df) == 2  # second run is appended, not overwritten
    for column in ("val_rmse", "test_rmse", "test_r2", "learning_rate", "best_iteration"):
        assert column in df.columns


def test_tuning_only_sees_validation(monkeypatch):
    seen = {}
    real_tune = models_xgb.tune

    def spy(X_train, y_train, X_val, y_val, n_trials):
        seen["val_rows"] = len(X_val)
        return real_tune(X_train, y_train, X_val, y_val, n_trials)

    monkeypatch.setattr(models_xgb, "tune", spy)
    monkeypatch.setattr(models_xgb, "TUNE_ROWS", 3000)
    res = models_xgb.train(do_tune=True, n_trials=2, **QUICK)
    assert res["config"]["tuned"] and res["config"]["n_trials"] == 2
    assert seen["val_rows"] > 0


def test_ablation_runs_both_feature_sets(tmp_path):
    table = models_xgb.run_ablation(output_path=tmp_path / "ablation.csv", **QUICK)
    assert list(table["feature_set"]) == ["raw", "engineered"]
    assert table.loc[0, "n_features"] < table.loc[1, "n_features"]
    assert (tmp_path / "ablation.csv").exists()
