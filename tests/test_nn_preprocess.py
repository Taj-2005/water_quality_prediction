"""Tests for neural-network data preparation.

These run on small synthetic frames. The guarantee that matters most -- every
transform fitted on train alone -- is checked directly rather than assumed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn", reason="scikit-learn is required for preprocessing tests")

from src.clean import MEASUREMENTS  # noqa: E402
from src.dl.nn_preprocess import (  # noqa: E402
    CATEGORICAL_FEATURES,
    build_nn_preprocessor,
    conditioning_table,
    skew_report,
    transform_splits,
)


def make_frame(n: int = 400, seed: int = 0, skew: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {c: (rng.lognormal(0, 1.5, n) if skew else rng.normal(5, 1, n)) for c in MEASUREMENTS}
    data["Country"] = rng.choice(["England", "USA", "Ireland"], n)
    data["Waterbody Type"] = rng.choice(["River", "Lake"], n)
    return pd.DataFrame(data)


class TestTrainOnlyFitting:
    """The central guarantee: validation and test never inform the transform."""

    def test_validation_does_not_change_the_fit(self):
        tr = make_frame(300, seed=1)
        va_small = make_frame(50, seed=2)
        va_huge = make_frame(50, seed=2)
        va_huge[MEASUREMENTS[0]] *= 1000.0  # a wildly different validation set

        a = build_nn_preprocessor(tr, strategy="phase1")
        b = build_nn_preprocessor(tr, strategy="phase1")
        np.testing.assert_allclose(a.transform(va_small), b.transform(va_small))

        # Transforming a distorted validation set must not alter the fitted stats.
        before = a.named_transformers_["num"].named_steps["scale"].mean_.copy()
        a.transform(va_huge)
        np.testing.assert_allclose(before, a.named_transformers_["num"].named_steps["scale"].mean_)

    def test_transform_splits_returns_three_arrays_and_the_fitted_object(self):
        tr, va, te = make_frame(200, 1), make_frame(60, 2), make_frame(60, 3)
        a, b, c, pre = transform_splits(tr, va, te, strategy="phase1")
        assert a.shape[1] == b.shape[1] == c.shape[1]
        assert a.dtype == np.float32
        assert hasattr(pre, "transform")


class TestStrategies:
    def test_every_strategy_runs_and_preserves_row_count(self):
        tr, va, te = make_frame(300, 1), make_frame(80, 2), make_frame(80, 3)
        for s in ("phase1", "robust", "yeo-johnson", "quantile"):
            a, b, c, _ = transform_splits(tr, va, te, strategy=s)
            assert len(a) == 300 and len(b) == 80 and len(c) == 80

    def test_unknown_strategy_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            build_nn_preprocessor(make_frame(50), strategy="log1p")

    def test_power_transform_reduces_skew(self):
        rep = skew_report(make_frame(600, seed=5), "yeo-johnson")
        before = np.mean([abs(v) for v in rep.skew_before.values()])
        after = np.mean([abs(v) for v in rep.skew_after.values()])
        assert after < before


class TestFlagHandling:
    """Binary indicators pass through unscaled; the count is scaled."""

    def test_binary_flags_keep_their_zero(self):
        tr = make_frame(300, seed=1)
        flag = f"{MEASUREMENTS[0]}__imputed_upstream"
        tr[flag] = np.r_[np.ones(100), np.zeros(200)]

        pre = build_nn_preprocessor(tr, strategy="phase1")
        out = pre.transform(tr)
        names = list(pre.get_feature_names_out())
        col = out[:, [i for i, n in enumerate(names) if n.endswith(flag)][0]]
        assert set(np.unique(col)) == {0.0, 1.0}, "a standardised flag would not be 0/1"

    def test_count_column_is_scaled(self):
        tr = make_frame(300, seed=1)
        tr["n_imputed_upstream"] = np.random.default_rng(0).integers(0, 9, 300)
        pre = build_nn_preprocessor(tr, strategy="phase1")
        names = list(pre.get_feature_names_out())
        col = pre.transform(tr)[:, [i for i, n in enumerate(names) if "n_imputed" in n][0]]
        assert abs(col.mean()) < 1e-6 and abs(col.std() - 1) < 0.1


class TestCategoricalEncoding:
    def test_onehot_is_wider_than_ordinal(self):
        tr = make_frame(300, seed=1)
        wide = build_nn_preprocessor(tr, categorical="onehot").transform(tr).shape[1]
        narrow = build_nn_preprocessor(tr, categorical="ordinal").transform(tr).shape[1]
        assert wide > narrow

    def test_unseen_category_does_not_raise(self):
        tr = make_frame(300, seed=1)
        va = make_frame(50, seed=2)
        va["Country"] = "Atlantis"  # never present in training
        pre = build_nn_preprocessor(tr, categorical="onehot")
        assert pre.transform(va).shape[0] == 50

    def test_area_is_never_a_feature(self):
        """Area is the split grouping key; using it would be leakage by another name."""
        assert "Area" not in CATEGORICAL_FEATURES


def test_conditioning_table_ranks_strategies():
    tr, va = make_frame(600, 1), make_frame(200, 2)
    t = conditioning_table(tr, va, strategies=["phase1", "quantile"])
    assert set(t.strategy) == {"phase1", "quantile"}
    q = t[t.strategy == "quantile"].max_abs_value.iloc[0]
    p = t[t.strategy == "phase1"].max_abs_value.iloc[0]
    assert q <= p, "quantile-normal should bound the inputs at least as tightly"
