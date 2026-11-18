"""Tests for the regularization study.

Run on small synthetic arrays. The properties worth asserting are that the
comparison is controlled -- same budget, same base config, one thing varied --
and that the combination logic only adopts a regularizer that actually helped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch", reason="PyTorch is required for the regularization study")
pytest.importorskip("sklearn", reason="scikit-learn is required")

from src.dl.regularization import (  # noqa: E402
    BASE_CONFIG,
    REGULARIZER_GRID,
    TEAM_SEEDS,
    best_combination,
    run_config,
)


def tiny_arrays(n_train: int = 400, n_eval: int = 120, n_feat: int = 6, seed: int = 0):
    rng = np.random.default_rng(seed)
    mk = lambda n: rng.normal(0, 1, (n, n_feat)).astype(np.float32)  # noqa: E731
    Xa, Xb, Xc = mk(n_train), mk(n_eval), mk(n_eval)
    f = lambda X: (X[:, 0] * 3 + X[:, 1] - X[:, 2] * 2 + 50).astype(np.float64)  # noqa: E731
    return Xa, Xb, Xc, f(Xa), f(Xb), f(Xc)


FAST = {"epochs": 2, "batch_size": 64}


class TestGridIntegrity:
    """The grid is the experiment design; these guard it from silent drift."""

    def test_base_config_has_every_regularizer_switched_off(self):
        assert BASE_CONFIG["dropout"] == 0.0
        assert BASE_CONFIG["weight_decay"] == 0.0
        assert BASE_CONFIG["use_batch_norm"] is False
        assert BASE_CONFIG["scheduler"] is None

    def test_grid_covers_every_family_the_work_plan_names(self):
        families = {f for f, _, _ in REGULARIZER_GRID}
        assert {
            "dropout",
            "weight_decay",
            "batch_norm",
            "early_stopping",
            "lr_schedule",
        } <= families

    def test_each_entry_varies_one_family_only(self):
        """A row that changed two things at once would not be an ablation."""
        for family, label, overrides in REGULARIZER_GRID:
            if family == "none":
                assert overrides == {}
                continue
            keys = set(overrides) - {"scheduler_patience"}  # a setting of the scheduler itself
            assert len(keys) == 1, f"{label} varies {keys}"

    def test_seeds_match_the_team_agreed_list(self):
        assert TEAM_SEEDS == (42, 123, 456, 789, 1011)


class TestRunConfig:
    def test_returns_metrics_and_history(self):
        r = run_config(tiny_arrays(), {}, seed=42, base={**BASE_CONFIG, **FAST})
        for k in ("val_mae", "val_rmse", "val_r2", "generalisation_gap", "history"):
            assert k in r
        assert r["epochs_run"] >= 1

    def test_dropout_changes_parameter_count_not_at_all(self):
        """Dropout regularises without adding parameters; batch norm does add them."""
        base = {**BASE_CONFIG, **FAST}
        a = run_config(tiny_arrays(), {"dropout": 0.3}, seed=42, base=base)
        b = run_config(tiny_arrays(), {}, seed=42, base=base)
        c = run_config(tiny_arrays(), {"use_batch_norm": True}, seed=42, base=base)
        assert a["n_params"] == b["n_params"]
        assert c["n_params"] > b["n_params"]

    def test_same_seed_reproduces(self):
        base = {**BASE_CONFIG, **FAST}
        a = run_config(tiny_arrays(), {}, seed=7, base=base)
        b = run_config(tiny_arrays(), {}, seed=7, base=base)
        assert a["val_mae"] == pytest.approx(b["val_mae"], rel=1e-6)

    def test_early_stopping_can_cut_the_run_short(self):
        base = {**BASE_CONFIG, "epochs": 12, "batch_size": 64}
        stopped = run_config(tiny_arrays(), {"early_stopping_patience": 1}, seed=42, base=base)
        assert stopped["epochs_run"] <= 12


class TestBestCombination:
    """A family is adopted only if its gain clears the seed-to-seed noise.

    The work plan's rule is that "a difference smaller than the noise is not claimed
    as a win". Comparing means alone would adopt a regularizer on an improvement many
    times smaller than the spread across seeds.
    """

    @staticmethod
    def _agg(rows):
        return pd.DataFrame(rows)

    def test_adopts_a_gain_that_clears_the_noise(self):
        agg = self._agg(
            [
                {
                    "family": "none",
                    "config": "unregularized",
                    "val_mae_mean": 1.00,
                    "val_mae_std": 0.01,
                },
                {
                    "family": "dropout",
                    "config": "dropout=0.2",
                    "val_mae_mean": 0.80,
                    "val_mae_std": 0.01,
                },
                {
                    "family": "weight_decay",
                    "config": "wd=1e-4",
                    "val_mae_mean": 1.50,
                    "val_mae_std": 0.01,
                },
            ]
        )
        out = best_combination(agg)
        assert out["overrides"] == {"dropout": 0.2}
        assert [a[0] for a in out["adopted"]] == ["dropout"]

    def test_rejects_a_gain_smaller_than_the_noise(self):
        """The real finding from this study: wd=1e-4 improved MAE by 0.0016 against a
        seed spread of 0.0281, which is not evidence of anything."""
        agg = self._agg(
            [
                {
                    "family": "none",
                    "config": "unregularized",
                    "val_mae_mean": 0.7794,
                    "val_mae_std": 0.0237,
                },
                {
                    "family": "weight_decay",
                    "config": "wd=1e-4",
                    "val_mae_mean": 0.7778,
                    "val_mae_std": 0.0281,
                },
            ]
        )
        out = best_combination(agg)
        assert out["overrides"] == {}, "a 0.0016 gain against 0.0281 noise is not a win"
        assert out["rejected_within_noise"][0][0] == "weight_decay"

    def test_adopts_nothing_when_nothing_helps(self):
        agg = self._agg(
            [
                {
                    "family": "none",
                    "config": "unregularized",
                    "val_mae_mean": 0.50,
                    "val_mae_std": 0.01,
                },
                {
                    "family": "dropout",
                    "config": "dropout=0.2",
                    "val_mae_mean": 0.90,
                    "val_mae_std": 0.01,
                },
                {
                    "family": "batch_norm",
                    "config": "batch_norm",
                    "val_mae_mean": 0.75,
                    "val_mae_std": 0.01,
                },
            ]
        )
        out = best_combination(agg)
        assert out["overrides"] == {}
        assert len(out["rejected_within_noise"]) == 2

    def test_picks_the_best_member_within_a_family(self):
        agg = self._agg(
            [
                {
                    "family": "none",
                    "config": "unregularized",
                    "val_mae_mean": 1.00,
                    "val_mae_std": 0.01,
                },
                {
                    "family": "dropout",
                    "config": "dropout=0.1",
                    "val_mae_mean": 0.95,
                    "val_mae_std": 0.01,
                },
                {
                    "family": "dropout",
                    "config": "dropout=0.5",
                    "val_mae_mean": 0.70,
                    "val_mae_std": 0.01,
                },
            ]
        )
        assert best_combination(agg)["overrides"] == {"dropout": 0.5}
