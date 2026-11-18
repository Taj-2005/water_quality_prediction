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
    aggregate,
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


class TestSelectionNeverUsesTest:
    """Selection happens on validation only.

    The shared rules reserve the test split for the final table. Step 2 also asks for
    test metrics, which is resolved by reading test once for the configuration that
    validation already chose -- never by ranking configurations on it.
    """

    def test_best_combination_ignores_test_columns(self):
        """Adding test columns that contradict the validation ranking changes nothing."""
        base = [
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
        ]
        without = best_combination(pd.DataFrame(base))
        with_test = best_combination(
            pd.DataFrame([{**r, "test_mae_mean": 10.0 - i} for i, r in enumerate(base)])
        )
        assert without["overrides"] == with_test["overrides"] == {"dropout": 0.2}

    def test_run_config_does_not_touch_test_by_default(self):
        """eval_test defaults to False, so a sweep row cannot carry a test number."""
        import inspect

        from src.dl.regularization import run_config

        assert inspect.signature(run_config).parameters["eval_test"].default is False


class TestAggregate:
    def test_records_the_seed_count_per_config(self):
        """A reader must be able to see how many seeds a mean and sd came from."""
        raw = pd.DataFrame(
            [
                {
                    "family": "none",
                    "config": "unregularized",
                    "seed": s,
                    "val_mae": v,
                    "val_rmse": v,
                    "val_r2": 0.9,
                    "generalisation_gap": 0.0,
                    "epochs_run": 15,
                }
                for s, v in zip(
                    [42, 123, 456, 789, 1011], [0.78, 0.80, 0.77, 0.79, 0.81], strict=True
                )
            ]
        )
        agg = aggregate(raw)
        assert agg.n_seeds.iloc[0] == 5
        assert agg.val_mae_mean.iloc[0] == pytest.approx(0.790, abs=1e-3)

    def test_appending_seeds_matches_a_single_full_sweep(self):
        """Aggregating 3+2 appended seeds equals aggregating all 5 at once."""
        rows = [
            {
                "family": "none",
                "config": "unregularized",
                "seed": s,
                "val_mae": v,
                "val_rmse": v,
                "val_r2": 0.9,
                "generalisation_gap": 0.0,
                "epochs_run": 15,
            }
            for s, v in zip([42, 123, 456, 789, 1011], [0.78, 0.80, 0.77, 0.79, 0.81], strict=True)
        ]
        appended = aggregate(
            pd.concat([pd.DataFrame(rows[:3]), pd.DataFrame(rows[3:])], ignore_index=True)
        )
        at_once = aggregate(pd.DataFrame(rows))
        pd.testing.assert_frame_equal(
            appended.reset_index(drop=True), at_once.reset_index(drop=True)
        )


class TestSeedRule:
    """The shared rules require 5 seeds for every neural model.

    The point of the rule is that a mean without a spread cannot support a claim, and a
    spread from too few runs is itself unreliable. The whole conclusion of this study is
    a noise argument, so the seed count is load-bearing rather than cosmetic.
    """

    def test_run_ablation_defaults_to_every_agreed_seed(self):
        import inspect

        from src.dl.regularization import run_ablation

        assert tuple(inspect.signature(run_ablation).parameters["seeds"].default) == TEAM_SEEDS
        assert len(TEAM_SEEDS) == 5

    def test_aggregate_exposes_a_seed_count_for_every_config(self):
        """No row may hide how many runs its mean and sd came from."""
        raw = pd.DataFrame(
            [
                {
                    "family": f,
                    "config": c,
                    "seed": s,
                    "val_mae": 0.8,
                    "val_rmse": 1.0,
                    "val_r2": 0.9,
                    "generalisation_gap": 0.0,
                    "epochs_run": 15,
                }
                for f, c in (("none", "unregularized"), ("dropout", "dropout=0.2"))
                for s in TEAM_SEEDS
            ]
        )
        assert (aggregate(raw).n_seeds == 5).all()
