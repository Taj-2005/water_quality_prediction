"""Utilities for the fair five-seed final Phase 2 comparison.

Owner: Parthraj Singh Bhati.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.dl.regularization import TEAM_SEEDS


def summarize_seed_runs(raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate named model runs over seed and expose n_seeds."""
    required = {"model", "seed", "mae", "rmse", "r2"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Missing final-comparison columns: {sorted(missing)}")

    rows: list[dict[str, Any]] = []
    for model, group in raw.groupby("model", sort=False):
        rows.append(
            {
                "model": model,
                "n_seeds": len(group),
                "mae_mean": float(group["mae"].mean()),
                "mae_std": float(group["mae"].std(ddof=1)) if len(group) > 1 else 0.0,
                "rmse_mean": float(group["rmse"].mean()),
                "rmse_std": float(group["rmse"].std(ddof=1)) if len(group) > 1 else 0.0,
                "r2_mean": float(group["r2"].mean()),
                "r2_std": float(group["r2"].std(ddof=1)) if len(group) > 1 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def write_final_comparison(raw: pd.DataFrame, output: str | Path) -> Path:
    """Write the final seed summary and enforce the shared five-seed rule."""
    unknown = set(raw["seed"].unique()).difference(set(TEAM_SEEDS))
    if unknown:
        raise ValueError(f"Unapproved seed(s): {sorted(unknown)}")
    summary = summarize_seed_runs(raw)
    bad = summary.loc[summary["n_seeds"] != len(TEAM_SEEDS)]
    if not bad.empty:
        raise ValueError(
            f"Every neural model needs 5 seeds; incomplete models: {bad['model'].tolist()}"
        )
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(path, index=False)
    return path
