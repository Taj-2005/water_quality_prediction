"""Shared test configuration.

Most of the suite runs anywhere: the cleaning, feature, metric and preprocessing
tests build small synthetic frames and need no data on disk. A minority need the
Phase 1 artefacts, and those are not all available everywhere:

  * ``data/processed/clean.parquet`` IS committed, so a clone has it, but reading
    it needs a parquet engine.
  * ``data/splits/splits.parquet`` is deliberately git-ignored and regenerated
    locally with ``python -m src.split``, so a CI runner never has it.

Tests that need the split indices must therefore skip rather than fail when they
are absent. Decorate them with ``@requires_phase1_splits``.
"""

from __future__ import annotations

import importlib.util

import pytest

from src.config import DATA_SPLITS

PHASE1_SPLITS_AVAILABLE: bool = (DATA_SPLITS / "splits.parquet").exists() or (
    DATA_SPLITS / "splits.csv"
).exists()

PARQUET_ENGINE_AVAILABLE: bool = (
    importlib.util.find_spec("pyarrow") is not None
    or importlib.util.find_spec("fastparquet") is not None
)

requires_phase1_splits = pytest.mark.skipif(
    not PHASE1_SPLITS_AVAILABLE,
    reason=(
        "Phase 1 split indices not on disk (data/splits/ is git-ignored). "
        "Run 'python -m src.split' to generate them locally."
    ),
)

requires_parquet_engine = pytest.mark.skipif(
    not PARQUET_ENGINE_AVAILABLE,
    reason="No parquet engine installed; cannot read data/processed/clean.parquet.",
)
