"""Shared interface every member codes against.

This is the contract: no module loads the dataset, builds a split, or defines a
metric on its own. If two members call ``get_splits()`` and get different data,
that is a bug to fix before any result is reported.

Owner: Parthraj Singh Bhati. Implementations land in clean.py / split.py / metrics.py.
"""

from __future__ import annotations

import pandas as pd


def load_data() -> pd.DataFrame:
    """Return the cleaned, documented table from ``data/processed/clean.csv``.

    No imputation and no scaling happen here — those belong to the preprocessing
    pipeline, which is fitted on the training split only.
    """
    raise NotImplementedError("see src/clean.py (Taj) and src/config.CLEAN_CSV")


def get_splits():
    """Return the one shared, seeded split.

    Returns
    -------
    tuple
        ``(X_train, X_val, X_test, y_train, y_val, y_test)``

    Notes
    -----
    Split by site or time where the metadata permits, otherwise a seeded random
    split. Indices are persisted under ``data/splits/`` so the split is identical
    for every member and across runs. The test set is used exactly once, in the
    final evaluation.
    """
    raise NotImplementedError("see src/split.py (Parth)")


def evaluate(y_true, y_pred) -> dict:
    """Score predictions with the project's agreed metrics.

    MAE / RMSE / R-squared for a continuous target, or macro F1 plus a calibration
    curve for a defined class scheme. Returns a dict so results can be written
    straight to a CSV row in ``reports/``.
    """
    raise NotImplementedError("see src/metrics.py (Satya)")
