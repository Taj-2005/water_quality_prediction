"""Random Forest.   Owner: Himanshu Pandey.

Hyperparameter search over number of trees, depth, minimum leaf size and features
per split, selected on the VALIDATION set only. Writes reports/rf_results.csv using
the shared evaluate() function.

Keep this signature aligned with models_xgb.train() so the Phase 2 MLP plugs into
the same evaluation path.
"""

from __future__ import annotations


def train():
    raise NotImplementedError


if __name__ == "__main__":
    train()
