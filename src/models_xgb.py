"""XGBoost.   Owner: Jivit Rana.

Optuna or randomized search on the VALIDATION set only, with early stopping; tunes
learning rate, depth, number of trees, subsample and regularization. Writes
reports/xgb_results.csv using the shared evaluate() function.

Also runs the ablation: with and without Himanshu's engineered features on the
identical split.
"""

from __future__ import annotations


def train():
    raise NotImplementedError


if __name__ == "__main__":
    train()
