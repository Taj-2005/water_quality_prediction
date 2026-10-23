"""Derived features.   Owner: Himanshu Pandey.

Ratios between chemical parameters, log transforms for skewed columns, and time
features if a date column exists.

Two hard rules:
  * any aggregate (e.g. per-site averages) is computed from TRAINING rows only
  * run a leakage check - no feature may be a formula of the target - and document it
"""

from __future__ import annotations


def build_features(df):
    raise NotImplementedError
