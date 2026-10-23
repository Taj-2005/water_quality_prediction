"""Metric definitions.   Owner: Satya Yadav.

Pure functions of (y_true, y_pred), unit-tested on dummy data so the numbers in the
report are trustworthy.

  * continuous target  -> MAE, RMSE, R-squared
  * defined classes    -> macro F1 + calibration curve
"""

from __future__ import annotations


def regression_metrics(y_true, y_pred) -> dict:
    raise NotImplementedError


def classification_metrics(y_true, y_pred) -> dict:
    raise NotImplementedError
