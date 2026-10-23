"""Raw dataset -> data/processed/clean.csv.   Owner: Shaik Tajuddin.

Rules (from the work plan, enforced here and documented in 01_cleaning.ipynb):
  * make units consistent
  * handle detection-limit values such as "<0.5" with ONE stated rule
  * flag impossible values (pH outside 0-14, negative turbidity) instead of deleting them
  * remove exact duplicates
  * do NOT impute here - imputation belongs in preprocess.py, fitted on train only

Every decision is logged with its reason, because examiners will ask why.
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
