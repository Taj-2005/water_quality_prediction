"""Single source of truth for paths and the random seed.   Owner: Shaik Tajuddin.

Every module imports SEED from here. Nothing hardcodes a seed locally, otherwise
results stop being reproducible across members.
"""

from pathlib import Path

SEED = 42

ROOT = Path(__file__).resolve().parent.parent

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_SPLITS = ROOT / "data" / "splits"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

CLEAN_CSV = DATA_PROCESSED / "clean.csv"

# Set once docs/target_definition.md is agreed. Everything downstream reads these.
TARGET_COLUMN: str | None = None
TARGET_KIND: str | None = None  # "regression" or "classification"
