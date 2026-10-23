"""Raw dataset -> data/processed/clean.csv.   Owner: Shaik Tajuddin.

Rules (from the Phase 1 work plan, enforced here and logged in 01_cleaning.ipynb):
  * make units consistent
  * handle detection-limit values such as "<0.5" with ONE stated rule
  * flag impossible values (pH outside 0-14, negative concentrations) instead of
    deleting them
  * remove exact duplicates
  * do NOT impute here - imputation belongs in preprocess.py, fitted on train only

Two of those turned out to be no-ops for this dataset, and that is itself a finding:
units are already consistent per column, and no detection-limit values exist in any
measurement column (the "<" characters in the file are all inside Area site names).
Both are documented in docs/data_dictionary.md rather than silently skipped.

Run:
    python -m src.clean              # write data/processed/clean.csv
    python -m src.clean --profile    # print the profile behind the data dictionary
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from src.config import CLEAN_CSV, DATA_PROCESSED, RAW_CSV

# Column names are kept verbatim. src/config.py binds TARGET_COLUMN = "CCME_Values",
# so renaming columns here would break Parth's interface and every downstream module.
MEASUREMENTS = [
    "Ammonia (mg/l)",
    "Biochemical Oxygen Demand (mg/l)",
    "Dissolved Oxygen (mg/l)",
    "Orthophosphate (mg/l)",
    "pH (ph units)",
    "Temperature (cel)",
    "Nitrogen (mg/l)",
    "Nitrate (mg/l)",
]
CATEGORICALS = ["Country", "Waterbody Type", "CCME_WQI"]
DATE_FORMAT = "%d-%m-%Y"

# Physically impossible or implausible readings. Rows are flagged, never dropped:
# deleting them would hide a data-quality problem the report has to discuss.
PH_MIN, PH_MAX = 0.0, 14.0
TEMP_IMPLAUSIBLE_C = 40.0        # observed max is 98 C, impossible for surface water
NON_NEGATIVE = [c for c in MEASUREMENTS if c != "Temperature (cel)"]


def load_raw() -> pd.DataFrame:
    """Read the raw CSV with explicit dtypes and a parsed date column.

    ``Area`` is read as a category: 61,689 unique values over 2.8M rows makes the
    difference between roughly 1.5 GB and 400 MB of memory.
    """
    if not RAW_CSV.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at '{RAW_CSV}'. The file is tracked with Git LFS - "
            "run 'git lfs install' then 'git lfs pull' to fetch it."
        )
    df = pd.read_csv(
        RAW_CSV,
        dtype={c: "category" for c in CATEGORICALS + ["Area"]},
        low_memory=False,
    )
    # Every one of the 2,827,977 dates parses with this format; errors="raise" keeps it
    # that way, so a future data refresh fails loudly instead of silently producing NaT.
    df["Date_parsed"] = pd.to_datetime(df["Date"], format=DATE_FORMAT, errors="raise")
    return df


def flag_impossible_values(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Add boolean flags for physically impossible readings. Drops nothing."""
    counts: dict[str, int] = {}

    ph = df["pH (ph units)"]
    df["flag_ph_out_of_range"] = (ph < PH_MIN) | (ph > PH_MAX)
    counts["pH outside 0-14"] = int(df["flag_ph_out_of_range"].sum())

    negative = pd.Series(False, index=df.index)
    for col in NON_NEGATIVE:
        negative |= df[col] < 0
    df["flag_negative_measurement"] = negative
    counts["negative concentration"] = int(negative.sum())

    df["flag_temp_implausible"] = df["Temperature (cel)"] > TEMP_IMPLAUSIBLE_C
    counts[f"temperature above {TEMP_IMPLAUSIBLE_C:.0f} C"] = int(
        df["flag_temp_implausible"].sum()
    )

    # Sub-zero water temperature is plausible for saline water, so it is flagged for
    # review rather than treated as impossible.
    df["flag_temp_below_zero"] = df["Temperature (cel)"] < 0
    counts["temperature below 0 C (review only)"] = int(df["flag_temp_below_zero"].sum())

    df["flag_any_impossible"] = (
        df["flag_ph_out_of_range"]
        | df["flag_negative_measurement"]
        | df["flag_temp_implausible"]
    )
    counts["rows with any impossible value"] = int(df["flag_any_impossible"].sum())
    return df, counts


def drop_exact_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove rows identical across all 14 original columns.

    Compared on the raw columns only - the flags and the parsed date are derived, so
    including them would not change which rows match, but excluding them keeps the
    definition of "duplicate" tied to the source data.
    """
    original = [c for c in df.columns if not c.startswith("flag_") and c != "Date_parsed"]
    before = len(df)
    df = df.drop_duplicates(subset=original, keep="first").reset_index(drop=True)
    return df, before - len(df)


def check_detection_limits(df: pd.DataFrame) -> dict[str, int]:
    """Confirm no measurement column contains censored values such as "<0.5".

    Kept as an executable check rather than a comment: if a data refresh introduces
    detection limits, the cleaning notebook will report them instead of coercing them
    to NaN unnoticed.
    """
    hits = {}
    for col in MEASUREMENTS:
        if df[col].dtype == object:
            hits[col] = int(df[col].astype(str).str.contains("<|>", regex=True).sum())
    return hits


def clean(verbose: bool = True) -> pd.DataFrame:
    """Run the full cleaning pipeline and return the cleaned frame."""
    log = print if verbose else (lambda *a, **k: None)

    log(f"reading {RAW_CSV.name} ...")
    df = load_raw()
    log(f"  raw: {len(df):,} rows x {df.shape[1] - 1} columns")

    censored = check_detection_limits(df)
    log(f"  detection-limit values in measurement columns: "
        f"{sum(censored.values()) if censored else 0}")

    df, counts = flag_impossible_values(df)
    for label, n in counts.items():
        log(f"  flagged {label}: {n:,}")

    df, removed = drop_exact_duplicates(df)
    log(f"  removed exact duplicates: {removed:,}  ({100 * removed / (len(df) + removed):.2f}%)")
    log(f"  clean: {len(df):,} rows x {df.shape[1]} columns")

    # No imputation here, by design: it belongs in src/preprocess.py so it can be fitted
    # on the training split only.
    assert df.isna().sum().sum() == 0, "unexpected missing values after cleaning"
    return df


def profile(df: pd.DataFrame) -> None:
    """Print the figures that back docs/data_dictionary.md."""
    n = len(df)
    print(f"\nrows={n:,}  unique Area={df['Area'].nunique():,}  "
          f"dates {df['Date_parsed'].min().date()} -> {df['Date_parsed'].max().date()}")
    print("\nmeasurement columns:")
    for col in MEASUREMENTS:
        s = df[col]
        mode = s.value_counts().head(1)
        print(f"  {col:36s} min={s.min():>9.3f} max={s.max():>9.3f} "
              f"mean={s.mean():>8.3f} median={s.median():>8.3f} "
              f"mode={mode.index[0]:<8} ({100 * mode.iloc[0] / n:5.2f}%)")
    print("\ncategorical columns:")
    for col in CATEGORICALS:
        print(f"  {col}:")
        for k, v in df[col].value_counts().items():
            print(f"      {str(k):22s} {v:>10,}  ({100 * v / n:5.2f}%)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", action="store_true",
                    help="print the dataset profile instead of writing clean.csv")
    args = ap.parse_args()

    df = clean()
    if args.profile:
        profile(df)
        return

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_CSV, index=False)
    print(f"\nwrote {CLEAN_CSV.relative_to(CLEAN_CSV.parents[2])} "
          f"({CLEAN_CSV.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
