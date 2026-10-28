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

Writes two files. ``clean.parquet`` (45 MB) is the shared artifact and is committed, so
every member models identical bytes. ``clean.csv`` (586 MB) is a local convenience and
stays git-ignored - GitHub refuses any file above 100 MB.

Run:
    python -m src.clean              # write clean.parquet and clean.csv
    python -m src.clean --no-csv     # parquet only
    python -m src.clean --profile    # print the profile behind the data dictionary
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from src.config import CLEAN_CSV, CLEAN_PARQUET, DATA_PROCESSED, RAW_CSV

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

# --- anomaly screening thresholds -------------------------------------------------
# These SCREEN for values that warrant review. They are not assertions that a reading is
# physically impossible, and rows are flagged rather than dropped so the decision stays
# with the modelling pipeline. See docs/data_dictionary.md for counts and sources.

# The conventional pH scale for natural water is 0-14 (USGS Water Science School, "pH and
# Water"). It is an operational convention, not a physical bound: pH is defined as
# -log10(a_H+), and environmental waters as acidic as pH -3.6 have been measured
# (Nordstrom et al. 2000, doi:10.1021/es990646v). Values outside 0-14 are therefore
# treated as implausible for routine surface-water monitoring, not as impossible.
PH_MIN, PH_MAX = 0.0, 14.0

# PROJECT-SPECIFIC SCREENING HEURISTIC, not a standard or a physical limit. Natural
# surface water does exceed 40 C: USGS documents thermal springs discharging at 50 C and
# boiling at 93 C (Farrar, USGS Fact Sheet 2007-3045). The value is chosen because this
# corpus is routine ambient monitoring rather than geothermal survey, and because the
# flagged readings concentrate in one country in a pattern consistent with a unit-coding
# error (see docs/data_dictionary.md). The count is insensitive to the exact cutoff:
# 35/40/45 C flag 7,576 / 7,284 / 6,818 rows. Reproduce with src/sensitivity_temperature.py.
TEMP_IMPLAUSIBLE_C = 40.0

# Columns screened for negative values. NOTE: this list includes pH, which is a
# logarithmic activity rather than a concentration, so a negative pH is captured by both
# this flag and flag_ph_out_of_range. The overlap is one row and is documented rather
# than changed, because the flag semantics are consumed downstream.
NON_NEGATIVE = [c for c in MEASUREMENTS if c != "Temperature (cel)"]

# Share of rows at a single modal value above which a measurement column is treated as
# showing SUSPECTED upstream imputation. The repeated-value pattern is observed directly;
# that median imputation produced it is an inference from that pattern, not an established
# fact about the upstream pipeline. See docs/data_dictionary.md for the evidence and its
# limits.
IMPUTATION_SHARE_THRESHOLD = 0.05


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
        dtype=dict.fromkeys(CATEGORICALS + ["Area"], "category"),
        low_memory=False,
    )
    # Every one of the 2,827,977 dates parses with this format; errors="raise" keeps it
    # that way, so a future data refresh fails loudly instead of silently producing NaT.
    df["Date_parsed"] = pd.to_datetime(df["Date"], format=DATE_FORMAT, errors="raise")
    return df


def flag_impossible_values(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Flag values that fall outside the screening thresholds. Drops nothing.

    The flags mark observations as POTENTIALLY ANOMALOUS and retained for review. They do
    not establish that a reading is erroneous: each threshold is either a conventional
    operating range (pH) or a project-specific screening choice (temperature), and an
    out-of-range value may equally indicate a unit-coding error, an instrument fault, a
    transcription error, or a genuine extreme.

    Column names and semantics are fixed: downstream preprocessing, the ablation study and
    the report all consume them, so they are documented rather than renamed.
    """
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
    counts[f"temperature above {TEMP_IMPLAUSIBLE_C:.0f} C"] = int(df["flag_temp_implausible"].sum())

    # Sub-zero water temperature is physically possible: seawater at 35 ppt freezes near
    # -2 C (NOAA, "Sea Water"). That is a statement about saline water in general, NOT a
    # claim about these particular records - 30 of the 32 are English rivers, whose
    # salinity is unknown to us. The flag therefore marks them for review and is kept out
    # of flag_any_impossible.
    df["flag_temp_below_zero"] = df["Temperature (cel)"] < 0
    counts["temperature below 0 C (review only)"] = int(df["flag_temp_below_zero"].sum())

    df["flag_any_impossible"] = (
        df["flag_ph_out_of_range"] | df["flag_negative_measurement"] | df["flag_temp_implausible"]
    )
    counts["rows with any impossible value"] = int(df["flag_any_impossible"].sum())
    return df, counts


def flag_upstream_imputation(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """Flag values that were median-imputed before the dataset reached us.

    The file reports zero missing values, but five of the eight measurement columns
    carry their own exact median as the single most frequent value, at 11-32% of rows.
    No natural measurement distribution does that; it is the signature of a median fill
    applied upstream, during the harmonization we did not run.

    This matters beyond tidiness. The project rule is that imputation is fitted on the
    training split only, and that rule is already broken by the data as delivered: the
    upstream fill used medians computed over every row, including the rows that will
    become our test set. We cannot undo it, so we mark it, and let Parth's pipeline and
    Jivit's ablation measure the cost instead of inheriting it silently.

    Adds, per affected column, ``<column>__imputed_upstream``; then
    ``n_imputed_upstream`` (how many of a row's measurements are suspect) and
    ``flag_any_imputed_upstream``.
    """
    report: dict[str, dict[str, float]] = {}
    flag_cols: list[str] = []

    for col in MEASUREMENTS:
        counts = df[col].value_counts()
        modal_value = counts.index[0]
        share = counts.iloc[0] / len(df)
        if share < IMPUTATION_SHARE_THRESHOLD:
            continue

        flag = f"{col}__imputed_upstream"
        df[flag] = df[col] == modal_value
        flag_cols.append(flag)
        report[col] = {
            "modal_value": float(modal_value),
            "rows": int(counts.iloc[0]),
            "share": float(share),
            "median": float(df[col].median()),
            # The median signature is the strong evidence. Where the modal value is not
            # the median the fill is likelier a per-country or per-site constant, so the
            # flag is still raised but the finding is weaker - stated, not overstated.
            "equals_median": bool(np.isclose(modal_value, df[col].median())),
        }

    if flag_cols:
        df["n_imputed_upstream"] = df[flag_cols].sum(axis=1).astype("int8")
        df["flag_any_imputed_upstream"] = df["n_imputed_upstream"] > 0
    else:
        df["n_imputed_upstream"] = np.int8(0)
        df["flag_any_imputed_upstream"] = False

    return df, report


def drop_exact_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove rows identical across all 14 original columns.

    Compared on the raw columns only - the flags and the parsed date are derived, so
    including them would not change which rows match, but excluding them keeps the
    definition of "duplicate" tied to the source data.
    """
    derived = ("flag_", "n_imputed_upstream")
    original = [
        c
        for c in df.columns
        if not c.startswith(derived) and not c.endswith("__imputed_upstream") and c != "Date_parsed"
    ]
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
        # Test for "not numeric" rather than "is object": pandas 3 gives string columns
        # the dedicated `str` dtype, so an `== object` test silently never fires and this
        # check would quietly pass on exactly the data it exists to catch.
        if not is_numeric_dtype(df[col]):
            hits[col] = int(df[col].astype(str).str.contains("<|>", regex=True).sum())
    return hits


def clean(verbose: bool = True) -> pd.DataFrame:
    """Run the full cleaning pipeline and return the cleaned frame."""
    log = print if verbose else (lambda *a, **k: None)

    log(f"reading {RAW_CSV.name} ...")
    df = load_raw()
    log(f"  raw: {len(df):,} rows x {df.shape[1] - 1} columns")

    censored = check_detection_limits(df)
    log(
        f"  detection-limit values in measurement columns: "
        f"{sum(censored.values()) if censored else 0}"
    )

    df, counts = flag_impossible_values(df)
    for label, n in counts.items():
        log(f"  flagged {label}: {n:,}")

    df, imputation = flag_upstream_imputation(df)
    for col, info in imputation.items():
        log(
            f"  flagged upstream imputation in {col}: {info['rows']:,} rows "
            f"({100 * info['share']:.2f}%) at value {info['modal_value']:g}"
            f"{' = column median' if info['equals_median'] else ''}"
        )
    log(
        f"  columns showing the median signature: "
        f"{sum(i['equals_median'] for i in imputation.values())} of {len(MEASUREMENTS)}"
    )

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
    print(
        f"\nrows={n:,}  unique Area={df['Area'].nunique():,}  "
        f"dates {df['Date_parsed'].min().date()} -> {df['Date_parsed'].max().date()}"
    )
    print("\nmeasurement columns:")
    for col in MEASUREMENTS:
        s = df[col]
        mode = s.value_counts().head(1)
        print(
            f"  {col:36s} min={s.min():>9.3f} max={s.max():>9.3f} "
            f"mean={s.mean():>8.3f} median={s.median():>8.3f} "
            f"mode={mode.index[0]:<8} ({100 * mode.iloc[0] / n:5.2f}%)"
        )
    print("\ncategorical columns:")
    for col in CATEGORICALS:
        print(f"  {col}:")
        for k, v in df[col].value_counts().items():
            print(f"      {str(k):22s} {v:>10,}  ({100 * v / n:5.2f}%)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--profile",
        action="store_true",
        help="print the dataset profile instead of writing the output",
    )
    ap.add_argument(
        "--no-csv", action="store_true", help="write only clean.parquet, skip the 586 MB CSV"
    )
    args = ap.parse_args()

    df = clean()
    if args.profile:
        profile(df)
        return

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    # Parquet is the shared artifact: it is committed, so every member models identical
    # bytes instead of trusting that each rerun produced the same file. Categories are
    # restored first so the dtypes survive the round trip.
    for col in CATEGORICALS + ["Area"]:
        df[col] = df[col].astype("category")
    df.to_parquet(CLEAN_PARQUET, compression="zstd", index=False)
    print(f"\nwrote {CLEAN_PARQUET.name}  ({CLEAN_PARQUET.stat().st_size / 1e6:.0f} MB, committed)")

    if not args.no_csv:
        df.to_csv(CLEAN_CSV, index=False)
        print(f"wrote {CLEAN_CSV.name}  ({CLEAN_CSV.stat().st_size / 1e6:.0f} MB, git-ignored)")
    print(f"{len(df):,} rows x {df.shape[1]} columns")


if __name__ == "__main__":
    main()
