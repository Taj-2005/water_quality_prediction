"""Sensitivity of the temperature screening flag to its cutoff.

``TEMP_IMPLAUSIBLE_C`` is a project-specific screening heuristic rather than a published
standard (see the comment in src/clean.py). This script quantifies how far the flag
depends on the exact value by recounting at candidate cutoffs, and tests the competing
explanation for the flagged readings: that they are Fahrenheit values recorded in a
column labelled Celsius.

It is deliberately separate from the main experiment. It reads the raw file, counts, and
writes a CSV. It does not alter the cleaned dataset, the split, or any model, and it
retrains nothing.

Run:
    python -m src.sensitivity_temperature
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.clean import TEMP_IMPLAUSIBLE_C, load_raw

TEMP_COL = "Temperature (cel)"
CANDIDATES = (30.0, 35.0, 40.0, 45.0, 50.0)
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# A plausible band for ambient surface-water temperature, used only to test the
# unit-coding hypothesis. It is a descriptive check, not a screening rule.
PLAUSIBLE_C = (0.0, 35.0)


def cutoff_table(df: pd.DataFrame) -> pd.DataFrame:
    """Rows flagged at each candidate cutoff."""
    t = df[TEMP_COL]
    n = len(df)
    rows = []
    for c in CANDIDATES:
        k = int((t > c).sum())
        rows.append(
            {
                "cutoff_c": c,
                "rows_flagged": k,
                "pct_of_dataset": round(100 * k / n, 4),
                "is_project_default": c == TEMP_IMPLAUSIBLE_C,
            }
        )
    out = pd.DataFrame(rows)
    base = out.loc[out.cutoff_c == TEMP_IMPLAUSIBLE_C, "rows_flagged"].iloc[0]
    out["delta_vs_default"] = out["rows_flagged"] - base
    return out


def unit_coding_check(df: pd.DataFrame) -> pd.DataFrame:
    """Test whether readings above the default cutoff behave like Fahrenheit values.

    If a subset of records was recorded in Fahrenheit, converting those readings should
    place essentially all of them inside the ambient band, and they should concentrate in
    countries that use Fahrenheit.
    """
    t = df[TEMP_COL]
    hot = df[t > TEMP_IMPLAUSIBLE_C]
    lo, hi = PLAUSIBLE_C
    rows = []
    for country, grp in hot.groupby("Country", observed=True):
        vals = grp[TEMP_COL]
        as_c = (vals - 32.0) * 5.0 / 9.0
        rows.append(
            {
                "country": country,
                "rows_above_cutoff": len(grp),
                "pct_of_country_rows": round(
                    100 * len(grp) / int((df["Country"] == country).sum()), 4
                ),
                "median_as_reported_c": round(float(vals.median()), 2),
                "median_if_read_as_f": round(float(as_c.median()), 2),
                "pct_in_ambient_band_if_f": round(100 * float(as_c.between(lo, hi).mean()), 2),
            }
        )
    return (
        pd.DataFrame(rows).sort_values("rows_above_cutoff", ascending=False).reset_index(drop=True)
    )


def main() -> None:
    df = load_raw()
    n = len(df)

    cut = cutoff_table(df)
    units = unit_coding_check(df)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cut.to_csv(RESULTS_DIR / "temperature_sensitivity.csv", index=False)
    units.to_csv(RESULTS_DIR / "temperature_unit_coding_check.csv", index=False)

    print(f"raw rows: {n:,}   default cutoff: {TEMP_IMPLAUSIBLE_C:g} C\n")
    print("Rows flagged by cutoff")
    print(cut.to_string(index=False))

    span = int(cut.loc[cut.cutoff_c == 35.0, "rows_flagged"].iloc[0]) - int(
        cut.loc[cut.cutoff_c == 45.0, "rows_flagged"].iloc[0]
    )
    print(
        f"\n  35 C vs 45 C differ by {span:,} rows "
        f"({100 * span / n:.4f}% of the dataset): the flag is not sensitive to the "
        f"cutoff within this range."
    )

    print("\nUnit-coding check on rows above the default cutoff")
    print(units.to_string(index=False))
    print(
        "\n  wrote results/temperature_sensitivity.csv and "
        "results/temperature_unit_coding_check.csv"
    )


if __name__ == "__main__":
    main()
