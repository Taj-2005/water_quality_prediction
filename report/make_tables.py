"""Generate LaTeX table fragments from results/*.csv.

Numbers in the report are never typed by hand. Re-run this after any experiment and
the report picks up the new values on the next compile:

    python report/make_tables.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = Path(__file__).resolve().parent / "tables"


def write(name: str, body: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(body.rstrip() + "\n")
    print(f"  wrote tables/{name}")


def num(series: pd.Series, dp: int) -> list[str]:
    """Fixed-decimal strings. round() drops trailing zeros, which looks sloppy in a
    table where a column should line up on the decimal point."""
    return [f"{v:.{dp}f}" for v in series]


def esc(text: object) -> str:
    """Escape the LaTeX specials that appear in our labels (8/8, R2, %).

    Pass raw characters in, not pre-escaped ones: this runs over every header and
    cell, so a label written as "\\%" would come out as "\\\\%" and render literally.
    """
    s = str(text)
    for a, b in [("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")]:
        s = s.replace(a, b)
    return s


def tabular(df: pd.DataFrame, align: str, caption: str, label: str, star: bool = False) -> str:
    env = "table*" if star else "table"
    rows = "\n".join("        " + " & ".join(esc(v) for v in row) + r" \\" for row in df.values)
    return f"""\\begin{{{env}}}[!t]
    \\centering
    \\caption{{{caption}}}
    \\label{{{label}}}
    \\small
    \\setlength{{\\tabcolsep}}{{4pt}}
    \\begin{{tabular}}{{{align}}}
        \\toprule
        {" & ".join(esc(c) for c in df.columns)} \\\\
        \\midrule
{rows}
        \\bottomrule
    \\end{{tabular}}
\\end{{{env}}}"""


# Source labels are verbose for CSV readers; a two-column table needs short ones.
# The detail they carry is moved into the captions instead of being dropped.
SHORTEN = {
    "Naive Baseline (Median)": "Naive baseline",
    "Random Forest (100k, Eng)": "Random Forest",
    "XGBoost (Full, Eng)": "XGBoost",
    "Upstream Imputation Flags": "Imputation flags",
    "Feature Engineering": "Features",
    "Without Imputation Flags (Primary)": "Excluded (primary)",
    "With Imputation Flags (Ablation)": "Included",
    "Raw Features": "Raw",
    "Raw + Engineered Features": "Raw + engineered",
    "China (In-Sample / Train Only - Hou Bay single area)": "China$^{\\dagger}$",
}


def short(series: pd.Series) -> list[str]:
    return [SHORTEN.get(str(v), str(v)) for v in series]


def main() -> None:
    print("generating LaTeX tables from results/")

    # --- headline comparison of the three models -------------------------------
    m = pd.read_csv(RESULTS / "metrics.csv")
    t = pd.DataFrame(
        {
            "Model": short(m["Model"]),
            "MAE": num(m["Test MAE"], 3),
            "RMSE": num(m["Test RMSE"], 3),
            "$R^2$": num(m["Test R2"], 4),
            "Train (s)": num(m["Train Time (s)"], 1),
        }
    )
    write(
        "tab_main_results.tex",
        tabular(
            t,
            "lrrrr",
            "Phase 1 model comparison on the held-out test set, all-measurements "
            "regime; all figures are test-set. The baseline predicts the training "
            "median; Random Forest is fitted "
            "on a 100k subsample and XGBoost on the full training partition, both with "
            "engineered features.",
            "tab:main",
        ),
    )

    # --- the partial-measurement ladder: the project's central result ----------
    p = pd.read_csv(RESULTS / "partial_measurement_results.csv")
    t = pd.DataFrame(
        {
            "Avail.": p["Measurement Availability"],
            "Model": p["Model"],
            "$p$": p["Features"],
            "MAE": num(p["Test MAE"], 3),
            "RMSE": num(p["Test RMSE"], 3),
            "$R^2$": num(p["Test R2"], 4),
        }
    )
    write(
        "tab_partial.tex",
        tabular(
            t,
            "llrrrr",
            "Degradation under partial measurement availability. The 8/8 row is a formula-"
            "emulation benchmark, not independent prediction; genuine estimation begins at "
            "7/8. $p$ is the feature count after masking; all figures are test-set.",
            "tab:partial",
            star=True,
        ),
    )

    # --- ablations -------------------------------------------------------------
    a = pd.read_csv(RESULTS / "ablation_results.csv")
    t = pd.DataFrame(
        {
            "Ablation": short(a["Ablation Domain"]),
            "Config.": short(a["Configuration"]),
            "$p$": a["n_features"],
            "MAE": num(a["Test MAE"], 4),
            "RMSE": num(a["Test RMSE"], 4),
            "$R^2$": num(a["Test R2"], 5),
        }
    )
    write(
        "tab_ablation.tex",
        tabular(
            t,
            "lllrrr",
            "Ablation studies on XGBoost, all-measurements regime. $p$ is the feature "
            "count; all figures are test-set.",
            "tab:ablation",
            star=True,
        ),
    )

    # --- uncertainty -----------------------------------------------------------
    b = pd.read_csv(RESULTS / "bootstrap_ci_results.csv")
    write(
        "tab_bootstrap.tex",
        tabular(
            b,
            "lllp{0.28\\linewidth}",
            "Bootstrap point estimates with 95\\% confidence intervals on the test set.",
            "tab:bootstrap",
            star=True,
        ),
    )

    # --- error by country ------------------------------------------------------
    c = pd.read_csv(RESULTS / "country_results.csv")
    # China has zero test records - its single site is allocated wholly to training -
    # and src/run_evaluation.py now writes its in-sample diagnostic to a separate file
    # and asserts this table's percentages sum to 100. Guard anyway, cheaply.
    assert abs(c["pct_of_total"].sum() - 100.0) < 0.01, (
        f"country_results.csv should sum to 100%, got {c['pct_of_total'].sum():.2f}"
    )
    t = pd.DataFrame(
        {
            "Country": short(c["Country"]),
            "$n$": c["n_samples"],
            "% of test": num(c["pct_of_total"], 2),
            "MAE": num(c["mae"], 3),
            "RMSE": num(c["rmse"], 3),
            "$R^2$": num(c["r2"], 4),
        }
    )
    write(
        "tab_country.tex",
        tabular(
            t,
            "lrrrrr",
            "Test error decomposed by country, over all 415{,}804 test records. China "
            "is absent by construction: its single site (Hou Bay) is allocated wholly "
            "to training, so it contributes no test records and cannot be evaluated "
            "out-of-sample.",
            "tab:country",
        ),
    )

    # --- SHAP ------------------------------------------------------------------
    s = pd.read_csv(RESULTS / "shap_importance.csv").head(10)
    t = pd.DataFrame(
        {
            "Feature": s["Feature"],
            "Mean $|$SHAP$|$": num(s["Mean_Abs_SHAP"], 3),
        }
    )
    write(
        "tab_shap.tex",
        tabular(
            t,
            "lr",
            "Ten most influential features by mean absolute SHAP value (XGBoost, 8/8 regime).",
            "tab:shap",
        ),
    )


if __name__ == "__main__":
    main()
