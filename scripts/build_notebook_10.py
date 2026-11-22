"""Script to build and execute notebooks/10_final_comparison.ipynb cleanly.

Ensures:
  1. Full compliance with CI / Notebook hygiene (valid nbformat, all code cells executed, 0 errors).
  2. Professional presentation with rich markdown and clear table displays.
  3. Embedding of generated diagnostic figures.
"""

from __future__ import annotations

import pathlib
import sys

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def create_notebook() -> nbformat.NotebookNode:
    nb = new_notebook()
    nb.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }

    cells = []

    # Cell 1: Header Markdown
    cells.append(
        new_markdown_cell(
            """# 10. Phase 2 Final Model Comparison & Technical Validation

**Author / Owner:** Parthraj Singh Bhati  
**Role:** Integration & Technical Validation Lead (Phase 2 Steps 2 & 3)  
**Collaborators:** Taj (Regularization), Himanshu (Architecture), Jivit (Hybridization), Satya (EDA)  
**Evaluation Interface:** `src.evaluate_api.evaluate()` across all architectures  
**Random Seeds:** Fixed 5-seed protocol `[42, 123, 456, 789, 1011]`  

---

> [!IMPORTANT]
> **Mandatory Benchmark Disclosure:**  
> `CCME_Values` is an index computed deterministically from 8 water quality parameters. Consequently, all evaluations presented herein are **formula-approximation benchmarks**, not predictions of unobserved physical phenomena. All models are scored strictly on the identical saved Phase 1 test partition; this comparison scores the test split once without re-tuning.
"""
        )
    )

    # Cell 2: Imports Code
    cells.append(
        new_code_cell(
            """import pandas as pd
import numpy as np
from pathlib import Path
from IPython.display import display, Image

REPORTS = Path("../reports") if Path("../reports").exists() else Path("reports")
FIGURES = REPORTS / "figures"
print("Reports directory:", REPORTS.resolve())
print("Figures directory:", FIGURES.resolve())
"""
        )
    )

    # Cell 3: Executive Summary Markdown
    cells.append(
        new_markdown_cell(
            """## 1. Multi-Seed Benchmark Evaluation

To provide a fair and rigorous comparison between tree ensembles, standalone deep learning architectures, and residual hybrid models, we evaluate two structured benchmarks:

1. **Table 1: Controlled Comparison Benchmark (Exact 20,000 samples, 5 seeds)**  
   *Every single model* is trained on the exact identical 20,000-sample training slice across all 5 seeds (`[42, 123, 456, 789, 1011]`). This eliminates sample size confounding and matches Jivit's hybrid ablation scale. To ensure complete clarity, both XGBoost configurations are evaluated:
   - **XGBoost (Phase 1 Tuned, 23 Eng Features)**: Uses Phase 1 Optuna hyperparameters (`learning_rate=0.0552, max_depth=7, min_child_weight=30, reg_lambda=33.12`) on 23 engineered features (ratios and log transforms).
   - **XGBoost (Jivit Hybrid Base, 10 Raw Features)**: Uses the base XGBoost configuration from `src/dl/hybrid.py` (`learning_rate=0.1, max_depth=8, min_child_weight=5, reg_lambda=1.0`) on the 10 raw features without feature engineering.

2. **Table 2: Reported Operating Points from Earlier Single-Seed Runs (Different Training Sizes, Not Comparable)**  
   Operating points reported from earlier single-seed experiments across different training partition sizes (ranging from 20,000 to 1,986,397 rows). Because training sample sizes differ by up to two orders of magnitude, these numbers reflect distinct computational operating regimes and cannot be used to rank model architectures.
"""
        )
    )

    # Cell 4: Table 1 Code
    cells.append(
        new_code_cell(
            """# Table 1: Controlled 20,000-Row Benchmark across 5 Random Seeds
df_summary = pd.read_csv(REPORTS / "final_comparison_summary.csv")

ctrl_cols = [
    "model", "training_samples", "n_seeds", 
    "test_rmse_mean", "test_rmse_std", 
    "test_mae_mean", "test_mae_std", 
    "test_p95_ae_mean", "test_p99_ae_mean", "test_max_ae_mean",
    "test_r2_mean", "test_r2_std", 
    "fit_time_s_mean"
]
ctrl_summary = df_summary[df_summary["benchmark"] == "controlled_20k"][ctrl_cols].sort_values("test_rmse_mean")
print("=== TABLE 1: CONTROLLED COMPARISON BENCHMARK (Exact 20,000 samples, 5 seeds) ===")
print("Disclosure: formula-approximation benchmark (CCME_Values computed from 8 input parameters)")
display(ctrl_summary.style.format({
    "test_rmse_mean": "{:.4f}",
    "test_rmse_std": "±{:.4f}",
    "test_mae_mean": "{:.4f}",
    "test_mae_std": "±{:.4f}",
    "test_p95_ae_mean": "{:.4f}",
    "test_p99_ae_mean": "{:.4f}",
    "test_max_ae_mean": "{:.4f}",
    "test_r2_mean": "{:.4f}",
    "test_r2_std": "±{:.4f}",
    "fit_time_s_mean": "{:.2f}s",
    "training_samples": "{:,}"
}))
"""
        )
    )

    # Cell 5: Table 2 Code
    cells.append(
        new_code_cell(
            """# Table 2: Reported Operating Points from Earlier Single-Seed Runs (Different Training Sizes, Not Comparable)
size_col = "size (parameters or tree nodes)_mean" if "size (parameters or tree nodes)_mean" in df_summary.columns else "n_params_mean"
scale_cols = [
    "model", "training_samples", size_col,
    "test_rmse_mean", "test_mae_mean",
    "test_p95_ae_mean", "test_p99_ae_mean", "test_max_ae_mean",
    "test_r2_mean", "fit_time_s_mean"
]
scale_summary = df_summary[df_summary["benchmark"] == "reported_operating_points_not_comparable"][scale_cols].sort_values("test_rmse_mean")
print("=== TABLE 2: REPORTED OPERATING POINTS FROM EARLIER SINGLE-SEED RUNS (DIFFERENT TRAINING SIZES, NOT COMPARABLE) ===")
print("Disclosure: formula-approximation benchmark (CCME_Values computed from 8 input parameters)")
display(scale_summary.style.format({
    "test_rmse_mean": "{:.4f}",
    "test_mae_mean": "{:.4f}",
    "test_p95_ae_mean": "{:.4f}",
    "test_p99_ae_mean": "{:.4f}",
    "test_max_ae_mean": "{:.4f}",
    "test_r2_mean": "{:.4f}",
    size_col: "{:,.0f}",
    "fit_time_s_mean": "{:.2f}s",
    "training_samples": "{:,}"
}))
"""
        )
    )

    # Cell 6: Visualization Markdown & Code
    cells.append(
        new_markdown_cell(
            """### Visual Performance Comparison

The figures below illustrate the test RMSE and $R^2$ scores across models under the controlled 20k benchmark.
"""
        )
    )
    cells.append(
        new_code_cell(
            """fig_path = FIGURES / "final_comparison_models.png"
if fig_path.exists():
    display(Image(filename=str(fig_path)))
else:
    print("Figure not found:", fig_path)
"""
        )
    )

    # Cell 7: Detailed Per-Seed Raw Data
    cells.append(
        new_markdown_cell(
            """## 2. Detailed Per-Seed Empirical Results

Below are the per-seed runs demonstrating the variance and stability across seeds `[42, 123, 456, 789, 1011]`.
"""
        )
    )
    cells.append(
        new_code_cell(
            """df_raw = pd.read_csv(REPORTS / "final_comparison.csv")
ctrl_raw = df_raw[df_raw["benchmark"] == "controlled_20k"][
    ["seed", "model", "test_rmse", "test_mae", "test_p95_ae", "test_p99_ae", "test_max_ae", "test_r2", "fit_time_s"]
].sort_values(["model", "seed"])
display(ctrl_raw)
"""
        )
    )

    # Cell 8: Error Breakdown Markdown & Code
    cells.append(
        new_markdown_cell(
            """## 3. Fine-Grained Error Breakdown (Top 2 Models: XGBoost Tuned vs Tuned MLP)

To understand where individual models perform best, we examine the grouped test residuals across geographic countries and top monitoring station areas.
"""
        )
    )
    cells.append(
        new_code_cell(
            """# Error breakdown across geographic countries
df_country = pd.read_csv(REPORTS / "final_comparison_by_country.csv")
display(df_country)

fig_country = FIGURES / "final_comparison_error_by_country.png"
if fig_country.exists():
    display(Image(filename=str(fig_country)))
"""
        )
    )
    cells.append(
        new_code_cell(
            """# Error breakdown across top 10 monitoring station areas by sample volume
df_area = pd.read_csv(REPORTS / "final_comparison_by_area.csv")
display(df_area)

fig_area = FIGURES / "final_comparison_error_by_area.png"
if fig_area.exists():
    display(Image(filename=str(fig_area)))
"""
        )
    )

    # Cell 9: Written Analytical Reading Markdown
    cells.append(
        new_markdown_cell(
            """### Analytical Reading: When Does the MLP Beat Trees, and When Does It Not?

Based strictly on the empirical values in the benchmark tables:

1. **Test Error Comparisons (Table 1: Controlled 20,000 Rows):**
   * **Standalone Accuracy:** Standalone XGBoost (Phase 1 Tuned, 23 Eng Features) achieves the lowest error across all metrics on the 20k-row sample: Test RMSE of **0.5602**, MAE of **0.2130**, P95 absolute error of **0.8736**, P99 of **2.4743**, and Max absolute error of **11.7142** ($R^2 = 0.9991$).
   * **Comparison on Raw 10 Features:** When evaluated on identical 10 raw features without engineered ratios or log transforms:
     - Standalone XGBoost Base scores Test RMSE **0.6757**, MAE **0.3016**, P95 **1.2090**, and Max **15.0811**.
     - Standalone Tuned MLP scores Test RMSE **1.7295**, MAE **1.1464**, P95 **3.6101**, and Max **36.1018**.
     - Standalone Random Forest scores Test RMSE **1.4543**, MAE **0.6545**, P95 **2.6878**, and Max **24.6490**.
     - Standalone MLP does not beat XGBoost or Random Forest in RMSE, MAE, or tail percentiles on tabular CCME approximation.
   * **Hybrid Residual Correction on Raw Features:** When XGBoost Base is combined with a residual MLP (the Hybrid model, 10 raw features), the residual MLP reduces error relative to standalone XGBoost Base: Test RMSE decreases from **0.6757** to **0.6356**, and P95 absolute error improves from **1.2090** to **1.1865** (P99 from **2.7325** to **2.5806**, Max from **15.0811** to **12.5283**). However, standalone XGBoost with Phase 1 feature engineering (23 features) remains superior across all error metrics (RMSE **0.5602**, P95 **0.8736**).

2. **Inference Throughput and Parameter Footprint (Table 4: Compute Cost Profiling):**
   * **Inference Speed:** The Tuned MLP executes inference on CPU at **0.1882s per 100k rows** via vectorized PyTorch BLAS matrix multiplications, which is **8.9× faster** than XGBoost (**1.6714s per 100k rows**) and **20.3× faster** than full Random Forest (**3.8200s per 100k rows**).
   * **Model Footprint:** The Tuned MLP comprises **204,289 parameters**, while Random Forest comprises **2,191,352 tree nodes** at full scale (398,998 at 100k scale) and XGBoost comprises **109,501 tree nodes**. The Hybrid architecture contains an MLP of **11,809 parameters** (its tree node count is separate).
"""
        )
    )

    # Cell 10: Technical Validation Markdown & Code
    cells.append(
        new_markdown_cell(
            r"""## 4. Technical Validation: Anti-Leakage Audit

To guarantee the integrity of all Phase 2 conclusions, we run a 5-point data leakage and isolation audit:
1. **Scaler Isolation (`LEAK-01`):** Feature scalers (`QuantileTransformer`, `StandardScaler`) fitted strictly on training data; val/test transforms leave quantile arrays strictly unchanged.
2. **Target Isolation (`LEAK-02`):** Target standardization statistics ($\mu_y, \sigma_y$) computed strictly on `train` indices.
3. **Partition Disjointness (`LEAK-03`):** Train, validation, and test index sets have zero intersection.
4. **Site Group Independence (`LEAK-04`):** GroupKFold validation sets maintain strict monitoring station isolation.
5. **Feature Direct Formula Exclusion (`LEAK-05`):** Target `CCME_Values` and post-computation status flags strictly excluded from feature matrices.
"""
        )
    )
    cells.append(
        new_code_cell(
            """df_leakage = pd.read_csv(REPORTS / "leakage_check.csv")
display(df_leakage)
assert df_leakage["status"].isin(["PASS", "PASSED"]).all(), "One or more leakage checks failed!"
print("All anti-leakage audits PASSED successfully.")
"""
        )
    )

    # Cell 11: Compute Cost Markdown & Code
    cells.append(
        new_markdown_cell(
            """## 5. Technical Validation: Compute Cost & Operational Profiling

Measured parameter counts, tree node counts, training wall-clock times, and inference throughput (per 100,000 samples) benchmarked on Apple Silicon CPU.
"""
        )
    )
    cells.append(
        new_code_cell(
            """df_cost = pd.read_csv(REPORTS / "compute_cost.csv")
display(df_cost)
"""
        )
    )

    nb.cells = cells
    return nb


def main():
    root = pathlib.Path(__file__).resolve().parents[1]
    nb_path = root / "notebooks" / "10_final_comparison.ipynb"

    print(f"Creating notebook: {nb_path}")
    nb = create_notebook()

    # Execute the notebook using nbclient
    print("Executing notebook top to bottom...")
    client = NotebookClient(
        nb,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(root / "notebooks")}},
    )
    client.execute()

    # Validate for zero errors
    errors = [
        o for c in nb.cells for o in getattr(c, "outputs", []) if o.get("output_type") == "error"
    ]
    if errors:
        print(f"FAILED: Found {len(errors)} error outputs in executed notebook:")
        for e in errors:
            print(e)
        sys.exit(1)

    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    executed = sum(1 for c in nb.cells if c.cell_type == "code" and c.execution_count)
    total = sum(1 for c in nb.cells if c.cell_type == "code")
    print(
        f"SUCCESS: Wrote {nb_path.name} with {executed}/{total} code cells executed and 0 errors."
    )


if __name__ == "__main__":
    main()
