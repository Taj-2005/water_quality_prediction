"""Neural-network data preparation.   Owner: Shaik Tajuddin.

Phase 2 work plan, Taj Step 1: decide which features need a transform because of
skew, which scaling choice suits a neural network, how the upstream-imputation
flags should be represented, and whether the categoricals want one-hot encoding
or learned embeddings.

Tree ensembles split on order, so none of this mattered in Phase 1. A neural
network is not invariant to monotone transforms: it multiplies inputs by weights
and sums them, so a feature whose validation values reach 48 standard deviations
dominates every unit it feeds, and saturates smooth activations.

Everything here is fitted on the TRAINING partition only and applied unchanged to
validation and test. The split itself is never regenerated.

The four strategies, and the evidence for choosing between them, are recorded in
notebooks/11_nn_data_prep.ipynb. Summary of what the training data shows:

  * Five of the eight measurements contain negative values, so a plain log1p is
    undefined for them. Those negatives are the anomalies Phase 1 deliberately
    flagged and retained rather than deleting, so this is a direct consequence of
    that decision, not a data error.
  * log1p is actively harmful on the two columns that are not right-skewed: it
    drives dissolved oxygen from -1.39 to -3.17 and pH from 1.58 to -2.91.
  * RobustScaler, the usual advice when outliers are present, is far worse here.
    It divides by the IQR, and on a long right tail the IQR is small, so the tail
    explodes: max |z| on validation is 740 against 48 for StandardScaler.

Run:
    python -m src.dl.nn_preprocess --report       # evidence table, no training
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from src.clean import MEASUREMENTS
from src.config import SEED

Strategy = Literal["phase1", "yeo-johnson", "quantile", "robust"]

#: Categorical covariates carried into Phase 2. ``Area`` is deliberately absent:
#: it is the grouping key for the split, so under site-held-out evaluation every
#: test site is unseen and the column can only add noise.
CATEGORICAL_FEATURES = ["Country", "Waterbody Type"]

#: Columns that are already 0/1 indicators. They are passed through unscaled --
#: see ``_flag_columns`` and the note in ``build_nn_preprocessor``.
FLAG_SUFFIX = "__imputed_upstream"
FLAG_EXACT = ("flag_any_imputed_upstream",)
COUNT_COLUMNS = ("n_imputed_upstream",)


@dataclass
class PreprocessReport:
    """Evidence behind the transform choice, for the notebook and the report."""

    skew_before: dict[str, float] = field(default_factory=dict)
    skew_after: dict[str, float] = field(default_factory=dict)
    negatives: dict[str, int] = field(default_factory=dict)
    max_abs_z: float = float("nan")
    pct_beyond_5sd: float = float("nan")
    strategy: str = ""

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "skew_before": pd.Series(self.skew_before),
                "skew_after": pd.Series(self.skew_after),
                "negative_values": pd.Series(self.negatives),
            }
        ).round(3)


def _flag_columns(columns) -> list[str]:
    """Binary indicator columns present in the frame."""
    return [c for c in columns if c.endswith(FLAG_SUFFIX) or c in FLAG_EXACT]


def _numeric_columns(columns) -> list[str]:
    """Continuous measurements present in the frame."""
    return [c for c in columns if c in MEASUREMENTS]


def build_nn_preprocessor(
    X_train: pd.DataFrame,
    strategy: Strategy = "phase1",
    categorical: Literal["onehot", "ordinal"] = "onehot",
    seed: int = SEED,
):
    """Fit a neural-network preprocessing pipeline on the training partition.

    Parameters
    ----------
    X_train : DataFrame
        Training features only. Nothing from validation or test informs the fit.
    strategy : {'phase1', 'yeo-johnson', 'quantile', 'robust'}
        How to condition the continuous measurements.

        ``phase1``
            Standardisation alone, matching the Phase 1 pipeline. The default,
            because the work plan says to keep Phase 1 unless a new pipeline is
            clearly better.
        ``yeo-johnson``
            Power transform then standardise. Handles the negative values that
            rule out log1p, and is monotone, so the ordering of a measurement is
            preserved and the model stays interpretable in the original units.
        ``quantile``
            Map each feature to a normal distribution by rank. Best-conditioned
            input by a wide margin, but it distorts distances between values:
            equal gaps in concentration no longer map to equal gaps in the input.
        ``robust``
            Median and IQR. Included because it is the conventional answer to
            outliers, and because on this data it is measurably the worst.
    categorical : {'onehot', 'ordinal'}
        ``onehot`` is the default. ``ordinal`` emits integer codes instead, which
        is the form an embedding layer expects; it is provided so the architecture
        track can test embeddings without rebuilding the pipeline.
    seed : int
        Seed for the quantile transformer's subsampling.

    Returns
    -------
    sklearn ColumnTransformer
        Fitted. Apply with ``.transform`` to validation and test.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import (
        OneHotEncoder,
        OrdinalEncoder,
        PowerTransformer,
        QuantileTransformer,
        RobustScaler,
        StandardScaler,
    )

    num_cols = _numeric_columns(X_train.columns)
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in X_train.columns]
    flag_cols = _flag_columns(X_train.columns)
    count_cols = [c for c in COUNT_COLUMNS if c in X_train.columns]

    if strategy == "phase1":
        numeric = Pipeline([("scale", StandardScaler())])
    elif strategy == "yeo-johnson":
        numeric = Pipeline(
            [("power", PowerTransformer(method="yeo-johnson")), ("scale", StandardScaler())]
        )
    elif strategy == "quantile":
        numeric = Pipeline(
            [
                (
                    "quantile",
                    QuantileTransformer(
                        output_distribution="normal",
                        n_quantiles=1000,
                        subsample=200_000,
                        random_state=seed,
                    ),
                )
            ]
        )
    elif strategy == "robust":
        numeric = Pipeline([("scale", RobustScaler())])
    else:
        raise ValueError(
            f"Unknown strategy {strategy!r}. Expected one of: "
            "'phase1', 'yeo-johnson', 'quantile', 'robust'."
        )

    transformers = [("num", numeric, num_cols)]

    if cat_cols:
        if categorical == "onehot":
            # Country has 5 levels and Waterbody Type 12, so one-hot costs 17
            # columns. Embeddings earn their keep at high cardinality; at this
            # size they would add parameters and a tuning axis for no new
            # capacity. 'ordinal' exists so that claim can be tested rather than
            # assumed.
            enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore", dtype=np.float32)
        else:
            enc = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.float32
            )
        transformers.append(("cat", enc, cat_cols))

    if flag_cols:
        # Binary indicators are passed through unscaled. Standardising a 0/1
        # column rescales it by its prevalence in the training partition, so the
        # "absent" state stops being zero and the feature's magnitude becomes a
        # function of how common the flag happened to be -- which interacts badly
        # with weight decay and with dropout's zero-masking.
        transformers.append(("flags", "passthrough", flag_cols))

    if count_cols:
        # n_imputed_upstream is an ordered count on 0-8, not an indicator, so it
        # is scaled like a continuous feature.
        transformers.append(("counts", Pipeline([("scale", StandardScaler())]), count_cols))

    pre = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)
    pre.fit(X_train)
    return pre


def skew_report(X_train: pd.DataFrame, strategy: Strategy = "yeo-johnson") -> PreprocessReport:
    """Measure skew before and after a strategy, and count blocking negatives.

    The negative counts matter because they are what rules out ``log1p``: the
    transform is undefined below -1 and is the obvious first choice for data this
    right-skewed.
    """
    rep = PreprocessReport(strategy=strategy)
    num_cols = _numeric_columns(X_train.columns)

    pre = build_nn_preprocessor(X_train, strategy=strategy)
    num_tf = pre.named_transformers_["num"]
    after = num_tf.transform(X_train[num_cols])

    for i, c in enumerate(num_cols):
        rep.skew_before[c] = float(X_train[c].skew())
        rep.skew_after[c] = float(pd.Series(after[:, i]).skew())
        rep.negatives[c] = int((X_train[c] < 0).sum())
    rep.max_abs_z = float(np.abs(after).max())
    rep.pct_beyond_5sd = float(100 * (np.abs(after) > 5).mean())
    return rep


def transform_splits(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    strategy: Strategy = "phase1",
    categorical: Literal["onehot", "ordinal"] = "onehot",
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, object]:
    """Fit on train, transform all three partitions, return float32 arrays.

    Returns ``(X_train_arr, X_val_arr, X_test_arr, fitted_preprocessor)``.
    """
    pre = build_nn_preprocessor(X_train, strategy=strategy, categorical=categorical, seed=seed)
    out = tuple(np.asarray(pre.transform(df), dtype=np.float32) for df in (X_train, X_val, X_test))
    return (*out, pre)


def conditioning_table(
    X_train: pd.DataFrame, X_val: pd.DataFrame, strategies=None, seed: int = SEED
) -> pd.DataFrame:
    """Compare how well each strategy conditions the validation inputs.

    A neural network saturates on inputs far outside roughly [-5, 5], so the share
    of validation values beyond that, and the worst single value, are the numbers
    that decide suitability. Fitted on train, measured on validation.
    """
    strategies = strategies or ["phase1", "robust", "yeo-johnson", "quantile"]
    num_cols = _numeric_columns(X_train.columns)
    rows = []
    for s in strategies:
        pre = build_nn_preprocessor(X_train, strategy=s, seed=seed)
        z = pre.named_transformers_["num"].transform(X_val[num_cols])
        rows.append(
            {
                "strategy": s,
                "pct_beyond_5sd": round(100 * float((np.abs(z) > 5).mean()), 4),
                "pct_beyond_10sd": round(100 * float((np.abs(z) > 10).mean()), 4),
                "max_abs_value": round(float(np.abs(z).max()), 2),
                "mean_abs_skew": round(float(np.abs(pd.DataFrame(z).skew()).mean()), 3),
            }
        )
    return pd.DataFrame(rows)


def compare_on_mlp(
    strategies=None,
    train_rows: int = 150_000,
    epochs: int = 6,
    hidden=(128, 64),
    seed: int = SEED,
    verbose: bool = True,
) -> pd.DataFrame:
    """Train an identical small MLP under each strategy and score on validation.

    The work plan requires the Phase 1 pipeline to remain the default unless a new
    one is *clearly* better, so conditioning statistics alone cannot decide this --
    they describe the inputs, not the result. Everything except the preprocessing
    is held fixed: same subsample, same architecture, same seed, same epochs, same
    shared ``evaluate()``.

    Deliberately cheap. This selects a preprocessing strategy; it is not a model
    result and nothing here goes in the results table.
    """
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    from src.dl.seeds import set_seed
    from src.evaluate_api import evaluate, get_splits

    strategies = strategies or ["phase1", "robust", "yeo-johnson", "quantile"]
    X_train, X_val, X_test, y_train, y_val, _ = get_splits(
        include_metadata=True, scale_numeric=False
    )
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(X_train), min(train_rows, len(X_train)), replace=False))
    X_train, y_train = X_train.iloc[idx], y_train.iloc[idx]

    # The target is standardised with training statistics only, and predictions are
    # mapped back before scoring, so every strategy is judged on the original scale.
    y_mu, y_sd = float(y_train.mean()), float(y_train.std() or 1.0)
    y_tr = ((y_train.to_numpy() - y_mu) / y_sd).astype(np.float32)
    y_va_true = y_val.to_numpy()

    rows = []
    for strat in strategies:
        set_seed(seed)
        Xtr_a, Xva_a, _, _ = transform_splits(X_train, X_val, X_test, strategy=strat, seed=seed)

        model = nn.Sequential(
            nn.Linear(Xtr_a.shape[1], hidden[0]),
            nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
            nn.Linear(hidden[1], 1),
        )
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_fn = nn.MSELoss()
        loader = DataLoader(
            TensorDataset(torch.from_numpy(Xtr_a), torch.from_numpy(y_tr).unsqueeze(1)),
            batch_size=256,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
        )

        model.train()
        for _ in range(epochs):
            for xb, yb in loader:
                opt.zero_grad()
                loss_fn(model(xb), yb).backward()
                opt.step()

        model.eval()
        with torch.no_grad():
            pred = model(torch.from_numpy(Xva_a)).squeeze(1).numpy() * y_sd + y_mu
        m = evaluate(y_va_true, pred)
        rows.append(
            {
                "strategy": strat,
                "n_features": Xtr_a.shape[1],
                "val_mae": round(m["mae"], 4),
                "val_rmse": round(m["rmse"], 4),
                "val_r2": round(m["r2"], 5),
            }
        )
        if verbose:
            print(f"  {strat:14} MAE {m['mae']:7.4f}  RMSE {m['rmse']:7.4f}  R2 {m['r2']:.5f}")

    out = pd.DataFrame(rows).sort_values("val_mae").reset_index(drop=True)
    base = out.loc[out.strategy == "phase1", "val_mae"].iloc[0]
    out["mae_vs_phase1_pct"] = ((out.val_mae / base - 1) * 100).round(2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", action="store_true", help="print the evidence tables and exit")
    ap.add_argument("--subsample", type=int, default=300_000)
    ap.add_argument("--compare", action="store_true", help="quick MLP comparison of strategies")
    args = ap.parse_args()

    from src.evaluate_api import get_splits

    X_train, X_val, _, _, _, _ = get_splits(include_metadata=True, scale_numeric=False)
    if args.subsample and args.subsample < len(X_train):
        rng = np.random.default_rng(SEED)
        idx = np.sort(rng.choice(len(X_train), args.subsample, replace=False))
        X_train = X_train.iloc[idx]

    print(f"train rows used: {len(X_train):,}   val rows: {len(X_val):,}\n")
    print("Skew before and after Yeo-Johnson, with the negatives that rule out log1p:")
    print(skew_report(X_train, "yeo-johnson").to_frame().to_string())
    print("\nHow well each strategy conditions the validation inputs:")
    print(conditioning_table(X_train, X_val).to_string(index=False))

    if args.compare:
        print("\nQuick MLP comparison (identical model, seed and budget):")
        print(compare_on_mlp().to_string(index=False))


if __name__ == "__main__":
    main()
