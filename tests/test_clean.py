"""Unit tests for the cleaning rules.

These run on small synthetic frames on purpose. CI never has the 305 MiB raw dataset
— pulling it on every push would burn the repository's Git LFS bandwidth quota — so the
tests verify the *rules*, not the data. Correctness on the real file is demonstrated in
notebooks/01_cleaning.ipynb, whose outputs are committed.
"""

import numpy as np
import pandas as pd
import pytest

from src.clean import (
    CATEGORICALS,
    MEASUREMENTS,
    check_detection_limits,
    drop_exact_duplicates,
    flag_impossible_values,
    flag_upstream_imputation,
)


def make_frame(n: int = 20, **overrides) -> pd.DataFrame:
    """A minimal frame with the same columns as the real dataset."""
    rng = np.random.default_rng(0)
    data = {
        "Country": ["England"] * n,
        "Area": [f"SITE-{i}" for i in range(n)],
        "Waterbody Type": ["River"] * n,
        "Date": ["01-01-2020"] * n,
        "CCME_Values": rng.uniform(40, 100, n),
        "CCME_WQI": ["Good"] * n,
    }
    for col in MEASUREMENTS:
        data[col] = rng.uniform(1, 9, n)
    data.update(overrides)
    return pd.DataFrame(data)


class TestImpossibleValues:
    def test_ph_outside_range_is_flagged(self):
        df = make_frame(4, **{"pH (ph units)": [7.0, -1.0, 30.0, 7.5]})
        df, counts = flag_impossible_values(df)
        assert df["flag_ph_out_of_range"].tolist() == [False, True, True, False]
        assert counts["pH outside 0-14"] == 2

    def test_negative_concentration_is_flagged(self):
        df = make_frame(3, **{"Ammonia (mg/l)": [0.5, -0.005, 1.0]})
        df, counts = flag_impossible_values(df)
        assert counts["negative concentration"] == 1

    def test_negative_temperature_is_not_impossible(self):
        """Saline water can be sub-zero, so it is flagged for review only."""
        df = make_frame(2, **{"Temperature (cel)": [-5.26, 10.0]})
        df, _ = flag_impossible_values(df)
        assert df["flag_temp_below_zero"].tolist() == [True, False]
        assert not df["flag_any_impossible"].any()

    def test_implausible_temperature_is_flagged(self):
        df = make_frame(2, **{"Temperature (cel)": [98.0, 10.0]})
        df, _ = flag_impossible_values(df)
        assert df["flag_temp_implausible"].tolist() == [True, False]

    def test_no_rows_are_dropped(self):
        df = make_frame(5, **{"pH (ph units)": [-1.0] * 5})
        out, _ = flag_impossible_values(df)
        assert len(out) == 5, "impossible values must be flagged, never deleted"


class TestDuplicates:
    def test_exact_duplicates_removed_keeping_first(self):
        df = make_frame(3)
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        out, removed = drop_exact_duplicates(df)
        assert removed == 1
        assert len(out) == 3

    def test_near_duplicates_are_kept(self):
        df = make_frame(2)
        near = df.iloc[[0]].copy()
        near.loc[:, "Ammonia (mg/l)"] += 0.001
        out, removed = drop_exact_duplicates(pd.concat([df, near], ignore_index=True))
        assert removed == 0

    def test_derived_columns_do_not_affect_matching(self):
        """Duplicate means identical source data, regardless of flags we derived."""
        df = make_frame(2)
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        df, _ = flag_impossible_values(df)
        _, removed = drop_exact_duplicates(df)
        assert removed == 1


class TestUpstreamImputation:
    def test_constant_fill_is_detected(self):
        n = 100
        filled = [5.0] * 40 + list(np.linspace(1, 9, 60))
        df = make_frame(n, **{"Dissolved Oxygen (mg/l)": filled})
        df, report = flag_upstream_imputation(df)
        assert "Dissolved Oxygen (mg/l)" in report
        info = report["Dissolved Oxygen (mg/l)"]
        assert info["rows"] == 40
        assert info["share"] == pytest.approx(0.40)
        assert df["Dissolved Oxygen (mg/l)__imputed_upstream"].sum() == 40

    def test_median_signature_is_reported(self):
        n = 100
        values = [5.0] * 51 + list(np.linspace(1, 9, 49))
        df = make_frame(n, **{"Nitrate (mg/l)": values})
        _, report = flag_upstream_imputation(df)
        assert report["Nitrate (mg/l)"]["equals_median"] is True

    def test_counts_roll_up(self):
        n = 50
        df = make_frame(n, **{c: [2.0] * n for c in MEASUREMENTS})
        df, report = flag_upstream_imputation(df)
        assert len(report) == len(MEASUREMENTS)
        assert (df["n_imputed_upstream"] == len(MEASUREMENTS)).all()
        assert df["flag_any_imputed_upstream"].all()


class TestDetectionLimits:
    def test_censored_values_are_reported_not_silently_coerced(self):
        df = make_frame(3)
        df["Ammonia (mg/l)"] = ["<0.5", "1.0", "2.0"]
        assert check_detection_limits(df)["Ammonia (mg/l)"] == 1

    def test_numeric_columns_report_nothing(self):
        assert check_detection_limits(make_frame(5)) == {}


def test_categoricals_cover_the_label_column():
    """CCME_WQI must stay categorical: it is excluded as leakage, not modelled."""
    assert "CCME_WQI" in CATEGORICALS
