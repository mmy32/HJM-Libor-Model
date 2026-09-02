import numpy as np
import pandas as pd

from project.data_processing.cleaning import (
    clean_treasury_yields,
    diagnose_yield_quality,
    drop_incomplete_rows,
    to_decimal,
)


def test_to_decimal_converts_percent_to_decimal():
    df = pd.DataFrame({"1.0": [4.5, 5.0]})
    result = to_decimal(df)
    assert np.isclose(result.iloc[0, 0], 0.045)


def test_drop_incomplete_rows_forward_fills_then_drops_leading_gap():
    df = pd.DataFrame({"a": [np.nan, 1.0, np.nan, 2.0], "b": [1.0, 1.0, 1.0, 1.0]})
    cleaned, stats = drop_incomplete_rows(df)
    assert stats["rows_before"] == 4
    assert stats["rows_after"] == 3
    assert list(cleaned["a"]) == [1.0, 1.0, 2.0]


def test_clean_treasury_yields_composes_decimal_and_drop():
    df = pd.DataFrame({"1.0": [np.nan, 4.5]})
    cleaned = clean_treasury_yields(df)
    assert len(cleaned) == 1
    assert np.isclose(cleaned.iloc[0, 0], 0.045)


def test_diagnose_yield_quality_flags_stale_run():
    df = pd.DataFrame({"a": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0]})
    diagnostics = diagnose_yield_quality(df, stale_run_threshold=5)
    assert diagnostics["a"]["max_stale_run"] == 6
    assert diagnostics["a"]["stale_flag"] is True


def test_diagnose_yield_quality_flags_forward_fill_run():
    df = pd.DataFrame({"a": [1.0, np.nan, np.nan, np.nan, 1.1]})
    diagnostics = diagnose_yield_quality(df)
    assert diagnostics["a"]["max_ffill_run"] == 3


def test_diagnose_yield_quality_flags_outlier_jump():
    values = [0.04, 0.041, 0.0405, 0.042, 0.5, 0.0415, 0.0418, 0.0412, 0.0409, 0.0421]
    df = pd.DataFrame({"a": values})
    diagnostics = diagnose_yield_quality(df, jump_z_threshold=3.0)
    assert diagnostics["a"]["n_outliers"] >= 1
