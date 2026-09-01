import numpy as np
import pandas as pd

from src.data_processing.cleaning import clean_treasury_yields, drop_incomplete_rows, to_decimal


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
