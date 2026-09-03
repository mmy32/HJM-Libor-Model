"""FRED series definitions for the raw Treasury data loader."""

TREASURY_SYMBOL_MAP = {
    "DGS1MO": 0.0833,
    "DGS3MO": 0.25,
    "DGS6MO": 0.5,
    "DGS1": 1.0,
    "DGS2": 2.0,
    "DGS3": 3.0,
    "DGS5": 5.0,
    "DGS7": 7.0,
    "DGS10": 10.0,
    "DGS20": 20.0,
    "DGS30": 30.0,
}

DEFAULT_START_DATE = "2000-01-01"
# The effective start of the *cleaned* panel is later than this: DGS1MO (the
# shortest tenor) has no FRED data before 2001-07-31, and `drop_incomplete_rows`
# correctly drops any row that's still missing a tenor after forward-filling
# rather than fabricating a value with nothing to forward-fill from. So the
# cleaned matrix in practice starts ~2001-07-31, not 2000-01-01 -- see
# `clean_treasury_yields`.
