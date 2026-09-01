"""Cleaning policy for raw Treasury yield panels."""
import pandas as pd


def to_decimal(df: pd.DataFrame) -> pd.DataFrame:
    """Convert whole-percentage yields (e.g. 4.5) to decimal form (0.045)."""
    return df / 100.0


def drop_incomplete_rows(df: pd.DataFrame):
    """Forward-fill short gaps, then drop any rows still missing a tenor.

    Forward-filling assumes an unobserved quote is unchanged from the prior
    observation, which avoids introducing interpolation noise; any row still
    incomplete after filling (e.g. a leading gap) is dropped. Returns
    (cleaned_df, stats).
    """
    filled = df.ffill()
    cleaned = filled.dropna()
    stats = {
        "rows_before": len(df),
        "rows_after": len(cleaned),
        "rows_dropped": len(df) - len(cleaned),
    }
    return cleaned, stats


def clean_treasury_yields(df: pd.DataFrame) -> pd.DataFrame:
    """Standard cleaning pipeline: percentage -> decimal, forward-fill, drop remaining gaps."""
    cleaned, _ = drop_incomplete_rows(to_decimal(df))
    return cleaned
