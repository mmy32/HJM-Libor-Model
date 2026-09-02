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


def diagnose_yield_quality(df: pd.DataFrame, stale_run_threshold=5, jump_z_threshold=5.0) -> dict:
    """Data-quality diagnostics on a (decimal-scale, already forward-filled) yield panel.

    Runs three checks per tenor column:
    - stale runs: consecutive identical quotes (a genuine flat market, or a
      symptom of a source that stopped updating and left the prior value
      forward-filled indefinitely).
    - forward-fill runs: consecutive NaNs in the *pre-fill* frame, i.e. how
      much of `drop_incomplete_rows`'s forward-fill each tenor actually
      relied on.
    - outliers: day-over-day changes whose robust (median-absolute-deviation
      based) z-score exceeds `jump_z_threshold` -- flags likely data errors
      (fat-fingered quotes, unit mismatches) rather than genuine rate moves,
      which cluster far tighter than that. MAD-based rather than a plain
      mean/std z-score because a single large jump otherwise inflates the
      std enough to mask its own z-score.

    Returns a dict keyed by tenor column, each value a dict with
    `max_stale_run`, `max_ffill_run`, `n_outliers`, and `outlier_dates`
    (a list of the flagged dates' index labels).
    """

    def _max_run_length(mask: pd.Series) -> int:
        if not mask.any():
            return 0
        groups = (~mask).cumsum()
        return int(mask.groupby(groups).sum().max())

    diagnostics = {}
    for col in df.columns:
        series = df[col]
        stale_mask = series == series.shift(1)
        max_stale_run = _max_run_length(stale_mask.fillna(False)) + 1 if stale_mask.any() else 0

        nan_mask = series.isna()
        max_ffill_run = _max_run_length(nan_mask)

        deltas = series.diff().dropna()
        median = deltas.median()
        mad = (deltas - median).abs().median()
        outlier_mask = pd.Series(False, index=deltas.index)
        if mad and mad > 0:
            robust_z = 0.6745 * (deltas - median) / mad
            outlier_mask = robust_z.abs() > jump_z_threshold

        diagnostics[col] = {
            "max_stale_run": max_stale_run,
            "stale_flag": max_stale_run >= stale_run_threshold,
            "max_ffill_run": max_ffill_run,
            "n_outliers": int(outlier_mask.sum()),
            "outlier_dates": [str(d) for d in deltas.index[outlier_mask]],
        }
    return diagnostics
