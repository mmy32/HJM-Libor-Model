"""Sanity-check simulated yield curves against the historical distribution
they were calibrated from.

This is a plausibility check, not a formal statistical test: Monte Carlo
paths at a fixed horizon aren't independent historical observations, and a
P-measure simulation is expected to revert toward calibrated OU means that
may differ somewhat from the full historical sample average. It exists to
catch the kind of large, obviously-wrong discrepancy the blow-up bug
produced (median simulated 10Y rate at 35% against a ~4% historical mean),
not to certify the model is well-specified.
"""

import numpy as np
import pandas as pd


def compare_simulated_to_historical(
    sim_result, historical_yields_df, flag_threshold_stds=3.0
) -> pd.DataFrame:
    """Compare each maturity's terminal-horizon simulated rate distribution
    to the historical distribution of realized rates at the nearest
    available historical maturity.

    `historical_yields_df` columns must be tenor-in-years (float-castable),
    e.g. the cleaned raw yield panel from `data_processing.io.load_yield_matrix`.
    Returns a DataFrame indexed by `sim_result.maturities` with
    [historical_mean, historical_std, simulated_mean, simulated_std,
    mean_diff_in_historical_stds, flagged].
    """
    maturities = np.asarray(sim_result.maturities, dtype=float)
    terminal_rates = sim_result.zero_curves[:, -1, :]  # (n_paths, n_maturities)
    sim_mean = terminal_rates.mean(axis=0)
    sim_std = terminal_rates.std(axis=0)

    hist_cols = np.asarray(historical_yields_df.columns, dtype=float)

    rows = []
    for i, T in enumerate(maturities):
        nearest_idx = int(np.argmin(np.abs(hist_cols - T)))
        hist_series = historical_yields_df.iloc[:, nearest_idx]
        hist_mean, hist_std = float(hist_series.mean()), float(hist_series.std())
        diff_in_stds = (sim_mean[i] - hist_mean) / hist_std if hist_std > 0 else np.nan
        rows.append(
            {
                "maturity": T,
                "historical_mean": hist_mean,
                "historical_std": hist_std,
                "simulated_mean": float(sim_mean[i]),
                "simulated_std": float(sim_std[i]),
                "mean_diff_in_historical_stds": diff_in_stds,
            }
        )

    result = pd.DataFrame(rows).set_index("maturity")
    result["flagged"] = result["mean_diff_in_historical_stds"].abs() > flag_threshold_stds
    return result
