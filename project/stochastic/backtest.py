"""Rolling-origin (walk-forward) backtest: the statistic this project didn't
have before -- see TODO.md and the discussion that motivated it. Every other
validation in this codebase (`validation.py`'s plausibility check, PCA's
out-of-sample reconstruction error, the Bayesian credible-interval coverage
test) either isn't out-of-sample, or isn't about forecast accuracy, or both.
This module is: for a series of historical origin dates, it refits the whole
NS -> PCA -> OU -> HJM pipeline using *only* data available up to that date,
simulates forward, and compares the simulated distribution to what the yield
curve actually did next.

The train/validation/test split governing which origins' results are treated
as a reportable number lives in `registry.backtest_spec` -- see that module's
docstring for why the boundary is drawn where it is.
"""

import numpy as np
import pandas as pd

from project.calibration.ou_process import estimate_ou_parameters_for_factors
from project.calibration.pca import fit_pca
from project.calibration.sensitivities import compute_forward_sensitivities
from project.curves.nelson_siegel import calibrate_all_days_fixed_lambda
from project.registry.backtest_spec import (
    DEFAULT_HORIZON_DAYS,
    DEFAULT_N_PCA_FACTORS,
    DEFAULT_STEP_DAYS,
    MIN_TRAIN_WINDOW_DAYS,
)
from project.registry.curve_spec import MATURITY_GRID
from project.registry.factor_spec import NS_PARAM_NAMES
from project.stochastic.hjm_model import HJMModel, HJMModelParams


def _fit_pipeline_at_origin(
    train_df: pd.DataFrame, tenors, n_pca_factors=DEFAULT_N_PCA_FACTORS
) -> HJMModel:
    """Fit NS (lambda fixed) -> PCA -> OU using only `train_df`, and assemble
    an in-memory HJMModel -- the walk-forward analogue of `HJMModel.from_disk`,
    which instead re-derives every artifact from scratch on the caller's
    window rather than reading files fit once on the *entire* sample.
    Mirrors the notebook's Sections 4-7 exactly, just scoped to `train_df`.
    """
    ns_params_df = calibrate_all_days_fixed_lambda(train_df, tenors)
    pca_model = fit_pca(
        ns_params_df[["b0_level", "b1_slope", "b2_curvature"]], n_components=n_pca_factors
    )
    ou_params = estimate_ou_parameters_for_factors(pca_model.scores)
    mean_params = ns_params_df.mean().to_dict()
    pc_sens_df = compute_forward_sensitivities(
        mean_params, pca_model.loadings, MATURITY_GRID, param_scale=pca_model.scaler.scale_
    )

    loadings = pca_model.loadings.reindex(NS_PARAM_NAMES, fill_value=0.0).values
    param_scale = (
        pd.Series(pca_model.scaler.scale_, index=pca_model.loadings.index)
        .reindex(NS_PARAM_NAMES, fill_value=0.0)
        .values
    )

    params = HJMModelParams(
        ou_params=ou_params,
        loadings=loadings,
        pc_sensitivities=pc_sens_df.values,
        mean_params=mean_params,
        maturities=MATURITY_GRID,
        factor_names=list(ou_params.keys()),
        param_scale=param_scale,
    )
    return HJMModel(params)


def generate_backtest_origins(
    date_index,
    start=None,
    end=None,
    min_train_window=MIN_TRAIN_WINDOW_DAYS,
    step=DEFAULT_STEP_DAYS,
    horizon=DEFAULT_HORIZON_DAYS,
) -> list:
    """Walk-forward origin schedule over an ascending `date_index`.

    An origin is only eligible if it has at least `min_train_window`
    observations strictly before it (enough history to identify OU
    mean-reversion) and a realized observation `horizon` trading days after
    it (so its forecast can actually be scored) -- both bounds are about
    what's available in `date_index`, not about `start`/`end`. `start`/`end`
    additionally restrict eligible origins to a calendar span -- e.g. pass
    `registry.backtest_spec.TEST_START` as `start` to draw origins only from
    the held-out test region.
    """
    n = len(date_index)
    first_pos = min_train_window
    last_pos = n - 1 - horizon
    origins = []
    for pos in range(first_pos, last_pos + 1, step):
        candidate = date_index[pos]
        if start is not None and candidate < pd.Timestamp(start):
            continue
        if end is not None and candidate > pd.Timestamp(end):
            continue
        origins.append(candidate)
    return origins


def _crps_from_samples(samples, observation) -> float:
    """Empirical CRPS (continuous ranked probability score) of an ensemble
    forecast against one realized observation -- the energy-score form,
    exact for a finite sample (Gneiting & Raftery, 2007, eq. 21): lower is
    better, and it collapses to absolute error for a single-point ensemble.
    """
    samples = np.asarray(samples, dtype=float)
    term1 = np.mean(np.abs(samples - observation))
    term2 = np.mean(np.abs(samples[:, None] - samples[None, :])) / 2
    return float(term1 - term2)


def run_backtest(
    yields_df: pd.DataFrame,
    origins=None,
    horizon_days=DEFAULT_HORIZON_DAYS,
    n_paths=300,
    band=0.90,
    n_pca_factors=DEFAULT_N_PCA_FACTORS,
    random_seed=0,
) -> pd.DataFrame:
    """Run the walk-forward backtest and return one row per
    (origin, tenor) pair.

    For each origin, the pipeline is refit on `yields_df.loc[:origin]` only
    -- nothing at or after the origin's next trading day is visible to the
    fit -- then simulated `horizon_days` trading days forward under the P
    measure. Each observed tenor's simulated terminal distribution is
    compared against what that tenor's yield actually did, at the nearest
    available maturity on the model's own (finer) grid -- the same
    nearest-maturity matching `stochastic.validation.compare_simulated_to_historical`
    already uses, for the same reason: the model's simulation grid and the
    raw data's observed tenors don't line up exactly.

    Returns columns: origin, realized_date, tenor, realized, simulated_median,
    simulated_mean, band_lo, band_hi, covered, squared_error, crps.
    """
    if origins is None:
        origins = generate_backtest_origins(yields_df.index, horizon=horizon_days)

    tenors = np.array([float(c) for c in yields_df.columns])
    lo_q, hi_q = (1 - band) / 2, 1 - (1 - band) / 2

    rows = []
    for i, origin in enumerate(origins):
        train_df = yields_df.loc[:origin]
        model = _fit_pipeline_at_origin(train_df, tenors, n_pca_factors=n_pca_factors)

        seed = None if random_seed is None else random_seed + i
        result = model.simulate(
            n_paths=n_paths, T_horizon=horizon_days / 252, dt=1 / 252, measure="P", random_seed=seed
        )
        terminal = result.zero_curves[:, -1, :]  # (n_paths, n_maturities)

        realized_pos = yields_df.index.get_loc(origin) + horizon_days
        realized_date = yields_df.index[realized_pos]
        realized_row = yields_df.iloc[realized_pos]

        for tenor_col in yields_df.columns:
            T = float(tenor_col)
            nearest_idx = int(np.argmin(np.abs(result.maturities - T)))
            samples = terminal[:, nearest_idx]
            realized = float(realized_row[tenor_col])
            median_sim = float(np.median(samples))
            lo, hi = (float(q) for q in np.quantile(samples, [lo_q, hi_q]))

            rows.append(
                {
                    "origin": origin,
                    "realized_date": realized_date,
                    "tenor": T,
                    "realized": realized,
                    "simulated_median": median_sim,
                    "simulated_mean": float(samples.mean()),
                    "band_lo": lo,
                    "band_hi": hi,
                    "covered": bool(lo <= realized <= hi),
                    "squared_error": (median_sim - realized) ** 2,
                    "crps": _crps_from_samples(samples, realized),
                }
            )

    return pd.DataFrame(rows)


def summarize_backtest(results_df: pd.DataFrame, nominal_band=0.90) -> pd.DataFrame:
    """Aggregate per-(origin, tenor) backtest rows into per-tenor statistics:
    RMSE of the simulated median against the realized rate, empirical
    coverage of the simulated band against its nominal level (should sit
    near `nominal_band` if the model's uncertainty is well-calibrated, not
    just its point forecast), mean CRPS, and how many origins contributed.
    """
    grouped = results_df.groupby("tenor")
    summary = pd.DataFrame(
        {
            "n_origins": grouped["origin"].nunique(),
            "rmse": grouped["squared_error"].mean() ** 0.5,
            "coverage": grouped["covered"].mean(),
            "mean_crps": grouped["crps"].mean(),
        }
    )
    summary["coverage_gap_vs_nominal"] = summary["coverage"] - nominal_band
    return summary
