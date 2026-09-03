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
from scipy.stats import norm

from project.calibration.ou_process import estimate_ou_parameters_for_factors
from project.calibration.pca import fit_pca, transform_pca
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


def _fit_pipeline_at_origin(train_df: pd.DataFrame, tenors, n_pca_factors=DEFAULT_N_PCA_FACTORS):
    """Fit NS (lambda fixed) -> PCA -> OU using only `train_df`, and assemble
    an in-memory HJMModel -- the walk-forward analogue of `HJMModel.from_disk`,
    which instead re-derives every artifact from scratch on the caller's
    window rather than reading files fit once on the *entire* sample.
    Mirrors the notebook's Sections 4-7 exactly, just scoped to `train_df`.

    Also returns the origin day's own PC scores (`train_df`'s last row,
    projected onto this fit's PCA basis via `transform_pca`) -- the correct
    starting point for a forecast simulated *from* this origin. Without it,
    `HJMModel.simulate()` defaults to starting every path at PC score 0, i.e.
    `mean_params`, the training window's *average* curve rather than its most
    recent one; those two are the same only by coincidence, and during a
    trending period (a hiking or cutting cycle) they can be far apart -- see
    TODO.md for how badly that mismatch showed up in this backtest's first
    version.

    Returns (HJMModel, origin_alpha) where `origin_alpha` has shape
    (n_pca_factors,).
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

    origin = train_df.index[-1]
    origin_row = ns_params_df.loc[[origin], ["b0_level", "b1_slope", "b2_curvature"]]
    origin_alpha = transform_pca(pca_model, origin_row).values[0]

    return HJMModel(params), origin_alpha


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
    measure, starting from that origin's own actual PC scores (not the
    training window's average curve -- see `_fit_pipeline_at_origin`). Each
    observed tenor's simulated terminal distribution is
    compared against what that tenor's yield actually did, at the nearest
    available maturity on the model's own (finer) grid -- the same
    nearest-maturity matching `stochastic.validation.compare_simulated_to_historical`
    already uses, for the same reason: the model's simulation grid and the
    raw data's observed tenors don't line up exactly.

    Every row also carries a naive random-walk ("no change") forecast --
    tomorrow's curve equals today's -- scored the same way, since a model's
    RMSE or CRPS means little without a reference point: `naive_forecast` is
    just that tenor's rate on the origin date itself, `naive_squared_error`
    and `naive_crps` its errors against the same realized value (a
    single-point forecast's CRPS is its absolute error). `summarize_backtest`
    turns this into a skill score.

    Returns columns: origin, realized_date, tenor, realized, simulated_median,
    simulated_mean, band_lo, band_hi, covered, squared_error, crps,
    naive_forecast, naive_squared_error, naive_crps.
    """
    if origins is None:
        origins = generate_backtest_origins(yields_df.index, horizon=horizon_days)

    tenors = np.array([float(c) for c in yields_df.columns])
    lo_q, hi_q = (1 - band) / 2, 1 - (1 - band) / 2

    rows = []
    for i, origin in enumerate(origins):
        train_df = yields_df.loc[:origin]
        model, origin_alpha = _fit_pipeline_at_origin(train_df, tenors, n_pca_factors=n_pca_factors)

        seed = None if random_seed is None else random_seed + i
        result = model.simulate(
            n_paths=n_paths,
            T_horizon=horizon_days / 252,
            dt=1 / 252,
            measure="P",
            random_seed=seed,
            initial_alpha=origin_alpha,
        )
        terminal = result.zero_curves[:, -1, :]  # (n_paths, n_maturities)

        realized_pos = yields_df.index.get_loc(origin) + horizon_days
        realized_date = yields_df.index[realized_pos]
        realized_row = yields_df.iloc[realized_pos]
        origin_row = yields_df.loc[origin]

        for tenor_col in yields_df.columns:
            T = float(tenor_col)
            nearest_idx = int(np.argmin(np.abs(result.maturities - T)))
            samples = terminal[:, nearest_idx]
            realized = float(realized_row[tenor_col])
            median_sim = float(np.median(samples))
            lo, hi = (float(q) for q in np.quantile(samples, [lo_q, hi_q]))
            naive_forecast = float(origin_row[tenor_col])

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
                    "naive_forecast": naive_forecast,
                    "naive_squared_error": (naive_forecast - realized) ** 2,
                    "naive_crps": abs(naive_forecast - realized),
                }
            )

    return pd.DataFrame(rows)


def _wilson_interval(successes: int, n: int, confidence=0.95) -> tuple:
    """Wilson score interval for a binomial proportion -- better-behaved
    than the naive normal approximation at small n or proportions near 0/1,
    both of which apply here (a few dozen origins, and coverage that should
    sit near 90%). Treats each row as an independent Bernoulli trial, which
    rows from *overlapping* backtest windows (step < horizon) aren't
    strictly -- see `run_backtest`'s docstring and TODO.md; this interval is
    therefore a lower bound on the true uncertainty, not the final word.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    z = norm.ppf(1 - (1 - confidence) / 2)
    phat = successes / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    margin = (z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def _summarize_group(df: pd.DataFrame, nominal_band=0.90, ci_confidence=0.95) -> dict:
    n_obs = len(df)
    rmse = float(np.sqrt(df["squared_error"].mean()))
    naive_rmse = float(np.sqrt(df["naive_squared_error"].mean()))
    coverage = float(df["covered"].mean())
    ci_lo, ci_hi = _wilson_interval(int(df["covered"].sum()), n_obs, confidence=ci_confidence)
    return {
        "n_origins": int(df["origin"].nunique()),
        "n_obs": n_obs,
        "rmse": rmse,
        "naive_rmse": naive_rmse,
        "skill_vs_naive": (1 - rmse / naive_rmse) if naive_rmse > 0 else float("nan"),
        "coverage": coverage,
        "coverage_ci_lo": ci_lo,
        "coverage_ci_hi": ci_hi,
        "coverage_gap_vs_nominal": coverage - nominal_band,
        "mean_crps": float(df["crps"].mean()),
        "naive_mean_crps": float(df["naive_crps"].mean()),
    }


def summarize_backtest(
    results_df: pd.DataFrame, nominal_band=0.90, ci_confidence=0.95
) -> pd.DataFrame:
    """Aggregate per-(origin, tenor) backtest rows into per-tenor statistics:
    RMSE of the simulated median against the realized rate (and against a
    naive random-walk forecast, as `skill_vs_naive` -- the fraction of RMSE
    the model removes relative to "no change"; 0 means no better than naive,
    negative means worse), empirical coverage of the simulated band with a
    Wilson confidence interval around it (`_wilson_interval`), mean CRPS
    (model and naive), and how many origins contributed.
    """
    return pd.DataFrame(
        {
            tenor: _summarize_group(group, nominal_band=nominal_band, ci_confidence=ci_confidence)
            for tenor, group in results_df.groupby("tenor")
        }
    ).T


def summarize_backtest_overall(
    results_df: pd.DataFrame, nominal_band=0.90, ci_confidence=0.95
) -> dict:
    """Like `summarize_backtest`, but pools every (origin, tenor) row into
    one summary instead of breaking out by tenor. Tenors differ in how hard
    they are to forecast (short-end noise vs. long-end drift), so
    `summarize_backtest`'s per-tenor breakdown remains the more diagnostic
    view for a single horizon; this pooled version exists for comparing one
    number across *horizons* (`run_backtest_across_horizons`), where a full
    per-tenor table at every horizon would be too much to take in at once.
    """
    return _summarize_group(results_df, nominal_band=nominal_band, ci_confidence=ci_confidence)


def run_backtest_across_horizons(
    yields_df: pd.DataFrame,
    horizons=(21, 63, 126, 252),
    start=None,
    end=None,
    n_paths=300,
    band=0.90,
    n_pca_factors=DEFAULT_N_PCA_FACTORS,
    step=DEFAULT_STEP_DAYS,
    random_seed=0,
) -> pd.DataFrame:
    """Run the walk-forward backtest separately at each horizon in
    `horizons` and return one pooled-across-tenors summary row per horizon,
    indexed by `horizon_days`.

    A forecast's accuracy should degrade as the horizon grows -- more time
    for the curve to move away from wherever it started -- and how quickly
    it degrades is itself informative (a model that's only a little better
    than naive at 21 days but falls to naive's level by 252 days is telling
    you something different than one that stays ahead throughout). `start`/
    `end` restrict eligible origins the same way `generate_backtest_origins`
    does -- e.g. pass `registry.backtest_spec.TEST_START` as `start` to keep
    this on the held-out test region.
    """
    rows = {}
    for h in horizons:
        origins = generate_backtest_origins(
            yields_df.index, start=start, end=end, step=step, horizon=h
        )
        if not origins:
            continue
        results = run_backtest(
            yields_df,
            origins=origins,
            horizon_days=h,
            n_paths=n_paths,
            band=band,
            n_pca_factors=n_pca_factors,
            random_seed=random_seed,
        )
        rows[h] = summarize_backtest_overall(results, nominal_band=band)

    summary = pd.DataFrame(rows).T
    summary.index.name = "horizon_days"
    return summary
