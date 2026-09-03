"""Canonical Nelson-Siegel yield and forward curve math.

The decay parameter is always named `lam` here (never `tau`, which is
reserved for the maturity argument). Earlier versions of this project had
three independent forward-rate implementations that silently disagreed with
each other because of this naming collision: `nelson_siegel_forward` below is
derived so that `nelson_siegel_yield` is exactly its running average
(y(T) = (1/T) * integral_0^T f(s) ds), which is the standard Nelson-Siegel
relationship and the one the project's actual curve-fitting bounds are
calibrated against.
"""

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from project.registry.curve_spec import NS_OPTIMIZER_SEED, NS_PARAM_BOUNDS


def nelson_siegel_yield(tau, b0, b1, b2, lam):
    """Nelson-Siegel zero-coupon yield curve y(tau)."""
    tau = np.asarray(tau, dtype=float)
    tau = np.where(tau == 0, 1e-6, tau)
    decay = (1 - np.exp(-lam * tau)) / (lam * tau)
    return b0 + b1 * decay + b2 * (decay - np.exp(-lam * tau))


def nelson_siegel_forward(tau, b0, b1, b2, lam):
    """Nelson-Siegel instantaneous forward curve f(tau).

    f(tau) = b0 + b1*exp(-lam*tau) + b2*lam*tau*exp(-lam*tau)
    """
    tau = np.asarray(tau, dtype=float)
    exp_term = np.exp(-lam * tau)
    return b0 + b1 * exp_term + b2 * lam * tau * exp_term


def fit_ns_robust(yields, tenors, bounds=None, seed=None, prev_params=None, smoothing_weight=0.0):
    """Fit NS yield-curve parameters for a single day via global optimization.

    `prev_params`/`smoothing_weight` add an optional penalty term for
    day-over-day parameter drift, normalized by each parameter's bound
    width so b0/b1/b2 (narrow bounds) and lambda (wide bounds) are
    penalized comparably. See `calibrate_all_days` for why this exists.

    Returns np.array([b0, b1, b2, lam]).
    """
    yields = np.asarray(yields, dtype=float)
    tenors = np.asarray(tenors, dtype=float)
    bounds = bounds if bounds is not None else NS_PARAM_BOUNDS
    seed = NS_OPTIMIZER_SEED if seed is None else seed
    smooth = prev_params is not None and smoothing_weight > 0
    if smooth:
        prev_params = np.asarray(prev_params, dtype=float)
        param_range = np.array([hi - lo for lo, hi in bounds])

    def objective(params):
        residual = np.sum((yields - nelson_siegel_yield(tenors, *params)) ** 2)
        if smooth:
            drift = (np.asarray(params) - prev_params) / param_range
            residual += smoothing_weight * np.sum(drift**2)
        return residual

    result = differential_evolution(objective, bounds, seed=seed)
    return result.x


def fit_ns_fixed_lambda(yields, tenors, lam):
    """Fit NS *level/slope/curvature* parameters for a single day by ordinary
    least squares, with `lam` (decay) held fixed rather than freely optimized.

    `nelson_siegel_yield` is linear in b0/b1/b2 for a given lam and tenor
    grid -- fixing lam turns the fit from a 4-parameter global (stochastic)
    optimization into an exact, deterministic 3-parameter linear regression.
    This is the classical Diebold & Li (2006) "Dynamic Nelson-Siegel"
    approach, motivated by the same weak-identification problem
    `calibrate_all_days`'s `smoothing_weight` was trying (and failing, see
    TODO.md) to patch after the fact: lam is poorly identified from a
    handful of tenor quotes via a global optimizer, so letting it float
    freely day to day introduces optimizer-driven noise that swamps genuine
    curve dynamics and shows up downstream as implausibly fast OU
    mean-reversion. Fixing lam removes that noise source at its origin
    instead of correcting for it afterward.

    Returns np.array([b0, b1, b2, lam]) (lam echoed back for a uniform
    return shape with `fit_ns_robust`).
    """
    yields = np.asarray(yields, dtype=float)
    tenors = np.asarray(tenors, dtype=float)
    tenors_safe = np.where(tenors == 0, 1e-6, tenors)
    decay = (1 - np.exp(-lam * tenors_safe)) / (lam * tenors_safe)
    exp_term = np.exp(-lam * tenors_safe)

    design = np.column_stack([np.ones_like(tenors_safe), decay, decay - exp_term])
    b0, b1, b2 = np.linalg.lstsq(design, yields, rcond=None)[0]
    return np.array([b0, b1, b2, lam])


def calibrate_all_days_fixed_lambda(df, tenors, lam=None, seed=None):
    """Fit NS parameters for every row (date) in df with lam fixed across
    all days (see `fit_ns_fixed_lambda`).

    If `lam` isn't given, it's estimated once via a single free 4-parameter
    `fit_ns_robust` on the panel's across-day average yield curve, then held
    fixed for every day's OLS fit -- i.e. lam is chosen to fit the panel's
    typical curve shape, not any individual day.

    Returns a DataFrame indexed by date with columns
    [b0_level, b1_slope, b2_curvature, lambda]. Unlike `calibrate_all_days`,
    the `lambda` column here is constant by construction, so a downstream
    PCA fit should drop it (zero variance) and use 3 factors, not 4 -- see
    TODO.md.
    """
    tenors = np.asarray(tenors, dtype=float)
    if lam is None:
        average_yields = df.mean(axis=0).values
        lam = fit_ns_robust(average_yields, tenors, seed=seed)[3]

    records = [fit_ns_fixed_lambda(row.values, tenors, lam) for _, row in df.iterrows()]
    return pd.DataFrame(
        records, index=df.index, columns=["b0_level", "b1_slope", "b2_curvature", "lambda"]
    )


def calibrate_all_days(df, tenors, bounds=None, seed=None, progress=False, smoothing_weight=0.0):
    """Fit NS parameters for every row (date) in df.

    `smoothing_weight` (default 0, i.e. today's exact prior behavior) adds a
    day-over-day continuity penalty to each day's fit, seeded from the
    previous day's accepted parameters. On the project's original 2018-2026
    sample (before its history was extended back to 2001), independent
    per-day fits leave `lambda` (the decay parameter) weakly
    identified day to day -- differential_evolution finds similarly-good
    optima that whipsaw more than genuine economic dynamics would, which
    downstream shows up as OU calibration pinning `kappa` at its optimizer
    bound (an implausibly fast, ~3.5-day half-life) for the PCA factors that
    load heavily on lambda. A modest smoothing_weight (e.g. 0.01-0.1)
    discourages that whipsaw without materially degrading same-day fit
    quality. Left opt-in rather than a new default so changing already
    -persisted calibrated values remains an explicit, reviewable choice.

    Returns a DataFrame indexed by date with columns
    [b0_level, b1_slope, b2_curvature, lambda].
    """
    rows = df.iterrows()
    if progress:
        from tqdm.auto import tqdm

        rows = tqdm(rows, total=len(df), desc="Calibrating NS curves")

    records, dates = [], []
    prev_params = None
    for date, row in rows:
        params = fit_ns_robust(
            row.values,
            tenors,
            bounds=bounds,
            seed=seed,
            prev_params=prev_params,
            smoothing_weight=smoothing_weight,
        )
        records.append(params)
        dates.append(date)
        prev_params = params

    return pd.DataFrame(
        records, index=dates, columns=["b0_level", "b1_slope", "b2_curvature", "lambda"]
    )
