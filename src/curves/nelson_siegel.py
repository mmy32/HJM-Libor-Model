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

from src.registry.curve_spec import NS_OPTIMIZER_SEED, NS_PARAM_BOUNDS


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


def fit_ns_robust(yields, tenors, bounds=None, seed=None):
    """Fit NS yield-curve parameters for a single day via global optimization.

    Returns np.array([b0, b1, b2, lam]).
    """
    yields = np.asarray(yields, dtype=float)
    tenors = np.asarray(tenors, dtype=float)
    bounds = bounds if bounds is not None else NS_PARAM_BOUNDS
    seed = NS_OPTIMIZER_SEED if seed is None else seed

    def objective(params):
        return np.sum((yields - nelson_siegel_yield(tenors, *params)) ** 2)

    result = differential_evolution(objective, bounds, seed=seed)
    return result.x


def calibrate_all_days(df, tenors, bounds=None, seed=None, progress=False):
    """Fit NS parameters for every row (date) in df.

    Returns a DataFrame indexed by date with columns
    [b0_level, b1_slope, b2_curvature, lambda].
    """
    rows = df.iterrows()
    if progress:
        from tqdm.auto import tqdm

        rows = tqdm(rows, total=len(df), desc="Calibrating NS curves")

    records, dates = [], []
    for date, row in rows:
        records.append(fit_ns_robust(row.values, tenors, bounds=bounds, seed=seed))
        dates.append(date)

    return pd.DataFrame(
        records, index=dates, columns=["b0_level", "b1_slope", "b2_curvature", "lambda"]
    )
