"""Ornstein-Uhlenbeck parameter estimation for mean-reverting PCA factors."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def estimate_ou_parameters(time_series, dt=1 / 252):
    """MLE fit of dX = kappa(theta - X)dt + sigma dW to discrete observations.

    Returns a dict with keys: kappa (mean reversion speed), theta (long-run
    mean), sigma (volatility), half_life_days, log_likelihood, converged.
    """
    X = np.asarray(time_series, dtype=float)
    n = len(X)

    def neg_log_likelihood(params):
        kappa, theta, sigma = params
        if kappa <= 0 or sigma <= 0:
            return 1e10

        exp_term = np.exp(-kappa * dt)
        mu_cond = theta + (X[:-1] - theta) * exp_term
        var_cond = (sigma**2 / (2 * kappa)) * (1 - np.exp(-2 * kappa * dt))
        if var_cond <= 0:
            return 1e10

        residuals = X[1:] - mu_cond
        ll = -0.5 * n * np.log(2 * np.pi * var_cond) - 0.5 * np.sum(residuals**2) / var_cond
        return -ll

    empirical_mean = np.mean(X)
    empirical_std = np.std(X)
    if len(X) > 1:
        acf_1 = np.corrcoef(X[:-1], X[1:])[0, 1]
        initial_kappa = -np.log(max(acf_1, 0.01)) / dt
    else:
        initial_kappa = 0.1

    result = minimize(
        neg_log_likelihood,
        [initial_kappa, empirical_mean, empirical_std],
        method="L-BFGS-B",
        bounds=[(1e-6, 50), (None, None), (1e-6, None)],
    )

    kappa, theta, sigma = result.x
    half_life = np.log(2) / kappa

    return {
        "kappa": kappa,
        "theta": theta,
        "sigma": sigma,
        "half_life_days": half_life * 252,
        "log_likelihood": -result.fun,
        "converged": result.success,
    }


def estimate_ou_parameters_for_factors(scores_df, dt=1 / 252):
    """Apply estimate_ou_parameters independently to each column of a PC-scores DataFrame."""
    return {col: estimate_ou_parameters(scores_df[col].values, dt=dt) for col in scores_df.columns}


def estimate_ou_parameters_rolling(time_series, dt=1 / 252, window=252, step=21):
    """Fit OU MLE independently on successive rolling windows instead of once
    over the whole series.

    A single stationary OU fit over a multi-year sample that actually spans
    different rate regimes (e.g. 2018's near-zero rates through the 2022-23
    hiking cycle to a later plateau) doesn't have a single well-defined
    "true" kappa to recover -- and on this project's real data, that single
    full-sample fit pins kappa at its optimizer bound for some factors
    (implausibly fast mean-reversion). A day-to-day *smoothing* penalty on
    the upstream NS fit was tried as a fix and rejected (see TODO.md): it
    manufactures serial correlation that mechanically biases the estimated
    kappa downward regardless of the true dynamics, so it just moves the
    pinning to the other bound. Rolling windows sidestep that confound
    entirely -- no smoothing is applied, each window's MLE is fit on
    unmodified data, short enough that within-window regime is closer to
    homogeneous.

    Returns a DataFrame indexed by each window's end position (integer
    position in `time_series`, or the corresponding index label if
    `time_series` is a pandas Series) with columns [kappa, theta, sigma,
    half_life_days, log_likelihood, converged].
    """
    is_series = isinstance(time_series, pd.Series)
    index = time_series.index if is_series else None
    X = np.asarray(time_series, dtype=float)

    records, positions = [], []
    for end in range(window, len(X) + 1, step):
        records.append(estimate_ou_parameters(X[end - window : end], dt=dt))
        positions.append(end - 1)

    result = pd.DataFrame(records)
    result.index = index[positions] if is_series else positions
    return result


def estimate_ou_parameters_for_factors_rolling(scores_df, dt=1 / 252, window=252, step=21):
    """Apply estimate_ou_parameters_rolling independently to each column of a PC-scores DataFrame.

    Returns {factor_name: rolling-fit DataFrame}.
    """
    return {
        col: estimate_ou_parameters_rolling(scores_df[col], dt=dt, window=window, step=step)
        for col in scores_df.columns
    }


def most_recent_ou_parameters(scores_df, dt=1 / 252, window=252, step=21) -> dict:
    """The single most recent rolling-window OU fit per factor -- i.e.
    calibrate to how each factor has actually been behaving recently rather
    than averaging over its entire, possibly regime-spanning, history.

    Returns the same {factor_name: {"kappa", "theta", "sigma", ...}} shape
    as `estimate_ou_parameters_for_factors`, so it's a drop-in alternative
    wherever that's used (e.g. before persisting via
    `persistence.artifacts.save_ou_parameters`).
    """
    rolling = estimate_ou_parameters_for_factors_rolling(scores_df, dt=dt, window=window, step=step)
    return {name: fits.iloc[-1].to_dict() for name, fits in rolling.items()}
