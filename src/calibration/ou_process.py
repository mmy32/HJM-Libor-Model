"""Ornstein-Uhlenbeck parameter estimation for mean-reverting PCA factors."""
import numpy as np
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
