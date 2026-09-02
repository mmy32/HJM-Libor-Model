"""Bayesian (MCMC) estimation of OU parameters -- a posterior instead of a
single point estimate.

`ou_process.estimate_ou_parameters` (MLE via L-BFGS-B) reports kappa/theta/
sigma as if they were known exactly -- e.g. "kappa=0.28" -- when in fact
they're estimated from ~2000 daily observations of a process this project
has already found reason to distrust at face value (see TODO.md and the
rejected day-to-day smoothing experiment in this repo's git history: naive
fixes to the point estimate changed the number without ever reporting how
uncertain it was to begin with). This module fits the same discrete-time OU
transition likelihood via MCMC (PyMC/NUTS) instead, producing a posterior
with real, checkable uncertainty rather than one number.

Import-time note: arviz 0.17.1 (the newest version compatible with this
project's Python 3.9) imports `scipy.signal.gaussian`, an alias scipy
removed in 1.13. The shim below restores it before arviz is imported.
Every sampling call in this module uses `cores=1` (single-process,
sequential chains) because that shim is process-local and does not
propagate into the subprocesses PyMC's multi-core sampler would otherwise
spawn -- with cores>1, arviz fails to import inside the worker and the
whole run dies with an opaque EOFError.
"""

import numpy as np
import scipy.signal

if not hasattr(scipy.signal, "gaussian"):
    scipy.signal.gaussian = scipy.signal.windows.gaussian

import arviz as az  # noqa: E402
import pymc as pm  # noqa: E402


def fit_bayesian_ou(time_series, dt=1 / 252, draws=2000, tune=1000, chains=4, random_seed=None):
    """Fit dX = kappa(theta - X)dt + sigma dW via MCMC instead of MLE.

    Priors:
    - kappa ~ Gamma(alpha=2, beta=1): weakly informative over economically
      plausible mean-reversion speeds (median implied half-life ~4 months,
      a long right tail out to multi-year slow reversion), strictly
      positive so it never hits the divide-by-kappa term at exactly 0 --
      unlike the MLE's `[1e-6, 50]` box constraint, there's no hard bound
      for the posterior to pile up against.
    - theta ~ Normal(empirical_mean, 2*empirical_std)
    - sigma ~ HalfNormal(3*empirical_std)
    Both theta/sigma priors are scaled to the input series' own units, so
    the prior is weakly informative regardless of which factor it's fit to
    rather than centered on an assumption borrowed from a different
    dataset's scale.

    Returns an arviz InferenceData with posterior samples for kappa, theta,
    sigma. Use `posterior_summary` for a point summary with credible
    intervals and convergence diagnostics, or `posterior_draws` for the raw
    samples (e.g. to propagate through `HJMModel.simulate_with_parameter_uncertainty`).
    """
    X = np.asarray(time_series, dtype=float)
    empirical_mean = X.mean()
    empirical_std = X.std()

    with pm.Model():
        kappa = pm.Gamma("kappa", alpha=2.0, beta=1.0)
        theta = pm.Normal("theta", mu=empirical_mean, sigma=2 * empirical_std)
        sigma = pm.HalfNormal("sigma", sigma=3 * empirical_std)

        exp_term = pm.math.exp(-kappa * dt)
        mu_cond = theta + (X[:-1] - theta) * exp_term
        var_cond = (sigma**2 / (2 * kappa)) * (1 - pm.math.exp(-2 * kappa * dt))

        pm.Normal("obs", mu=mu_cond, sigma=pm.math.sqrt(var_cond), observed=X[1:])

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=1,
            random_seed=random_seed,
            progressbar=False,
        )
    return idata


def fit_bayesian_ou_for_factors(
    scores_df, dt=1 / 252, draws=2000, tune=1000, chains=4, random_seed=None
) -> dict:
    """Apply fit_bayesian_ou independently to each column of a PC-scores DataFrame.

    Returns {factor_name: InferenceData}.
    """
    return {
        col: fit_bayesian_ou(
            scores_df[col].values,
            dt=dt,
            draws=draws,
            tune=tune,
            chains=chains,
            random_seed=random_seed,
        )
        for col in scores_df.columns
    }


def posterior_summary(idata, dt=1 / 252, hdi_prob=0.9) -> dict:
    """Posterior mean + credible interval per parameter, plus the
    convergence diagnostics (r_hat, effective sample size) that must be
    checked before trusting any of it -- an unconverged chain can still
    report a plausible-looking mean and interval.

    Returns {"kappa", "theta", "sigma", "half_life_days", "<param>_hdi",
    "<param>_r_hat", "<param>_ess_bulk"}.
    """
    summary = az.summary(idata, hdi_prob=hdi_prob)
    hdi = az.hdi(idata, hdi_prob=hdi_prob)

    result = {}
    for param in ("kappa", "theta", "sigma"):
        result[param] = float(summary.loc[param, "mean"])
        result[f"{param}_hdi"] = (
            float(hdi[param].sel(hdi="lower").values),
            float(hdi[param].sel(hdi="higher").values),
        )
        result[f"{param}_r_hat"] = float(summary.loc[param, "r_hat"])
        result[f"{param}_ess_bulk"] = float(summary.loc[param, "ess_bulk"])
    result["half_life_days"] = float(np.log(2) / result["kappa"] * 252)
    return result


def posterior_draws(idata, n_draws=None, random_seed=None) -> list:
    """Flatten posterior samples across chains into a list of
    {"kappa", "theta", "sigma"} dicts, one per draw -- for propagating full
    parameter uncertainty through simulation rather than collapsing to a
    single point summary. `n_draws`, if given, subsamples without
    replacement (running a Monte Carlo simulation batch per posterior draw
    gets expensive fast; a few hundred draws is usually enough to see the
    effect on the predictive interval).
    """
    stacked = idata.posterior.stack(sample=("chain", "draw"))
    kappa = stacked["kappa"].values
    theta = stacked["theta"].values
    sigma = stacked["sigma"].values

    idx = np.arange(len(kappa))
    if n_draws is not None and n_draws < len(idx):
        rng = np.random.default_rng(random_seed)
        idx = rng.choice(len(idx), size=n_draws, replace=False)

    return [
        {"kappa": float(kappa[i]), "theta": float(theta[i]), "sigma": float(sigma[i])} for i in idx
    ]
