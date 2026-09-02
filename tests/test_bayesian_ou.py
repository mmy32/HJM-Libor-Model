import numpy as np
import pytest

from project.calibration.bayesian_ou import fit_bayesian_ou, posterior_draws, posterior_summary


def _simulate_ou(kappa, theta, sigma, dt, n, seed):
    rng = np.random.default_rng(seed)
    X = np.zeros(n)
    X[0] = theta
    for t in range(1, n):
        mean = theta + (X[t - 1] - theta) * np.exp(-kappa * dt)
        var = (sigma**2 / (2 * kappa)) * (1 - np.exp(-2 * kappa * dt))
        X[t] = mean + np.sqrt(var) * rng.standard_normal()
    return X


@pytest.fixture(scope="module")
def known_process_fit():
    """One real MCMC fit, shared across tests -- sampling is the slow part
    (~10s even with modest draws), so multiple assertions run against a
    single fit rather than re-fitting per test."""
    kappa, theta, sigma, dt = 0.9, 0.02, 0.01, 1 / 252
    X = _simulate_ou(kappa, theta, sigma, dt, n=1500, seed=42)
    idata = fit_bayesian_ou(X, dt=dt, draws=500, tune=500, chains=2, random_seed=42)
    return {"true": {"kappa": kappa, "theta": theta, "sigma": sigma}, "idata": idata}


def test_posterior_mean_is_close_to_true_theta_and_sigma(known_process_fit):
    """kappa is deliberately not checked here -- see the docstring on
    test_credible_interval_coverage_across_replications for why a single
    fixed-seed realization is the wrong place to check it."""
    summary = posterior_summary(known_process_fit["idata"])
    true = known_process_fit["true"]
    assert np.isclose(summary["theta"], true["theta"], atol=0.01)
    assert np.isclose(summary["sigma"], true["sigma"], rtol=0.3)


def test_posterior_hdi_is_well_formed(known_process_fit):
    """Structural check only -- lower < mean < upper for every parameter.
    Deliberately does *not* assert the HDI contains the true value: a 90%
    interval is, by construction, expected to miss the truth on roughly 1
    fit in 10, so testing containment against one fixed seed is a flaky
    test by design, not a correctness check. That's exactly what happened
    while writing this test (a single-seed run put true theta=0.02 just
    outside a [0.013, 0.0196] interval -- unlucky, not wrong). Actual
    coverage is checked properly, across replications, below."""
    summary = posterior_summary(known_process_fit["idata"])
    for param in ("kappa", "theta", "sigma"):
        lo, hi = summary[f"{param}_hdi"]
        assert lo < summary[param] < hi


def test_posterior_summary_reports_convergence_diagnostics(known_process_fit):
    summary = posterior_summary(known_process_fit["idata"])
    for param in ("kappa", "theta", "sigma"):
        assert np.isclose(summary[f"{param}_r_hat"], 1.0, atol=0.05)
        assert summary[f"{param}_ess_bulk"] > 100


def test_posterior_draws_returns_requested_count(known_process_fit):
    draws = posterior_draws(known_process_fit["idata"], n_draws=50, random_seed=0)
    assert len(draws) == 50
    assert all(set(d.keys()) == {"kappa", "theta", "sigma"} for d in draws)
    assert all(d["kappa"] > 0 and d["sigma"] > 0 for d in draws)


def test_posterior_draws_without_subsampling_returns_all_draws(known_process_fit):
    draws = posterior_draws(known_process_fit["idata"])
    assert len(draws) == 2 * 500  # chains * draws from the fixture


@pytest.mark.slow
def test_credible_interval_coverage_across_replications():
    """A 90% credible interval is, by construction, expected to miss the
    true value on roughly 1 fit in 10 -- so the only honest way to check
    coverage is across several independent replications, not one fixed
    seed (see test_posterior_hdi_is_well_formed's docstring for a concrete
    example of exactly that flakiness). kappa is also the parameter this
    project has already found to be poorly identified from a single path
    (the existing point-estimate MLE recovers kappa=3.37 against a true
    0.9 on one specific realization used elsewhere in this file) -- a wide,
    honestly-reported posterior for it is the point, not a defect.

    Checks all three parameters' 90% HDI hit rate across 5 replications.
    Not a tight statistical guarantee at this replication count (expected
    misses if perfectly calibrated: ~0.5 per parameter), but enough to
    catch an interval that's systematically miscalibrated rather than just
    unlucky once."""
    kappa, theta, sigma, dt = 1.2, 0.0, 0.02, 1 / 252
    true = {"kappa": kappa, "theta": theta, "sigma": sigma}
    hits = {"kappa": 0, "theta": 0, "sigma": 0}
    n_reps = 5
    for seed in range(n_reps):
        X = _simulate_ou(kappa, theta, sigma, dt, n=1000, seed=100 + seed)
        idata = fit_bayesian_ou(X, dt=dt, draws=400, tune=400, chains=2, random_seed=seed)
        summary = posterior_summary(idata)
        for param in hits:
            lo, hi = summary[f"{param}_hdi"]
            if lo <= true[param] <= hi:
                hits[param] += 1

    for param, count in hits.items():
        assert (
            count >= n_reps - 2
        ), f"{param}: only {count}/{n_reps} replications covered the true value"
