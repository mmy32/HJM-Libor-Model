import numpy as np

from src.calibration.ou_process import estimate_ou_parameters


def test_estimate_ou_parameters_recovers_known_process():
    rng = np.random.default_rng(42)
    kappa, theta, sigma, dt = 5.0, 0.02, 0.01, 1 / 252
    n = 5000
    X = np.zeros(n)
    X[0] = theta
    for t in range(1, n):
        exp_term = np.exp(-kappa * dt)
        mean = theta + (X[t - 1] - theta) * exp_term
        var = (sigma**2 / (2 * kappa)) * (1 - np.exp(-2 * kappa * dt))
        X[t] = mean + np.sqrt(var) * rng.standard_normal()

    fitted = estimate_ou_parameters(X, dt=dt)
    assert np.isclose(fitted["kappa"], kappa, rtol=0.3)
    assert np.isclose(fitted["theta"], theta, atol=0.005)
    assert np.isclose(fitted["sigma"], sigma, rtol=0.2)
