import numpy as np
import pandas as pd

from project.calibration.ou_process import (
    estimate_ou_parameters,
    estimate_ou_parameters_for_factors_rolling,
    estimate_ou_parameters_rolling,
    most_recent_ou_parameters,
)


def _simulate_ou(kappa, theta, sigma, dt, n, seed):
    rng = np.random.default_rng(seed)
    X = np.zeros(n)
    X[0] = theta
    for t in range(1, n):
        exp_term = np.exp(-kappa * dt)
        mean = theta + (X[t - 1] - theta) * exp_term
        var = (sigma**2 / (2 * kappa)) * (1 - np.exp(-2 * kappa * dt))
        X[t] = mean + np.sqrt(var) * rng.standard_normal()
    return X


def test_estimate_ou_parameters_recovers_known_process():
    kappa, theta, sigma, dt = 5.0, 0.02, 0.01, 1 / 252
    X = _simulate_ou(kappa, theta, sigma, dt, n=5000, seed=42)

    fitted = estimate_ou_parameters(X, dt=dt)
    assert np.isclose(fitted["kappa"], kappa, rtol=0.3)
    assert np.isclose(fitted["theta"], theta, atol=0.005)
    assert np.isclose(fitted["sigma"], sigma, rtol=0.2)


def test_estimate_ou_parameters_rolling_recovers_known_process_on_array():
    kappa, theta, sigma, dt = 5.0, 0.02, 0.01, 1 / 252
    X = _simulate_ou(kappa, theta, sigma, dt, n=2000, seed=1)

    rolling = estimate_ou_parameters_rolling(X, dt=dt, window=500, step=250)
    assert list(rolling.columns) == [
        "kappa",
        "theta",
        "sigma",
        "half_life_days",
        "log_likelihood",
        "converged",
    ]
    assert len(rolling) == 7  # windows ending at 500, 750, ..., 2000
    # a constant-parameter process should be recovered similarly in every window
    assert np.allclose(rolling["theta"], theta, atol=0.01)


def test_estimate_ou_parameters_rolling_preserves_series_index():
    kappa, theta, sigma, dt = 5.0, 0.02, 0.01, 1 / 252
    X = _simulate_ou(kappa, theta, sigma, dt, n=600, seed=2)
    dates = pd.date_range("2020-01-01", periods=600)
    series = pd.Series(X, index=dates)

    rolling = estimate_ou_parameters_rolling(series, dt=dt, window=300, step=300)
    assert list(rolling.index) == [dates[299], dates[599]]


def test_most_recent_ou_parameters_detects_a_regime_change():
    """A factor that mean-reverts fast in its early history but has settled
    into a slower regime recently should be characterized by its *recent*
    behavior, not an average over both regimes -- this is the whole point of
    using a rolling window instead of one full-sample fit (see TODO.md)."""
    dt = 1 / 252
    fast_regime = _simulate_ou(kappa=40.0, theta=0.0, sigma=0.03, dt=dt, n=1000, seed=3)
    slow_regime = _simulate_ou(kappa=0.3, theta=0.0, sigma=0.01, dt=dt, n=1000, seed=4)
    combined = pd.DataFrame({"PC1": np.concatenate([fast_regime, slow_regime])})

    recent = most_recent_ou_parameters(combined, dt=dt, window=500, step=500)
    full_sample = estimate_ou_parameters(combined["PC1"].values, dt=dt)
    # the last window (positions 1500:2000) is entirely inside the slow
    # regime, so its kappa should be far below both the fast regime's true
    # value and a naive full-sample fit averaging over both regimes.
    assert recent["PC1"]["kappa"] < 20.0
    assert recent["PC1"]["kappa"] < full_sample["kappa"]


def test_estimate_ou_parameters_for_factors_rolling_returns_per_factor_dataframes():
    dt = 1 / 252
    scores_df = pd.DataFrame(
        {
            "PC1": _simulate_ou(5.0, 0.0, 0.01, dt, 600, seed=5),
            "PC2": _simulate_ou(2.0, 0.0, 0.02, dt, 600, seed=6),
        }
    )
    rolling = estimate_ou_parameters_for_factors_rolling(scores_df, dt=dt, window=300, step=300)
    assert set(rolling.keys()) == {"PC1", "PC2"}
    assert len(rolling["PC1"]) == 2
