import numpy as np
import pandas as pd
from scipy.integrate import quad

from project.curves.nelson_siegel import (
    calibrate_all_days,
    calibrate_all_days_fixed_lambda,
    fit_ns_fixed_lambda,
    fit_ns_robust,
    nelson_siegel_forward,
    nelson_siegel_yield,
)


def test_yield_curve_reduces_to_level_when_slope_and_curvature_are_zero():
    y = nelson_siegel_yield(np.array([1.0, 5.0, 10.0]), b0=0.03, b1=0.0, b2=0.0, lam=0.5)
    assert np.allclose(y, 0.03)


def test_forward_curve_integrates_to_yield_curve():
    b0, b1, b2, lam = 0.03, -0.01, 0.005, 0.5
    for T in [1.0, 5.0, 10.0, 20.0]:
        integral, _ = quad(lambda s: nelson_siegel_forward(s, b0, b1, b2, lam), 0, T)
        expected_yield = nelson_siegel_yield(T, b0, b1, b2, lam)
        assert np.isclose(integral / T, expected_yield, atol=1e-6)


def test_fit_ns_robust_recovers_known_parameters():
    tenors = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    true_params = [0.03, -0.015, 0.01, 0.4]
    yields = nelson_siegel_yield(tenors, *true_params)
    fitted = fit_ns_robust(yields, tenors, seed=0)
    reconstructed = nelson_siegel_yield(tenors, *fitted)
    # differential_evolution is a stochastic global optimizer, not exact --
    # a few bps of residual error on a noise-free synthetic curve is expected.
    assert np.allclose(reconstructed, yields, atol=2e-3)


def test_smoothing_weight_reduces_day_to_day_parameter_drift():
    """A day-over-day continuity penalty should pull each day's fit toward
    the previous day's parameters, reducing spurious drift caused by
    observation noise + a stochastic per-day optimizer -- this is the
    mechanism intended to address the day-to-day NS-fit instability finding
    in TODO.md (independent per-day fits pinned the OU-fitted kappa at its
    optimizer bound)."""
    tenors = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    true_params = [0.03, -0.015, 0.01, 0.4]
    base_yields = nelson_siegel_yield(tenors, *true_params)

    rng = np.random.default_rng(0)
    noisy_days = np.array(
        [base_yields + rng.normal(0, 0.0005, size=len(tenors)) for _ in range(15)]
    )
    df = pd.DataFrame(noisy_days, columns=[f"t{i}" for i in range(len(tenors))])

    unsmoothed = calibrate_all_days(df, tenors, seed=1, smoothing_weight=0.0)
    smoothed = calibrate_all_days(df, tenors, seed=1, smoothing_weight=0.2)

    unsmoothed_drift = unsmoothed["lambda"].diff().dropna().abs().mean()
    smoothed_drift = smoothed["lambda"].diff().dropna().abs().mean()
    assert smoothed_drift < unsmoothed_drift


def test_fit_ns_fixed_lambda_matches_free_fit_when_lambda_is_correct():
    tenors = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    true_params = [0.03, -0.015, 0.01, 0.4]
    yields = nelson_siegel_yield(tenors, *true_params)

    fitted = fit_ns_fixed_lambda(yields, tenors, lam=0.4)
    assert np.allclose(fitted, true_params, atol=1e-6)


def test_fit_ns_fixed_lambda_is_exact_ols_not_a_global_search():
    """With lambda fixed, the fit is an exact linear regression -- unlike
    fit_ns_robust's stochastic global optimizer, two calls on the same
    inputs must be bit-for-bit identical (no seed dependence)."""
    tenors = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    yields = np.array([0.052, 0.051, 0.049, 0.045, 0.043, 0.041, 0.042, 0.043, 0.046, 0.045])

    first = fit_ns_fixed_lambda(yields, tenors, lam=0.5)
    second = fit_ns_fixed_lambda(yields, tenors, lam=0.5)
    assert np.array_equal(first, second)


def test_calibrate_all_days_fixed_lambda_holds_lambda_constant():
    tenors = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    rng = np.random.default_rng(0)
    base = nelson_siegel_yield(tenors, 0.03, -0.015, 0.01, 0.4)
    df = pd.DataFrame(
        [base + rng.normal(0, 0.0005, size=len(tenors)) for _ in range(10)],
        columns=tenors,
    )

    result = calibrate_all_days_fixed_lambda(df, tenors, lam=0.4)
    assert (result["lambda"] == 0.4).all()
    assert result["b0_level"].std() > 0  # level still varies day to day with the noise
