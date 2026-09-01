import numpy as np
from scipy.integrate import quad

from src.curves.nelson_siegel import fit_ns_robust, nelson_siegel_forward, nelson_siegel_yield


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
