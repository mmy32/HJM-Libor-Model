import numpy as np

from src.curves.nelson_siegel import nelson_siegel_forward
from src.transform.representations import forward_to_zero_rates, ns_params_to_curve, pcs_to_ns_params


def test_pcs_to_ns_params_returns_mean_when_alpha_is_zero():
    mean_params = {"b0_level": 0.03, "b1_slope": -0.01, "b2_curvature": 0.005, "lambda": 0.5}
    loadings = np.eye(4)
    result = pcs_to_ns_params(np.zeros(4), mean_params, loadings)
    assert np.isclose(result["b0"], 0.03)
    assert np.isclose(result["lambda"], 0.5)


def test_ns_params_to_curve_matches_nelson_siegel_forward():
    params = {"b0": 0.03, "b1": -0.01, "b2": 0.005, "lambda": 0.5}
    maturities = np.array([1.0, 5.0, 10.0])
    curve = ns_params_to_curve(params, maturities)
    expected = nelson_siegel_forward(maturities, 0.03, -0.01, 0.005, 0.5)
    assert np.allclose(curve, expected)


def test_forward_to_zero_rates_constant_curve_is_unchanged():
    maturities = np.array([1.0, 2.0, 5.0, 10.0])
    forward = np.full_like(maturities, 0.03)
    zero = forward_to_zero_rates(forward, maturities)
    assert np.allclose(zero, 0.03, atol=1e-8)
