import numpy as np

from project.curves.nelson_siegel import nelson_siegel_forward
from project.transform.representations import (
    forward_to_zero_rates,
    ns_params_to_curve,
    pcs_to_ns_params,
)


def test_pcs_to_ns_params_returns_mean_when_alpha_is_zero():
    mean_params = {"b0_level": 0.03, "b1_slope": -0.01, "b2_curvature": 0.005, "lambda": 0.5}
    loadings = np.eye(4)
    result = pcs_to_ns_params(np.zeros(4), mean_params, loadings)
    assert np.isclose(result["b0"], 0.03)
    assert np.isclose(result["lambda"], 0.5)


def test_pcs_to_ns_params_rescales_by_param_scale():
    """A standardized-space PC delta must be multiplied by param_scale before
    being added to the raw-space mean -- omitting this was the root cause of
    the simulator's exponential blow-up on real calibrated data (see
    hjm_model.py / TODO.md)."""
    mean_params = {"b0_level": 0.03, "b1_slope": -0.01, "b2_curvature": 0.005, "lambda": 0.5}
    loadings = np.eye(4)
    alpha = np.array([0.0, 0.0, 0.0, 1.0])  # one standardized-score unit on lambda's own axis
    param_scale = np.array([0.01, 0.01, 0.02, 0.6])

    unscaled = pcs_to_ns_params(alpha, mean_params, loadings)
    scaled = pcs_to_ns_params(alpha, mean_params, loadings, param_scale=param_scale)

    assert np.isclose(unscaled["lambda"], 1.5)  # 0.5 + 1.0, the historical bug's behavior
    assert np.isclose(scaled["lambda"], 1.1)  # 0.5 + 0.6 * 1.0, the correct raw-unit delta


def test_pcs_to_ns_params_clips_to_bounds():
    mean_params = {"b0_level": 0.03, "b1_slope": -0.01, "b2_curvature": 0.005, "lambda": 0.5}
    loadings = np.eye(4)
    bounds = [(0, 0.15), (-0.1, 0.1), (-0.1, 0.1), (0.01, 2.0)]

    # A huge PC1 shock would otherwise push lambda far negative -- the exact
    # failure mode that exploded nelson_siegel_forward's exp(-lambda*tau).
    alpha = np.array([0.0, 0.0, 0.0, -100.0])
    result = pcs_to_ns_params(alpha, mean_params, loadings, bounds=bounds)
    assert result["lambda"] == 0.01

    alpha = np.array([0.0, 0.0, 0.0, 100.0])
    result = pcs_to_ns_params(alpha, mean_params, loadings, bounds=bounds)
    assert result["lambda"] == 2.0


def test_pcs_to_ns_params_batched_matches_single_path():
    mean_params = {"b0_level": 0.03, "b1_slope": -0.01, "b2_curvature": 0.005, "lambda": 0.5}
    loadings = np.array([[1.0, 0.2], [0.3, 1.0], [0.5, 0.5], [0.7, -0.4]])
    param_scale = np.array([0.01, 0.01, 0.02, 0.6])
    alphas = np.array([[0.5, -0.3], [1.2, 0.8], [-2.0, 0.1]])

    batched = pcs_to_ns_params(alphas, mean_params, loadings, param_scale=param_scale)
    for i in range(len(alphas)):
        single = pcs_to_ns_params(alphas[i], mean_params, loadings, param_scale=param_scale)
        for key in ("b0", "b1", "b2", "lambda"):
            assert np.isclose(batched[key][i], single[key])


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
