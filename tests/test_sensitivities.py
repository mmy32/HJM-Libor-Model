import numpy as np
import pandas as pd

from project.calibration.sensitivities import compute_forward_sensitivities, ns_sensitivities
from project.curves.nelson_siegel import nelson_siegel_forward


def test_ns_sensitivities_match_finite_differences():
    maturities = np.array([0.5, 1, 2, 5, 10, 20])
    b0, b1, b2, lam = 0.03, -0.01, 0.005, 0.5
    eps = 1e-6

    analytical = ns_sensitivities(maturities, b0, b1, b2, lam)

    numerical_db0 = (
        nelson_siegel_forward(maturities, b0 + eps, b1, b2, lam)
        - nelson_siegel_forward(maturities, b0 - eps, b1, b2, lam)
    ) / (2 * eps)
    numerical_dlambda = (
        nelson_siegel_forward(maturities, b0, b1, b2, lam + eps)
        - nelson_siegel_forward(maturities, b0, b1, b2, lam - eps)
    ) / (2 * eps)

    assert np.allclose(analytical["db0"], numerical_db0, atol=1e-4)
    assert np.allclose(analytical["dlambda"], numerical_dlambda, atol=1e-4)


def test_compute_forward_sensitivities_scales_by_param_scale():
    """dtheta_i/dPC_k = param_scale[i] * loadings[i, k] -- loadings are PCA
    components fit on standardized data, so omitting param_scale (the
    historical bug) mixes standardized-space and raw-space units."""
    mean_params = {"b0_level": 0.03, "b1_slope": -0.01, "b2_curvature": 0.005, "lambda": 0.5}
    maturities = np.array([1.0, 5.0, 10.0])
    loadings = pd.DataFrame(
        np.eye(4),
        index=["b0_level", "b1_slope", "b2_curvature", "lambda"],
        columns=["PC1", "PC2", "PC3", "PC4"],
    )
    param_scale = np.array([0.01, 0.01, 0.02, 0.6])

    unscaled = compute_forward_sensitivities(mean_params, loadings, maturities)
    scaled = compute_forward_sensitivities(
        mean_params, loadings, maturities, param_scale=param_scale
    )

    # PC4 loads purely onto lambda (identity loadings), so its sensitivity
    # column should be exactly param_scale[3] times the unscaled one.
    assert np.allclose(scaled["PC4"].values, param_scale[3] * unscaled["PC4"].values)
