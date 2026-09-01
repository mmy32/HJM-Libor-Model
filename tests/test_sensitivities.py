import numpy as np

from src.calibration.sensitivities import ns_sensitivities
from src.curves.nelson_siegel import nelson_siegel_forward


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
