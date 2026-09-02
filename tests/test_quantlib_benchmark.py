import numpy as np
import pytest

pytest.importorskip("QuantLib")

from project.curves.nelson_siegel import fit_ns_robust
from project.curves.quantlib_benchmark import compare_ns_to_quantlib


def test_compare_ns_to_quantlib_matches_input_quotes_closely():
    tenors = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    yields = np.array([0.052, 0.051, 0.049, 0.045, 0.043, 0.041, 0.042, 0.043, 0.046, 0.045])
    ns_params = fit_ns_robust(yields, tenors, seed=0)

    result = compare_ns_to_quantlib(tenors, yields, ns_params, tenors)

    # QuantLib's ZeroCurve interpolates directly through the quotes.
    assert np.allclose(result["quantlib_zero_rates"], yields, atol=1e-8)
    # NS is a parametric 4-parameter fit, not an exact interpolation --
    # it should track the quotes closely but not exactly.
    assert result["rmse"] < 0.005
