"""Pure conversions between PCA factor scores, Nelson-Siegel parameters, and curve representations.

These were originally private methods on the archived HJMSimulator class;
they are promoted to free functions here so the same conversions can be
shared between calibration (the sensitivity chain rule) and the model's
simulation loop, and so each is independently unit-testable without any
model/instance state.
"""
import numpy as np

from src.curves.nelson_siegel import nelson_siegel_forward
from src.registry.factor_spec import NS_PARAM_NAMES

_NS_KEYS = ["b0", "b1", "b2", "lambda"]


def pcs_to_ns_params(alpha, mean_params, loadings):
    """theta = theta_bar + Loadings @ alpha, for a single factor vector alpha (n_factors,).

    `loadings` is (n_ns_params, n_factors) ordered per
    registry.factor_spec.NS_PARAM_NAMES.
    """
    mean_array = np.array([mean_params[name] for name in NS_PARAM_NAMES])
    alpha = np.asarray(alpha, dtype=float)
    loadings = np.asarray(loadings, dtype=float)
    values = mean_array + loadings @ alpha
    return dict(zip(_NS_KEYS, values))


def ns_params_to_curve(params, maturities):
    """Generate a forward curve from NS parameters."""
    return nelson_siegel_forward(
        maturities, params["b0"], params["b1"], params["b2"], params["lambda"]
    )


def forward_to_zero_rates(forward_curve, maturities):
    """r(0,T) = (1/T) * integral_0^T f(0,s) ds, via trapezoidal rule on the maturity grid.

    The maturity grid typically starts above 0 (e.g. at 3 months), so the
    segment from 0 to the first observed maturity is filled in by
    extrapolating f(0) ~= f(maturities[0]) -- the archived version of this
    function integrated only from maturities[0] onward and silently dropped
    that segment, which understated every zero rate (badly so at short
    maturities).
    """
    maturities = np.asarray(maturities, dtype=float)
    forward_curve = np.asarray(forward_curve, dtype=float)
    full_maturities = np.concatenate(([0.0], maturities))
    full_forward = np.concatenate(([forward_curve[0]], forward_curve))

    zero_rates = np.zeros(len(maturities))
    for i, T in enumerate(maturities):
        if T <= 0:
            zero_rates[i] = forward_curve[i]
            continue
        mask = full_maturities <= T
        zero_rates[i] = np.trapezoid(full_forward[mask], full_maturities[mask]) / T
    return zero_rates
