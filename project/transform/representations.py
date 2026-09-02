"""Pure conversions between PCA factor scores, Nelson-Siegel parameters, and curve representations.

These were originally private methods on the archived HJMSimulator class;
they are promoted to free functions here so the same conversions can be
shared between calibration (the sensitivity chain rule) and the model's
simulation loop, and so each is independently unit-testable without any
model/instance state.
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid

from project.curves.nelson_siegel import nelson_siegel_forward
from project.registry.factor_spec import NS_PARAM_NAMES

_NS_KEYS = ["b0", "b1", "b2", "lambda"]


def pcs_to_ns_params(alpha, mean_params, loadings, param_scale=None, bounds=None):
    """theta = theta_bar + param_scale * (Loadings @ alpha).

    `alpha` is either a single factor vector (n_factors,) or a batch
    (n_paths, n_factors); the return dict's values match that shape (scalars
    or (n_paths,) arrays). `loadings` is (n_ns_params, n_factors) ordered per
    registry.factor_spec.NS_PARAM_NAMES.

    `loadings` here are raw PCA components, fit on *standardized* (zero-mean,
    unit-variance) NS-parameter data -- a one-unit move in a PC score
    therefore corresponds to one standard deviation of the *scaled* feature,
    not one raw unit. `param_scale` (the fitted StandardScaler's per-feature
    `scale_`, NS_PARAM_NAMES-ordered) converts that standardized-space delta
    back to raw NS-parameter units before it's added to the raw-space mean.
    Omitting it treats the scale as 1, i.e. no rescaling -- only correct if
    `loadings` was itself fit on unstandardized data. This rescaling was
    previously missing entirely, which overstated every reconstructed
    parameter's swing (particularly `lambda`, which the affected the least
    proportionally but sits inside an exponential in the forward-rate
    formula) and was the root cause of the simulator exploding on real
    calibrated data.

    `bounds`, if given (a NS_PARAM_NAMES-ordered list of (lo, hi) pairs),
    clips the reconstructed values into range as a defensive backstop: the
    affine PCA reconstruction is only a linear approximation, so individual
    Monte Carlo draws -- especially with fast-mean-reverting, high-variance
    OU factors -- can still land outside the region the original per-day
    fits were constrained to.
    """
    mean_array = np.array([mean_params[name] for name in NS_PARAM_NAMES])
    alpha = np.asarray(alpha, dtype=float)
    loadings = np.asarray(loadings, dtype=float)
    scale = (
        np.ones(len(NS_PARAM_NAMES))
        if param_scale is None
        else np.asarray(param_scale, dtype=float)
    )

    batched = alpha.ndim == 2
    if batched:
        values = mean_array + scale * (alpha @ loadings.T)
    else:
        values = mean_array + scale * (loadings @ alpha)

    if bounds is not None:
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        values = np.clip(values, lo, hi)

    if batched:
        return {k: values[:, i] for i, k in enumerate(_NS_KEYS)}
    return dict(zip(_NS_KEYS, values))


def ns_params_to_curve(params, maturities):
    """Generate a forward curve from NS parameters.

    `params["b0"/"b1"/"b2"/"lambda"]` are either scalars (-> a single
    (n_maturities,) curve) or (n_paths,) arrays (-> a batched
    (n_paths, n_maturities) curve).
    """
    maturities = np.asarray(maturities, dtype=float)
    b0 = np.asarray(params["b0"], dtype=float)
    b1 = np.asarray(params["b1"], dtype=float)
    b2 = np.asarray(params["b2"], dtype=float)
    lam = np.asarray(params["lambda"], dtype=float)

    if b0.ndim == 0:
        return nelson_siegel_forward(maturities, b0, b1, b2, lam)
    return nelson_siegel_forward(
        maturities[None, :], b0[:, None], b1[:, None], b2[:, None], lam[:, None]
    )


def forward_to_zero_rates(forward_curve, maturities):
    """r(0,T) = (1/T) * integral_0^T f(0,s) ds, via trapezoidal rule on the maturity grid.

    The maturity grid typically starts above 0 (e.g. at 3 months), so the
    segment from 0 to the first observed maturity is filled in by
    extrapolating f(0) ~= f(maturities[0]) -- the archived version of this
    function integrated only from maturities[0] onward and silently dropped
    that segment, which understated every zero rate (badly so at short
    maturities).

    `forward_curve` is either a single (n_maturities,) curve or a batched
    (n_paths, n_maturities) array of curves, vectorized via
    `scipy.integrate.cumulative_trapezoid` instead of a per-path Python loop.
    """
    maturities = np.asarray(maturities, dtype=float)
    forward_curve = np.asarray(forward_curve, dtype=float)
    batched = forward_curve.ndim == 2
    curves = forward_curve if batched else forward_curve[None, :]

    full_maturities = np.concatenate(([0.0], maturities))
    full_forward = np.concatenate([curves[:, :1], curves], axis=1)

    cumulative = cumulative_trapezoid(full_forward, full_maturities, axis=1, initial=0.0)
    integral_to_T = cumulative[:, 1:]  # integral(0 -> maturities[j]) for each j
    safe_maturities = np.where(maturities <= 0, 1.0, maturities)
    zero_rates = np.where(maturities <= 0, curves, integral_to_T / safe_maturities)

    return zero_rates if batched else zero_rates[0]
