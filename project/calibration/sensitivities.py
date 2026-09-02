"""Analytical sensitivities of the Nelson-Siegel forward curve to its parameters and to PCA factors."""

import numpy as np
import pandas as pd

from project.registry.factor_spec import NS_PARAM_NAMES


def ns_sensitivities(maturities, b0, b1, b2, lam):
    """Partial derivatives of curves.nelson_siegel.nelson_siegel_forward w.r.t. b0, b1, b2, lam.

    f(tau) = b0 + b1*exp(-lam*tau) + b2*lam*tau*exp(-lam*tau)
    """
    tau = np.asarray(maturities, dtype=float)
    exp_term = np.exp(-lam * tau)

    df_db0 = np.ones_like(tau)
    df_db1 = exp_term
    df_db2 = lam * tau * exp_term
    df_dlambda = tau * exp_term * (b2 * (1 - lam * tau) - b1)

    return {"db0": df_db0, "db1": df_db1, "db2": df_db2, "dlambda": df_dlambda}


def compute_forward_sensitivities(
    mean_params: dict, loadings: pd.DataFrame, maturities, param_scale=None
) -> pd.DataFrame:
    """Chain-rule PC -> forward-rate sensitivities: df/dPC_k = sum_i (df/dtheta_i) * dtheta_i/dPC_k.

    `loadings` are raw PCA components fit on *standardized* NS-parameter
    data, so dtheta_i/dPC_k = param_scale[i] * loadings[i, k] -- the same
    rescaling `transform.representations.pcs_to_ns_params` applies when
    reconstructing NS parameters from PC scores. `param_scale` is the fitted
    StandardScaler's per-feature `scale_`, NS_PARAM_NAMES-ordered; omitting
    it treats the scale as 1 (only correct for unstandardized loadings).

    `loadings` may have fewer than 4 rows -- e.g. when NS parameters were
    fit with lambda held fixed (`curves.nelson_siegel.calibrate_all_days_fixed_lambda`),
    PCA is only meaningful over b0/b1/b2 (lambda has zero cross-sectional
    variance), so there's no lambda row at all. Missing rows are filled with
    0 (no PC moves that parameter), which correctly reproduces df/dPC_k for
    the parameters that *are* in the PCA basis. `param_scale`, if given, is
    expected aligned with `loadings`'s own row index (e.g. the fitted
    StandardScaler's `scale_` in that same, possibly-shorter, order) and is
    padded/reindexed the same way.
    """
    sens = ns_sensitivities(
        maturities,
        mean_params["b0_level"],
        mean_params["b1_slope"],
        mean_params["b2_curvature"],
        mean_params["lambda"],
    )
    sens_matrix = np.column_stack([sens["db0"], sens["db1"], sens["db2"], sens["dlambda"]])
    loadings_ordered = loadings.reindex(NS_PARAM_NAMES, fill_value=0.0)
    if param_scale is None:
        scale = np.ones(len(NS_PARAM_NAMES))
    else:
        scale = (
            pd.Series(np.asarray(param_scale, dtype=float), index=loadings.index)
            .reindex(NS_PARAM_NAMES, fill_value=0.0)
            .values
        )
    scaled_loadings = loadings_ordered.values * scale[:, None]
    pc_sens = sens_matrix @ scaled_loadings
    return pd.DataFrame(
        pc_sens, index=np.asarray(maturities, dtype=float), columns=loadings_ordered.columns
    )
