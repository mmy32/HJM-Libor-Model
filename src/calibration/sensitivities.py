"""Analytical sensitivities of the Nelson-Siegel forward curve to its parameters and to PCA factors."""
import numpy as np
import pandas as pd

from src.registry.factor_spec import NS_PARAM_NAMES


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


def compute_forward_sensitivities(mean_params: dict, loadings: pd.DataFrame, maturities) -> pd.DataFrame:
    """Chain-rule PC -> forward-rate sensitivities: df/dPC_k = sum_i (df/dtheta_i) * loadings[i, k]."""
    sens = ns_sensitivities(
        maturities,
        mean_params["b0_level"],
        mean_params["b1_slope"],
        mean_params["b2_curvature"],
        mean_params["lambda"],
    )
    sens_matrix = np.column_stack([sens["db0"], sens["db1"], sens["db2"], sens["dlambda"]])
    loadings_ordered = loadings.loc[NS_PARAM_NAMES]
    pc_sens = sens_matrix @ loadings_ordered.values
    return pd.DataFrame(pc_sens, index=np.asarray(maturities, dtype=float), columns=loadings_ordered.columns)
