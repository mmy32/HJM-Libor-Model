"""Calibration diagnostics beyond visual inspection: residual analysis,
PCA factor stability over time, and out-of-sample reconstruction error.

These complement (don't replace) the notebook's plots -- each function
returns a plain DataFrame/dict of numbers so results can be asserted on in
tests and inspected without rendering a figure.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from project.curves.nelson_siegel import nelson_siegel_yield


def ns_fit_residuals(yields_df: pd.DataFrame, ns_params_df: pd.DataFrame, tenors) -> pd.DataFrame:
    """Actual-minus-fitted yield residuals for every (date, tenor), from the
    per-day Nelson-Siegel calibration.

    `yields_df` and `ns_params_df` must share the same date index (as
    produced by `curves.nelson_siegel.calibrate_all_days` on `yields_df`).
    A day's fit residual RMSE that's persistently large relative to other
    days flags either a poor optimizer run or a curve shape NS can't
    represent well that day (e.g. a sharp kink from an idiosyncratic
    auction/liquidity effect at one tenor).
    """
    tenors = np.asarray(tenors, dtype=float)
    common_dates = yields_df.index.intersection(ns_params_df.index)
    residuals = pd.DataFrame(index=common_dates, columns=yields_df.columns, dtype=float)
    for date in common_dates:
        params = ns_params_df.loc[date]
        fitted = nelson_siegel_yield(
            tenors, params["b0_level"], params["b1_slope"], params["b2_curvature"], params["lambda"]
        )
        residuals.loc[date] = yields_df.loc[date].values - fitted
    return residuals


def rolling_pca_loading_stability(
    params_df: pd.DataFrame, window=126, n_components=None
) -> pd.Series:
    """Cosine similarity, over time, between a rolling-window PCA's leading
    loading vector and the full-sample leading loading vector.

    A value near 1.0 means the rolling window agrees with the full-sample
    factor structure; values drifting toward 0 (or negative, i.e. a
    sign-flipped factor) flag periods where the dominant driver of NS
    parameter variation is genuinely different from the full-sample average
    -- e.g. a regime change -- rather than the day-to-day fit noise that
    `curves.nelson_siegel.calibrate_all_days`'s `smoothing_weight` targets.
    """
    scaler = StandardScaler()
    scaled_full = scaler.fit_transform(params_df)
    full_pca = PCA(n_components=n_components or 1)
    full_pca.fit(scaled_full)
    full_leading = full_pca.components_[0]

    similarities = {}
    for end in range(window, len(params_df) + 1):
        window_df = params_df.iloc[end - window : end]
        scaled_window = StandardScaler().fit_transform(window_df)
        window_pca = PCA(n_components=n_components or 1)
        window_pca.fit(scaled_window)
        window_leading = window_pca.components_[0]
        cosine_sim = np.dot(full_leading, window_leading) / (
            np.linalg.norm(full_leading) * np.linalg.norm(window_leading)
        )
        similarities[params_df.index[end - 1]] = cosine_sim

    return pd.Series(similarities, name="leading_pc_cosine_similarity")


def pca_out_of_sample_reconstruction_error(
    params_df: pd.DataFrame, train_frac=0.7, n_components=None
) -> dict:
    """Fit PCA on the first `train_frac` of dates, then measure NS-parameter
    reconstruction error on the held-out remainder.

    Returns {"in_sample_rmse", "out_of_sample_rmse"} (raw NS-parameter
    units). A much larger out-of-sample RMSE than in-sample indicates the
    fitted factor basis doesn't generalize -- e.g. the held-out period has a
    genuinely different curve-shape regime.
    """
    from project.registry.factor_spec import N_PCA_FACTORS

    n_components = n_components or N_PCA_FACTORS
    split = int(len(params_df) * train_frac)
    train_df, test_df = params_df.iloc[:split], params_df.iloc[split:]

    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_df)
    pca = PCA(n_components=n_components)
    scores_train = pca.fit_transform(scaled_train)
    reconstructed_train = scaler.inverse_transform(pca.inverse_transform(scores_train))
    in_sample_rmse = float(np.sqrt(np.mean((train_df.values - reconstructed_train) ** 2)))

    scaled_test = scaler.transform(test_df)
    scores_test = pca.transform(scaled_test)
    reconstructed_test = scaler.inverse_transform(pca.inverse_transform(scores_test))
    out_of_sample_rmse = float(np.sqrt(np.mean((test_df.values - reconstructed_test) ** 2)))

    return {"in_sample_rmse": in_sample_rmse, "out_of_sample_rmse": out_of_sample_rmse}
