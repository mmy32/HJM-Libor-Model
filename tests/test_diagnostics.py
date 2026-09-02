import numpy as np
import pandas as pd

from project.calibration.diagnostics import (
    ns_fit_residuals,
    pca_out_of_sample_reconstruction_error,
    rolling_pca_loading_stability,
)
from project.curves.nelson_siegel import nelson_siegel_yield


def test_ns_fit_residuals_are_near_zero_for_exact_params():
    tenors = np.array([0.25, 1, 2, 5, 10, 30])
    dates = pd.date_range("2020-01-01", periods=3)
    ns_params_df = pd.DataFrame(
        {
            "b0_level": [0.03] * 3,
            "b1_slope": [-0.01] * 3,
            "b2_curvature": [0.005] * 3,
            "lambda": [0.5] * 3,
        },
        index=dates,
    )
    yields_df = pd.DataFrame(
        [nelson_siegel_yield(tenors, 0.03, -0.01, 0.005, 0.5) for _ in range(3)],
        index=dates,
        columns=tenors,
    )
    residuals = ns_fit_residuals(yields_df, ns_params_df, tenors)
    assert np.allclose(residuals.values.astype(float), 0.0, atol=1e-10)


def test_ns_fit_residuals_nonzero_when_params_dont_match_curve():
    tenors = np.array([0.25, 1, 2, 5, 10, 30])
    dates = pd.date_range("2020-01-01", periods=1)
    ns_params_df = pd.DataFrame(
        {"b0_level": [0.02], "b1_slope": [-0.01], "b2_curvature": [0.005], "lambda": [0.5]},
        index=dates,
    )
    yields_df = pd.DataFrame(
        [nelson_siegel_yield(tenors, 0.03, -0.01, 0.005, 0.5)], index=dates, columns=tenors
    )
    residuals = ns_fit_residuals(yields_df, ns_params_df, tenors)
    # actual yields were built from b0=0.03, fitted params use b0=0.02, so
    # actual - fitted should be uniformly +0.01 (the b0 gap; other params match).
    assert np.allclose(residuals.values.astype(float), 0.01, atol=1e-10)


def test_rolling_pca_loading_stability_is_high_when_structure_is_constant():
    rng = np.random.default_rng(0)
    n = 200
    base = rng.normal(size=(n, 1)) @ np.array([[1.0, 0.5, 0.2, -0.3]])
    noise = rng.normal(scale=0.01, size=(n, 4))
    params_df = pd.DataFrame(
        base + noise + np.array([0.03, -0.01, 0.005, 0.5]),
        columns=["b0_level", "b1_slope", "b2_curvature", "lambda"],
    )
    similarities = rolling_pca_loading_stability(params_df, window=60)
    assert similarities.abs().mean() > 0.9


def test_pca_out_of_sample_reconstruction_error_returns_both_metrics():
    rng = np.random.default_rng(1)
    n = 100
    base = rng.normal(size=(n, 2)) @ rng.normal(size=(2, 4))
    params_df = pd.DataFrame(
        base + np.array([0.03, -0.01, 0.005, 0.5]),
        columns=["b0_level", "b1_slope", "b2_curvature", "lambda"],
    )
    result = pca_out_of_sample_reconstruction_error(params_df, train_frac=0.7, n_components=2)
    assert "in_sample_rmse" in result and "out_of_sample_rmse" in result
    assert result["in_sample_rmse"] >= 0
    assert result["out_of_sample_rmse"] >= 0
