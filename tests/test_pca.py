import numpy as np

from project.calibration.pca import fit_pca, transform_pca


def test_fit_pca_recovers_full_variance_with_all_components(synthetic_ns_params_df):
    model = fit_pca(synthetic_ns_params_df, n_components=4)
    assert np.isclose(model.explained_variance_ratio.sum(), 1.0, atol=1e-6)
    assert model.loadings.shape == (4, 4)


def test_transform_pca_matches_original_fit_scores(synthetic_ns_params_df):
    model = fit_pca(synthetic_ns_params_df, n_components=4)
    projected = transform_pca(model, synthetic_ns_params_df)
    assert np.allclose(projected.values, model.scores.values, atol=1e-8)
