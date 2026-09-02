import numpy as np
import pandas as pd

from project.calibration.pca import PCAFactorModel
from project.persistence import artifacts
from project.stochastic.hjm_model import HJMModel, HJMModelParams


def _tiny_params():
    return HJMModelParams(
        ou_params={
            "PC1": {"kappa": 5.0, "theta": 0.0, "sigma": 0.01},
            "PC2": {"kappa": 2.0, "theta": 0.0, "sigma": 0.005},
        },
        loadings=np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.1, -0.1]]),
        pc_sensitivities=np.array([[1.0, 0.5], [0.8, 0.4], [0.5, 0.2]]),
        mean_params={"b0_level": 0.03, "b1_slope": -0.01, "b2_curvature": 0.0, "lambda": 0.5},
        maturities=np.array([1.0, 5.0, 10.0]),
    )


def test_simulate_returns_correctly_shaped_nan_free_output():
    model = HJMModel(_tiny_params())
    result = model.simulate(n_paths=5, T_horizon=0.05, dt=1 / 252, measure="P", random_seed=0)
    assert result.zero_curves.shape == (5, result.time_grid.shape[0], 3)
    assert not np.isnan(result.zero_curves).any()
    assert not np.isnan(result.forward_curves).any()


def test_p_measure_reverts_toward_theta_over_long_horizon():
    model = HJMModel(_tiny_params())
    result = model.simulate(n_paths=200, T_horizon=2.0, dt=1 / 52, measure="P", random_seed=1)
    final_pc1 = result.pc_paths[:, -1, 0]
    assert abs(final_pc1.mean()) < 0.02  # theta for PC1 is 0


def test_q_measure_runs_without_error():
    model = HJMModel(_tiny_params())
    result = model.simulate(n_paths=3, T_horizon=0.05, dt=1 / 252, measure="Q", random_seed=2)
    assert result.measure == "Q"


def _blowup_prone_params():
    """Mimics the real-data conditions that exploded before the param_scale
    fix and bounds clamp: a fast-mean-reverting, high-sigma factor whose
    stationary standard deviation in PC-score space is large (~1.5), with a
    loading that puts most of its weight on `lambda`."""
    return HJMModelParams(
        ou_params={
            "PC1": {"kappa": 50.0, "theta": 0.0, "sigma": 15.0},
            "PC2": {"kappa": 0.8, "theta": 0.0, "sigma": 1.4},
        },
        loadings=np.array([[-0.28, 0.71], [0.21, 0.69], [0.72, -0.03], [0.60, 0.13]]),
        pc_sensitivities=np.array([[1.0, 0.5], [0.8, 0.4], [0.5, 0.2]]),
        mean_params={"b0_level": 0.035, "b1_slope": -0.008, "b2_curvature": -0.01, "lambda": 0.70},
        maturities=np.array([1.0, 5.0, 10.0, 30.0]),
        param_scale=np.array([0.012, 0.013, 0.025, 0.60]),
    )


def test_simulate_does_not_explode_with_real_world_like_ou_parameters():
    """Regression test for the exponential blow-up this exact combination of
    parameters (fast-mean-reverting, high-sigma PC1 pinned at the OU
    optimizer's kappa bound) produced on real calibrated data before the
    param_scale rescaling fix and the NS_PARAM_BOUNDS clamp in
    HJMModel._reconstruct_curve. Before the fix, this configuration drove
    `lambda` non-positive in ~29% of (path, timestep) pairs and blew up
    simulated 10Y zero rates to the hundreds of percent."""
    model = HJMModel(_blowup_prone_params())
    result = model.simulate(n_paths=500, T_horizon=0.25, dt=1 / 252, measure="P", random_seed=3)

    lam = result.ns_params[:, :, 3]
    assert (lam > 0).all()
    assert (lam >= 0.01).all() and (lam <= 2.0).all()

    ten_year_idx = 2
    ten_year_rate_pct = result.zero_curves[:, -1, ten_year_idx] * 100
    assert (
        np.abs(ten_year_rate_pct).max() < 50
    )  # sane range, nowhere near the pre-fix hundreds-of-percent blowup


def test_from_disk_handles_fixed_lambda_three_factor_pca(tmp_path):
    """When NS parameters were fit with lambda held fixed (see
    curves.nelson_siegel.calibrate_all_days_fixed_lambda), PCA only covers
    b0/b1/b2 -- the persisted pca_model.loadings has no 'lambda' row at all.
    from_disk must fill that missing row with 0 (no PC moves lambda) rather
    than KeyError, and the reconstructed lambda must always equal the fixed
    value from mean_params."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    fixed_lambda = 0.4
    columns = ["b0_level", "b1_slope", "b2_curvature"]
    rng = np.random.default_rng(0)
    level_slope_curve = pd.DataFrame(
        {
            "b0_level": 0.03 + rng.normal(scale=0.002, size=50),
            "b1_slope": -0.01 + rng.normal(scale=0.002, size=50),
            "b2_curvature": 0.0 + rng.normal(scale=0.002, size=50),
        }
    )

    scaler = StandardScaler()
    scaled = scaler.fit_transform(level_slope_curve)
    pca = PCA(n_components=2)
    scores = pca.fit_transform(scaled)
    factor_names = ["PC1", "PC2"]
    scores_df = pd.DataFrame(scores, columns=factor_names)
    loadings_df = pd.DataFrame(pca.components_.T, index=columns, columns=factor_names)

    pca_model = PCAFactorModel(
        scaler=scaler,
        pca=pca,
        scores=scores_df,
        loadings=loadings_df,
        explained_variance_ratio=pca.explained_variance_ratio_,
    )
    artifacts.save_pca_result(pca_model, path=tmp_path / "pca_model.pkl")

    ou_params = {
        "PC1": {"kappa": 1.0, "theta": 0.0, "sigma": 0.5},
        "PC2": {"kappa": 2.0, "theta": 0.0, "sigma": 0.3},
    }
    artifacts.save_ou_parameters(ou_params, path=tmp_path / "ou_parameters.json")

    mean_params = level_slope_curve.mean().to_dict()
    mean_params["lambda"] = fixed_lambda
    maturities = np.array([1.0, 5.0, 10.0])
    from project.calibration.sensitivities import compute_forward_sensitivities

    # unpadded scaler.scale_ (3-length, aligned with loadings_df's own
    # b0/b1/b2 index) -- exactly how the notebook calls this in the
    # fixed-lambda path; compute_forward_sensitivities pads it internally.
    pc_sens_df = compute_forward_sensitivities(
        mean_params, loadings_df, maturities, param_scale=scaler.scale_
    )
    artifacts.save_sensitivities(
        mean_params, maturities, pc_sens_df, path=tmp_path / "sensitivities.json"
    )

    model = HJMModel.from_disk(tmp_path)
    result = model.simulate(n_paths=20, T_horizon=0.1, dt=1 / 252, measure="P", random_seed=0)

    lam = result.ns_params[:, :, 3]
    assert np.allclose(lam, fixed_lambda)
