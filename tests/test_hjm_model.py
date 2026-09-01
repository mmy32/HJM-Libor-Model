import numpy as np

from src.stochastic.hjm_model import HJMModel, HJMModelParams


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
