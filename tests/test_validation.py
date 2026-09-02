import numpy as np
import pandas as pd

from project.stochastic.hjm_model import HJMModel, HJMModelParams
from project.stochastic.validation import compare_simulated_to_historical


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


def test_compare_simulated_to_historical_no_flags_when_close():
    model = HJMModel(_tiny_params())
    result = model.simulate(n_paths=300, T_horizon=0.1, dt=1 / 252, measure="P", random_seed=0)

    rng = np.random.default_rng(0)
    dates = pd.date_range("2020-01-01", periods=500)
    historical = pd.DataFrame(
        {
            1.0: rng.normal(0.03, 0.005, 500),
            5.0: rng.normal(0.03, 0.005, 500),
            10.0: rng.normal(0.03, 0.005, 500),
        },
        index=dates,
    )
    comparison = compare_simulated_to_historical(result, historical)
    assert not comparison["flagged"].any()


def test_compare_simulated_to_historical_flags_large_discrepancy():
    model = HJMModel(_tiny_params())
    result = model.simulate(n_paths=300, T_horizon=0.1, dt=1 / 252, measure="P", random_seed=0)

    rng = np.random.default_rng(0)
    dates = pd.date_range("2020-01-01", periods=500)
    # historical mean far from the model's ~3% mean_params, with a tight std
    historical = pd.DataFrame(
        {
            1.0: rng.normal(0.50, 0.001, 500),
            5.0: rng.normal(0.50, 0.001, 500),
            10.0: rng.normal(0.50, 0.001, 500),
        },
        index=dates,
    )
    comparison = compare_simulated_to_historical(result, historical)
    assert comparison["flagged"].all()
