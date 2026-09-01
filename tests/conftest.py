"""Shared fixtures for synthetic pipeline data."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def tiny_maturities():
    return np.array([0.5, 1.0, 2.0, 5.0, 10.0])


@pytest.fixture
def synthetic_yield_df():
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    tenors = [0.25, 1.0, 5.0, 10.0]
    rng = np.random.default_rng(0)
    data = 2.0 + rng.normal(scale=0.1, size=(10, len(tenors)))
    return pd.DataFrame(data, index=dates, columns=tenors)


@pytest.fixture
def synthetic_ns_params_df():
    dates = pd.date_range("2020-01-01", periods=30, freq="D")
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "b0_level": 0.03 + rng.normal(scale=0.002, size=30),
            "b1_slope": -0.01 + rng.normal(scale=0.002, size=30),
            "b2_curvature": 0.0 + rng.normal(scale=0.002, size=30),
            "lambda": 0.5 + rng.normal(scale=0.01, size=30),
        },
        index=dates,
    )
