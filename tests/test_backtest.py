import numpy as np
import pandas as pd
import pytest

from project.stochastic.backtest import (
    _fit_pipeline_at_origin,
    generate_backtest_origins,
    run_backtest,
    summarize_backtest,
)


def _synthetic_yields_df(n=700, seed=0):
    """A mean-reverting synthetic curve panel -- enough rows for a 500-day
    minimum training window plus several walk-forward origins beyond it."""
    dates = pd.bdate_range("2018-01-02", periods=n)
    tenors = [0.25, 1.0, 2.0, 5.0, 10.0, 30.0]
    rng = np.random.default_rng(seed)

    level = np.zeros(n)
    for t in range(1, n):
        level[t] = level[t - 1] + 0.02 * (0.03 - level[t - 1]) + 0.0008 * rng.standard_normal()
    slope_by_tenor = np.array([0.02, 0.01, 0.005, 0.0, -0.005, -0.01])
    data = (
        0.03
        + level[:, None]
        + slope_by_tenor[None, :]
        + 0.0005 * rng.standard_normal((n, len(tenors)))
    )
    return pd.DataFrame(data, index=dates, columns=[str(t) for t in tenors])


def test_generate_backtest_origins_respects_min_train_window_and_horizon():
    df = _synthetic_yields_df(n=700)
    origins = generate_backtest_origins(df.index, min_train_window=500, step=25, horizon=60)

    assert len(origins) > 0
    for origin in origins:
        pos = df.index.get_loc(origin)
        assert pos >= 500
        assert pos + 60 <= len(df) - 1


def test_generate_backtest_origins_respects_step_spacing():
    df = _synthetic_yields_df(n=700)
    origins = generate_backtest_origins(df.index, min_train_window=500, step=25, horizon=60)
    positions = [df.index.get_loc(o) for o in origins]
    diffs = np.diff(positions)
    assert np.all(diffs == 25)


def test_generate_backtest_origins_start_end_filter():
    df = _synthetic_yields_df(n=700)
    all_origins = generate_backtest_origins(df.index, min_train_window=500, step=10, horizon=60)
    cutoff = all_origins[len(all_origins) // 2]

    filtered = generate_backtest_origins(
        df.index, start=cutoff, min_train_window=500, step=10, horizon=60
    )
    assert all(o >= cutoff for o in filtered)
    assert len(filtered) < len(all_origins)


def test_fit_pipeline_at_origin_produces_a_usable_model():
    df = _synthetic_yields_df(n=700)
    tenors = np.array([float(c) for c in df.columns])
    origin = df.index[550]

    model = _fit_pipeline_at_origin(df.loc[:origin], tenors)
    result = model.simulate(n_paths=10, T_horizon=0.1, dt=1 / 252, measure="P", random_seed=0)

    assert not np.isnan(result.zero_curves).any()
    assert result.zero_curves.shape[0] == 10


def test_run_backtest_result_is_unaffected_by_data_after_the_scored_horizon():
    """The whole point of walk-forward backtesting: refitting at an origin,
    and scoring it against the realized value `horizon_days` later, must not
    depend on data that comes after that -- otherwise the "forecast" secretly
    saw its own answer. Appending clearly-anomalous future rows (a rate
    spike) well beyond every origin's scored horizon must not change a
    single number in the result."""
    df_short = _synthetic_yields_df(n=650)
    future_dates = pd.bdate_range(df_short.index[-1] + pd.Timedelta(days=1), periods=100)
    anomalous_future = pd.DataFrame(
        0.50,
        index=future_dates,
        columns=df_short.columns,  # a rate spike no legitimate fit should ever see
    )
    df_long = pd.concat([df_short, anomalous_future])

    origins = generate_backtest_origins(df_short.index, min_train_window=500, step=50, horizon=60)
    assert len(origins) >= 2

    result_short = run_backtest(
        df_short, origins=origins, horizon_days=60, n_paths=20, random_seed=0
    )
    result_long = run_backtest(df_long, origins=origins, horizon_days=60, n_paths=20, random_seed=0)

    pd.testing.assert_frame_equal(result_short, result_long)


def test_run_backtest_and_summarize_produce_sane_output():
    df = _synthetic_yields_df(n=650)
    origins = generate_backtest_origins(df.index, min_train_window=500, step=50, horizon=60)

    results = run_backtest(df, origins=origins, horizon_days=60, n_paths=50, random_seed=0)
    assert not results.isna().any().any()
    assert set(results["tenor"]) == {0.25, 1.0, 2.0, 5.0, 10.0, 30.0}
    assert (results["band_lo"] <= results["band_hi"]).all()
    assert (results["squared_error"] >= 0).all()
    assert (results["crps"] >= 0).all()

    summary = summarize_backtest(results)
    assert (summary["coverage"] >= 0).all() and (summary["coverage"] <= 1).all()
    assert (summary["rmse"] >= 0).all()


@pytest.mark.slow
def test_backtest_coverage_is_roughly_calibrated_on_a_well_specified_process():
    """A real (if generous) statistical check: run enough origins that if the
    simulated 90% band were badly miscalibrated against this synthetic,
    known-to-be-mean-reverting process, this would catch it. Not a tight
    bound -- a handful of correlated, overlapping-window origins isn't
    enough data for a precise coverage estimate -- just enough to flag gross
    miscalibration (e.g. an accidental leak or a sign error making every
    forecast overconfident)."""
    df = _synthetic_yields_df(n=1400, seed=7)
    origins = generate_backtest_origins(df.index, min_train_window=500, step=15, horizon=40)
    results = run_backtest(df, origins=origins, horizon_days=40, n_paths=300, random_seed=0)
    summary = summarize_backtest(results)

    assert (summary["coverage"] > 0.5).all()
