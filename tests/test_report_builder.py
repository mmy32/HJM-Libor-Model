import numpy as np
import pandas as pd
import pytest

from project.calibration.pca import PCAFactorModel
from project.reporting.report_builder import build_report, build_report_html
from project.stochastic.hjm_model import HJMModel, HJMModelParams


def _synthetic_yields_df():
    dates = pd.date_range("2020-01-01", periods=60, freq="B")
    tenors = ["0.25", "1.0", "5.0", "10.0"]
    rng = np.random.default_rng(0)
    data = 2.0 + rng.normal(scale=0.1, size=(60, len(tenors)))
    return pd.DataFrame(data, index=dates, columns=tenors)


def _synthetic_pca_model():
    dates = pd.date_range("2020-01-01", periods=60, freq="B")
    rng = np.random.default_rng(1)
    scores = pd.DataFrame(
        {"PC1": rng.normal(scale=1.0, size=60), "PC2": rng.normal(scale=0.5, size=60)},
        index=dates,
    )
    loadings = pd.DataFrame(
        {"PC1": [0.6, 0.5, -0.6], "PC2": [-0.2, 0.8, 0.5]},
        index=["b0_level", "b1_slope", "b2_curvature"],
    )
    return PCAFactorModel(
        scaler=None,
        pca=None,
        scores=scores,
        loadings=loadings,
        explained_variance_ratio=np.array([0.7, 0.2]),
    )


def _synthetic_ou_params():
    return {
        "PC1": {
            "kappa": 0.9,
            "theta": 0.0,
            "sigma": 1.0,
            "half_life_days": 194.0,
            "log_likelihood": -10.0,
            "converged": True,
        },
        "PC2": {
            "kappa": 2.5,
            "theta": 0.0,
            "sigma": 0.5,
            "half_life_days": 70.0,
            "log_likelihood": -8.0,
            "converged": True,
        },
    }


def _synthetic_pc_sens_df():
    return pd.DataFrame({"PC1": [1.0, 0.8, 0.5], "PC2": [0.5, 0.4, 0.2]}, index=[1.0, 5.0, 10.0])


def _synthetic_model():
    params = HJMModelParams(
        ou_params=_synthetic_ou_params(),
        loadings=np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.1, -0.1]]),
        pc_sensitivities=np.array([[1.0, 0.5], [0.8, 0.4], [0.5, 0.2]]),
        mean_params={"b0_level": 0.03, "b1_slope": -0.01, "b2_curvature": 0.0, "lambda": 0.5},
        maturities=np.array([1.0, 5.0, 10.0]),
    )
    return HJMModel(params)


def _build_synthetic_html(**overrides):
    kwargs = dict(
        yields_df=_synthetic_yields_df(),
        pca_model=_synthetic_pca_model(),
        ou_params=_synthetic_ou_params(),
        pc_sens_df=_synthetic_pc_sens_df(),
        model=_synthetic_model(),
        include_bayesian=False,
        n_sim_paths=10,
        # The synthetic fixture (60 rows) is far short of
        # backtest_spec.MIN_TRAIN_WINDOW_DAYS (504), so it has zero eligible
        # walk-forward origins -- the backtest section is exercised
        # separately, against real data, in the slow end-to-end test below.
        include_backtest=False,
        random_seed=0,
    )
    kwargs.update(overrides)
    return build_report_html(**kwargs)


def test_build_report_html_includes_every_section_and_renders_valid_looking_html():
    html = _build_synthetic_html()
    assert html.startswith("<!doctype html>")
    assert "</html>" in html
    for heading in [
        "What this model does",
        "raw data",
        "Principal Component Analysis",
        "mean reversion",
        "Connecting factors back to real rates",
        "simulating the future",
        "What this model doesn't do",
        "How the code is put together",
        "Package tour",
        "The pipeline, in code",
        "Where this follows good engineering practice",
    ]:
        assert heading in html
    assert html.count("data:image/png;base64,") == 5  # one figure per section that has one


def test_build_report_html_omits_bayesian_section_when_disabled():
    html = _build_synthetic_html(include_bayesian=False)
    assert "How sure are we" not in html


def test_build_report_html_omits_backtest_section_when_disabled():
    html = _build_synthetic_html(include_backtest=False)
    assert "How accurate is this, really?" not in html


def test_build_report_html_standalone_false_omits_wrapper_tags():
    """standalone=False is the shape an Artifact publish needs: a <title> +
    <style> block followed by content, with no <!doctype>/<html>/<head>/
    <body> of its own -- the Artifact tool supplies those."""
    html = _build_synthetic_html(standalone=False)
    assert not html.startswith("<!doctype html>")
    for forbidden in ("<html", "<head>", "<body>"):
        assert forbidden not in html
    assert html.startswith("<title>")
    assert "<style>" in html
    assert "What this model does" in html


@pytest.mark.slow
def test_build_report_html_bayesian_section_reports_convergence_diagnostic():
    """Real (small) MCMC fit for both synthetic factors -- slow like the
    other MCMC-backed tests in this suite, so marked accordingly."""
    html = _build_synthetic_html(
        include_bayesian=True, bayesian_draws=200, bayesian_tune=200, bayesian_chains=2
    )
    assert "How sure are we" in html
    assert "convergence" in html
    assert html.count("data:image/png;base64,") == 5  # bayesian section has a table, not a figure


def test_build_report_end_to_end_from_committed_artifacts(tmp_path):
    """Exercises the real disk-loading path (build_report, not
    build_report_html) against this repo's actual committed calibration
    artifacts -- the same data the CLI script uses -- skipping only the slow
    Bayesian refit."""
    output_path = build_report(
        output_path=tmp_path / "report.html",
        include_bayesian=False,
        n_sim_paths=20,
        include_backtest=False,
    )
    assert output_path.exists()
    html = output_path.read_text(encoding="utf-8")
    assert "What this model does" in html
    assert html.count("data:image/png;base64,") == 5


@pytest.mark.slow
def test_build_report_end_to_end_includes_backtest_against_real_data(tmp_path):
    """The backtest section needs real history (its minimum training window
    alone is longer than the whole synthetic fixture), so it's exercised
    here against the actual committed data rather than in the fast synthetic
    tests above -- with a reduced horizon set and path count to keep this
    from being as slow as a full `scripts/run_backtest.py --multi-horizon`."""
    output_path = build_report(
        output_path=tmp_path / "report.html",
        include_bayesian=False,
        n_sim_paths=20,
        include_backtest=True,
        backtest_horizons=(21, 63),
        backtest_n_paths=30,
    )
    html = output_path.read_text(encoding="utf-8")
    assert "How accurate is this, really?" in html
    assert "Skill vs. naive" in html
    assert html.count("data:image/png;base64,") == 6
