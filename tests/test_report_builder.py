import numpy as np
import pandas as pd
import pytest

from project.calibration.pca import PCAFactorModel
from project.reporting.report_builder import (
    _describe_coverage_pattern,
    _describe_horizon_pattern,
    build_report,
    build_report_html,
)
from project.stochastic.hjm_model import HJMModel, HJMModelParams


def _synthetic_yields_df():
    dates = pd.date_range("2020-01-01", periods=60, freq="B")
    tenors = ["0.25", "1.0", "5.0", "10.0"]
    rng = np.random.default_rng(0)
    data = 2.0 + rng.normal(scale=0.1, size=(60, len(tenors)))
    return pd.DataFrame(data, index=dates, columns=tenors)


def _synthetic_ns_params_df():
    dates = pd.date_range("2020-01-01", periods=60, freq="B")
    rng = np.random.default_rng(2)
    return pd.DataFrame(
        {
            "b0_level": 0.02 + rng.normal(scale=0.002, size=60),
            "b1_slope": -0.01 + rng.normal(scale=0.002, size=60),
            "b2_curvature": rng.normal(scale=0.002, size=60),
            "lambda": np.full(60, 0.4),
        },
        index=dates,
    )


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
        ns_params_df=_synthetic_ns_params_df(),
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
        "Abstract",
        "Introduction",
        "HJM framework's assumptions",
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
    assert "plotly-graph-div" in html  # the interactive NS-fit slider
    # 5 matplotlib PNGs (one figure per section that has one) + 2 that come
    # from plotly.js's own bundled icons, embedded once with the slider.
    assert html.count("data:image/png;base64,") == 7


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
    assert "Abstract" in html


@pytest.mark.slow
def test_build_report_html_bayesian_section_reports_convergence_diagnostic():
    """Real (small) MCMC fit for both synthetic factors -- slow like the
    other MCMC-backed tests in this suite, so marked accordingly."""
    html = _build_synthetic_html(
        include_bayesian=True, bayesian_draws=200, bayesian_tune=200, bayesian_chains=2
    )
    assert "How sure are we" in html
    assert "convergence" in html
    # 5 matplotlib PNGs (bayesian section has a table, not a figure) + 2 from
    # plotly.js's own bundled icons, embedded once with the NS-fit slider.
    assert html.count("data:image/png;base64,") == 7


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
    assert "Abstract" in html
    assert "Introduction" in html
    assert "plotly-graph-div" in html  # the interactive NS-fit slider
    assert html.count("data:image/png;base64,") == 7  # 5 matplotlib + 2 plotly.js icons


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
    assert html.count("data:image/png;base64,") == 8  # 6 matplotlib + 2 plotly.js icons


def _horizon_summary(**columns):
    return pd.DataFrame(columns, index=pd.Index([21, 63, 126, 252], name="horizon_days"))


def test_describe_horizon_pattern_all_positive():
    summary = _horizon_summary(skill_vs_naive=[0.02, 0.08, 0.30, 0.10])
    assert (
        _describe_horizon_pattern(summary, "skill_vs_naive") == "positive at every horizon tested"
    )


def test_describe_horizon_pattern_all_negative():
    summary = _horizon_summary(skill_vs_naive=[-0.02, -0.08, -0.30, -0.10])
    assert (
        _describe_horizon_pattern(summary, "skill_vs_naive") == "negative at every horizon tested"
    )


def test_describe_horizon_pattern_mixed_names_the_actual_horizons_on_each_side():
    # Regression case: this is exactly the real pattern the extended-history
    # backtest produced (negative only at the shortest horizon) -- a
    # hardcoded "negative past the short end" sentence silently stopped
    # being true here, which is the bug this dynamic helper replaced.
    summary = _horizon_summary(skill_vs_naive=[-0.17, 0.02, 0.08, 0.30])
    result = _describe_horizon_pattern(summary, "skill_vs_naive")
    assert "positive at the 63, 126, 252-day horizons" in result
    assert "negative at the 21-day horizon" in result
    assert "21-day horizons" not in result


def test_describe_coverage_pattern_all_at_or_above_nominal():
    # coverage_ci_lo dips below nominal at some horizons, so this shouldn't
    # trip the "significantly above nominal" branch -- just the plain one.
    summary = _horizon_summary(
        coverage=[1.0, 1.0, 0.95, 0.98], coverage_ci_lo=[0.85, 0.85, 0.80, 0.82]
    )
    result = _describe_coverage_pattern(summary, nominal=0.90)
    assert result.startswith("at or above the nominal 90%")
    assert "95%" in result  # the worst (lowest) horizon's coverage


def test_describe_coverage_pattern_avoids_as_low_as_phrasing_when_theres_no_range():
    # Regression case: flat 100% coverage, but the CI lower bound also dips
    # to nominal at some horizon -- not statistically distinguishable from a
    # genuinely-calibrated band, so this is still the plain branch, just
    # without the "as low as 100%" phrasing (which reads as a contradiction
    # when there's no actual range).
    summary = _horizon_summary(coverage=[1.0, 1.0, 1.0, 1.0], coverage_ci_lo=[0.85] * 4)
    result = _describe_coverage_pattern(summary, nominal=0.90)
    assert result == "at or above the nominal 90% at every horizon tested, holding steady at 100%"


def test_describe_coverage_pattern_flags_statistically_significant_overcoverage():
    # Regression case: this is exactly what the real extended-history
    # backtest produced -- 100% coverage with a Wilson CI that never dips
    # back down to nominal. That's not just a numerically-high point
    # estimate that could be noise around a genuinely-90% band; it's a real
    # signal the band is wider than it needs to be, and deserves its own
    # sentence rather than being folded into "at or above nominal."
    summary = _horizon_summary(
        coverage=[1.0, 1.0, 1.0, 1.0], coverage_ci_lo=[0.99, 0.99, 0.98, 0.99]
    )
    result = _describe_coverage_pattern(summary, nominal=0.90)
    assert result.startswith("significantly above the nominal 90%")
    assert "98%" in result  # the worst (lowest) CI lower bound
    assert "wider than it needs to be" in result


def test_describe_coverage_pattern_flags_the_shortfall_horizon():
    summary = _horizon_summary(coverage=[0.60, 0.95, 0.98, 1.0])
    result = _describe_coverage_pattern(summary, nominal=0.90)
    assert "below the nominal 90%" in result
    assert "21-day horizon" in result
    assert "21-day horizons" not in result
    assert "60%" in result
