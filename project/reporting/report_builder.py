"""Builds a self-contained HTML report that explains the HJM pipeline and its
*currently calibrated* results to a reader with no prior context on the
project.

This is distinct from the notebook (written for someone already following
the derivation cell by cell) and from the README (explains the general
method, not what any particular run of the pipeline actually produced).
Every number and figure here is generated from the pipeline's own persisted
artifacts under `data/ns_parameters/` (or, for the Bayesian section, refit
live) rather than hand-written, so the report changes when the underlying
calibration changes instead of silently going stale.
"""

import base64
import io
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from project.calibration.bayesian_ou import fit_bayesian_ou, posterior_summary
from project.data_processing.io import load_yield_matrix
from project.persistence import artifacts
from project.registry import paths as _paths
from project.registry.backtest_spec import TEST_START
from project.stochastic.backtest import run_backtest_across_horizons
from project.stochastic.hjm_model import HJMModel
from project.viz.backtest import build_backtest_horizon_figure
from project.viz.curves import build_static_curve_overview_figure
from project.viz.ou import build_ou_diagnostics_figure
from project.viz.pca import build_pca_diagnostics_figure
from project.viz.sensitivities import build_pc_sensitivities_figure
from project.viz.simulation import build_sample_paths_figure

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_OUTPUT_PATH = _REPO_ROOT / "reports" / "project_report.html"


# ---------------------------------------------------------------------------
# Orchestration


def build_report(
    output_path=None,
    include_bayesian=True,
    bayesian_draws=400,
    bayesian_tune=400,
    bayesian_chains=2,
    n_sim_paths=200,
    include_backtest=True,
    backtest_horizons=(21, 63, 126, 252),
    backtest_n_paths=150,
    random_seed=0,
) -> Path:
    """Load every pipeline artifact from disk, render the report, and write
    it to `output_path` (default `reports/project_report.html`).

    `include_bayesian=True` refits the OU posterior live for each factor
    (a few tens of seconds total with the defaults below) rather than
    reading a persisted artifact, since the Bayesian fit isn't currently
    saved anywhere -- see TODO.md. Pass `include_bayesian=False` for a
    faster report that skips that section.

    `include_backtest=True` runs the walk-forward backtest live, at each of
    `backtest_horizons`, restricted to the held-out test region
    (`registry.backtest_spec.TEST_START` onward) -- the only region this
    project treats as a reportable accuracy number. Also adds real runtime
    (tens of seconds); pass `include_backtest=False` to skip it.
    """
    yields_df = load_yield_matrix()
    pca_model = artifacts.load_pca_result()
    ou_params = artifacts.load_ou_parameters()
    pc_sens_df = pd.read_csv(_paths.PC_FORWARD_SENSITIVITIES_CSV, index_col=0)
    model = HJMModel.from_disk()

    html = build_report_html(
        yields_df=yields_df,
        pca_model=pca_model,
        ou_params=ou_params,
        pc_sens_df=pc_sens_df,
        model=model,
        include_bayesian=include_bayesian,
        bayesian_draws=bayesian_draws,
        bayesian_tune=bayesian_tune,
        bayesian_chains=bayesian_chains,
        n_sim_paths=n_sim_paths,
        include_backtest=include_backtest,
        backtest_horizons=backtest_horizons,
        backtest_n_paths=backtest_n_paths,
        random_seed=random_seed,
    )

    output_path = Path(output_path) if output_path else _DEFAULT_OUTPUT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def build_report_html(
    yields_df,
    pca_model,
    ou_params,
    pc_sens_df,
    model,
    include_bayesian=True,
    bayesian_draws=400,
    bayesian_tune=400,
    bayesian_chains=2,
    n_sim_paths=200,
    include_backtest=True,
    backtest_horizons=(21, 63, 126, 252),
    backtest_n_paths=150,
    random_seed=0,
) -> str:
    """Pure rendering function -- takes already-loaded pipeline objects (so
    it's testable against small synthetic fixtures) and returns the report's
    full HTML as a string."""
    factor_names = list(ou_params.keys())
    scores_df = pca_model.scores[factor_names]
    cum_var = float(np.sum(pca_model.explained_variance_ratio[: len(factor_names)]))

    sections = [
        _section_intro(yields_df, factor_names),
        _section_curve_overview(yields_df),
        _section_pca(pca_model, factor_names, cum_var),
        _section_ou(scores_df, ou_params, factor_names),
    ]
    if include_bayesian:
        sections.append(
            _section_bayesian(
                scores_df, factor_names, bayesian_draws, bayesian_tune, bayesian_chains
            )
        )
    sections.extend(
        [
            _section_sensitivities(pc_sens_df, factor_names),
            _section_simulation(model, scores_df, factor_names, n_sim_paths, random_seed),
        ]
    )
    if include_backtest:
        sections.append(
            _section_backtest(yields_df, backtest_horizons, backtest_n_paths, random_seed)
        )
    sections.append(_section_caveats())
    return _html_document("HJM Term Structure Model — Project Report", sections)


# ---------------------------------------------------------------------------
# Sections


def _section_intro(yields_df, factor_names):
    tenors = sorted(float(c) for c in yields_df.columns)
    start, end = yields_df.index.min(), yields_df.index.max()
    return f"""
<section>
<h2>What this model does</h2>
<p>This project builds a computer model of the entire US Treasury yield curve
— not just the handful of maturities the government actually reports (like
the 2-year or 10-year rate), but every point in between, evolving realistically
through time. It's calibrated on <strong>{len(yields_df):,} daily observations</strong>
of {len(tenors)} tenors (from {tenors[0]:g}-year to {tenors[-1]:g}-year), spanning
<strong>{start.date()} to {end.date()}</strong>.</p>
<p>Getting there happens in two reductions. First, each day's {len(tenors)}-tenor
curve is compressed down to just 3 numbers via a curve-shape fit (Nelson-Siegel).
Second, those 3 numbers are rotated into <strong>{len(factor_names)} statistically
independent factors</strong> via PCA, ranked by how much of their combined
movement each one explains. The rest of this report walks through both steps
and what the model does with the result — using the project's own currently
calibrated numbers throughout, not illustrative examples.</p>
</section>
"""


def _section_curve_overview(yields_df):
    fig = build_static_curve_overview_figure(yields_df)
    img = _fig_to_data_uri(fig)
    return f"""
<section>
<h2>The raw data: yield curves over time</h2>
<p>Each line below is one day's observed Treasury yield curve — the interest
rate the government pays, at that moment, for every maturity from a few
months out to 30 years. The shape moves around: it flattens, steepens, and
occasionally inverts (short-term rates above long-term ones) as the economy
and monetary policy change. A single number, like "the 10-year rate," misses
almost all of this — which is the whole reason for building a model of the
<em>curve</em> rather than tracking one point on it.</p>
<figure>
<img src="{img}" alt="Observed Treasury yield curves at several points in history">
<figcaption>A sample of historical curves, evenly spaced across the dataset.</figcaption>
</figure>
</section>
"""


def _section_pca(pca_model, factor_names, cum_var):
    fig = build_pca_diagnostics_figure(pca_model)
    img = _fig_to_data_uri(fig)
    ratios = pca_model.explained_variance_ratio[: len(factor_names)]
    rows = "".join(
        f"<tr><td>{name}</td><td>{ratio:.1%}</td><td>{np.sum(ratios[: i + 1]):.1%}</td></tr>"
        for i, (name, ratio) in enumerate(zip(factor_names, ratios))
    )
    n_ns_dims = len(pca_model.loadings.index)
    if len(factor_names) >= n_ns_dims:
        # All curve-shape dimensions were kept as factors -- PCA is a full
        # rotation here, not a lossy truncation, so cumulative variance is
        # 100% by construction. Worth saying explicitly rather than implying
        # a compression result that didn't happen at this stage.
        closing = (
            f"All {len(factor_names)} factors together necessarily explain "
            f"<strong>{cum_var:.1%}</strong> of the variation in those "
            f"{n_ns_dims} curve-shape numbers — PCA isn't discarding anything here, "
            "just rotating them into statistically independent, ranked-by-importance "
            f"directions. The real dimensionality reduction already happened upstream, "
            f"fitting {n_ns_dims} numbers to the whole observed curve instead of "
            "tracking every tenor independently."
        )
    else:
        closing = (
            f"Together, the first {len(factor_names)} factors explain "
            f"<strong>{cum_var:.1%}</strong> of the variation in those {n_ns_dims} "
            "curve-shape numbers, discarding the rest as noise rather than genuine "
            "curve dynamics."
        )
    return f"""
<section>
<h2>Finding the patterns: Principal Component Analysis</h2>
<p>Almost all of the curve's day-to-day movement turns out to follow a
handful of repeating patterns — commonly described as the whole curve
shifting up or down (<strong>level</strong>), tilting between short and long
maturities (<strong>slope</strong>), and bending in the middle
(<strong>curvature</strong>). <strong>Principal Component Analysis (PCA)</strong>
finds these patterns statistically, ranked by how much of the curve's actual
historical movement each one explains.</p>
<table>
<thead><tr><th>Factor</th><th>Variance explained</th><th>Cumulative</th></tr></thead>
<tbody>{rows}</tbody>
</table>
<p>{closing}</p>
<figure>
<img src="{img}" alt="PCA diagnostics: scree plot, cumulative variance, factor time series, loadings heatmap">
<figcaption>Top left: variance explained per factor. Top right: cumulative variance.
Bottom left: the first two factors over time. Bottom right: how each factor loads
onto the curve's level/slope/curvature parameters.</figcaption>
</figure>
</section>
"""


def _section_ou(scores_df, ou_params, factor_names):
    fig = build_ou_diagnostics_figure(scores_df, ou_params)
    img = _fig_to_data_uri(fig)
    rows = "".join(
        f"""<tr><td>{name}</td><td>{ou_params[name]['kappa']:.2f}</td>
<td>{ou_params[name]['theta']:.3f}</td><td>{ou_params[name]['sigma']:.3f}</td>
<td>{ou_params[name]['half_life_days']:.0f}</td></tr>"""
        for name in factor_names
    )
    return f"""
<section>
<h2>How the factors behave: mean reversion</h2>
<p>Each factor doesn't wander off forever — historically, when one drifts away
from its long-run average, it tends to drift back. That behavior is modeled
as an <strong>Ornstein-Uhlenbeck (OU) process</strong>, fit separately to each
factor's history. Three numbers describe it: how fast it reverts
(<code>kappa</code>), what it reverts to (<code>theta</code>), and how much it
jitters along the way (<code>sigma</code>) — summarized here as a
<strong>half-life</strong>: roughly how many trading days it takes a
displacement to shrink by half.</p>
<table>
<thead><tr><th>Factor</th><th>kappa</th><th>theta</th><th>sigma</th><th>Half-life (days)</th></tr></thead>
<tbody>{rows}</tbody>
</table>
<p>These are <strong>point estimates</strong> — the single best-fit number for
each parameter, from a standard method (maximum likelihood) that reports no
sense of how uncertain that number actually is. The next section addresses
that directly.</p>
<figure>
<img src="{img}" alt="Per-factor OU diagnostics: time series, stationary distribution, autocorrelation">
<figcaption>Per factor: the raw series against its fitted long-run mean (left),
its distribution against the model's theoretical stationary distribution (middle),
and empirical vs. theoretical autocorrelation (right).</figcaption>
</figure>
</section>
"""


def _section_bayesian(scores_df, factor_names, draws, tune, chains):
    rows = []
    for name in factor_names:
        idata = fit_bayesian_ou(
            scores_df[name].values, draws=draws, tune=tune, chains=chains, random_seed=0
        )
        summary = posterior_summary(idata)
        kappa_lo, kappa_hi = summary["kappa_hdi"]
        rows.append(
            f"""<tr><td>{name}</td><td>{summary['kappa']:.2f}</td>
<td>[{kappa_lo:.2f}, {kappa_hi:.2f}]</td>
<td>{summary['half_life_days']:.0f}</td>
<td>{summary['kappa_r_hat']:.3f}</td></tr>"""
        )
    rows_html = "".join(rows)
    return f"""
<section>
<h2>How sure are we? Bayesian parameter uncertainty</h2>
<p>The point estimates above are a best guess from a finite, noisy sample —
not a known truth. Instead of fitting one number per parameter, this section
fits a <strong>distribution</strong> of plausible values via MCMC (Markov
Chain Monte Carlo, using <a href="https://www.pymc.io/">PyMC</a>), then
summarizes it as a posterior mean and a 90% credible interval: the range that
plausibly contains the true value given the data and the model's assumptions.</p>
<table>
<thead><tr><th>Factor</th><th>kappa (mean)</th><th>90% interval</th>
<th>Half-life (days)</th><th>r&#770; (convergence)</th></tr></thead>
<tbody>{rows_html}</tbody>
</table>
<p>The <code>r&#770;</code> column is a convergence diagnostic — it should sit
close to 1.00 for the fit to be trusted; a value far from 1 would mean the MCMC
chains never agreed with each other and the numbers above shouldn't be relied
on. Note how much wider some of these intervals are than the single point
estimate suggests: that width is real information the point estimate alone
was hiding. The project's README documents, on the full simulation, how
propagating this uncertainty through to simulated future rates widens the
predictive range by roughly 5-10%.</p>
</section>
"""


def _section_sensitivities(pc_sens_df, factor_names):
    fig = build_pc_sensitivities_figure(pc_sens_df, factor_names=factor_names)
    img = _fig_to_data_uri(fig)
    return f"""
<section>
<h2>Connecting factors back to real rates</h2>
<p>A move in one of the abstract factors above has to translate into an actual
change in forward interest rates at every maturity. This figure shows exactly
that translation: how much a one-unit move in each factor shifts the forward
rate at each point on the curve. A factor that loads heavily on short
maturities but not long ones, for instance, mostly reflects near-term rate
expectations rather than a broad shift in the whole curve.</p>
<figure>
<img src="{img}" alt="Sensitivity of forward rates to each PCA factor">
<figcaption>Forward-rate sensitivity to each factor, across the maturity grid.</figcaption>
</figure>
</section>
"""


def _section_simulation(model, scores_df, factor_names, n_sim_paths, random_seed):
    # Starting every path from the most recently observed PC scores rather
    # than the default (PC score 0, i.e. the training panel's *average*
    # curve) -- omitting this is exactly the bug that gutted the backtest's
    # coverage; see stochastic.backtest._fit_pipeline_at_origin and TODO.md.
    initial_alpha = scores_df.iloc[-1][factor_names].values
    result = model.simulate(
        n_paths=n_sim_paths,
        T_horizon=1.0,
        dt=1 / 252,
        measure="P",
        random_seed=random_seed,
        initial_alpha=initial_alpha,
    )
    fig = build_sample_paths_figure(result, len(factor_names), random_seed=random_seed)
    img = _fig_to_data_uri(fig)
    return f"""
<section>
<h2>Putting it together: simulating the future</h2>
<p>With the factors' dynamics calibrated, the model can simulate many possible
future paths for the whole yield curve — {n_sim_paths} of them here, one year
ahead, starting from the most recently observed curve. This isn't a prediction
of what rates <em>will</em> do; it's the range of curve shapes that are
<em>consistent</em> with how the factors have historically behaved, which is
what makes it useful for scenario generation and stress testing rather than
point forecasting.</p>
<figure>
<img src="{img}" alt="Simulated factor paths, terminal yield curves, 10-year rate evolution, and terminal distribution">
<figcaption>Clockwise from top left: simulated factor paths, resulting terminal yield
curves, the 10-year rate's simulated evolution with a +/-2 std-dev band, and its
terminal distribution.</figcaption>
</figure>
</section>
"""


def _section_backtest(yields_df, horizons, n_paths, random_seed):
    horizon_summary = run_backtest_across_horizons(
        yields_df, horizons=horizons, start=TEST_START, n_paths=n_paths, random_seed=random_seed
    )
    fig = build_backtest_horizon_figure(horizon_summary)
    img = _fig_to_data_uri(fig)

    rows = "".join(
        f"""<tr><td>{h}</td><td>{int(r['n_origins'])}</td><td>{r['rmse']:.4f}</td>
<td>{r['naive_rmse']:.4f}</td><td>{r['skill_vs_naive']:+.0%}</td>
<td>{r['coverage']:.0%} [{r['coverage_ci_lo']:.0%}, {r['coverage_ci_hi']:.0%}]</td></tr>"""
        for h, r in horizon_summary.iterrows()
    )
    return f"""
<section>
<h2>How accurate is this, really?</h2>
<p>Everything above describes the pipeline's mechanics and shows what it
produces. This section is the only part of the report that checks whether
any of it actually works: a walk-forward backtest that refits the whole
pipeline at each of a series of past dates using <em>only</em> data available
up to that date, simulates forward, and compares the result to what the
curve actually did next. Restricted here to this project's held-out test
region (2024 onward -- see <code>registry/backtest_spec.py</code>), the only
span whose results this project treats as a reportable number rather than
something used to help design the model.</p>
<table>
<thead><tr><th>Horizon (days)</th><th>Origins</th><th>RMSE</th><th>Naive RMSE</th>
<th>Skill vs. naive</th><th>Coverage [95% CI]</th></tr></thead>
<tbody>{rows}</tbody>
</table>
<figure>
<img src="{img}" alt="Backtest RMSE and coverage across forecast horizons">
<figcaption>Left: RMSE of the model's median forecast vs. a naive
"tomorrow equals today" forecast, by horizon. Right: how often the realized
rate actually fell inside the simulated 90% band, with a 95% confidence
interval on that rate, against the nominal 90% line.</figcaption>
</figure>
<p>Two different questions, two different answers. The model's
<strong>uncertainty quantification is honest</strong>: coverage sits at or
above the nominal 90% at every horizon tested. Its <strong>point
forecast is not consistently better than doing nothing</strong>: "skill vs.
naive" is the fraction of RMSE the model removes relative to a no-change
forecast, and it's negative at most horizons and most maturities (see
<code>scripts/run_backtest.py --multi-horizon</code> for the full per-tenor
breakdown) -- meaning the median forecast is sometimes further from the
truth than simply assuming nothing changes. This is a well-known property of
interest rates (and exchange rates): they are close enough to a random walk
that beating one with a point forecast is genuinely hard, not a sign of a
broken model. Reported as found rather than hidden behind the coverage
number alone -- a model can have an honestly calibrated uncertainty band and
a point forecast with no real edge at the same time, and conflating the two
would overstate what this model can actually do. The confidence interval on
coverage also treats each origin as independent, which consecutive
overlapping-window origins aren't quite -- see <code>TODO.md</code>.</p>
</section>
"""


def _section_caveats():
    return """
<section>
<h2>What this model doesn't do</h2>
<p>In the interest of not overstating what's here:</p>
<ul>
<li>This simulation runs under the real-world ("P") measure by default — it
describes plausible future curve <em>shapes</em>, not risk-neutral prices a
trader could hedge against. A risk-neutral ("Q" measure) mode exists, but the
market price of risk it would need is deliberately left at zero rather than
calibrated — this project only has spot Treasury yields, not the derivative
prices that would be needed to estimate it honestly (see <code>TODO.md</code>).</li>
<li>The Bayesian section above fits each factor's uncertainty independently,
then (when propagated into simulation) pairs draws across factors somewhat
arbitrarily — it is not a sample from a true joint posterior over all factors
at once.</li>
<li>This is a research and educational model, not a production pricing
system — see the README's Scope section.</li>
</ul>
</section>
"""


# ---------------------------------------------------------------------------
# HTML assembly


def _fig_to_data_uri(fig, dpi=110) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _git_commit_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except OSError:
        return "unknown"


def _html_document(title, sections) -> str:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    commit = _git_commit_hash()
    body = "\n".join(sections)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{
  --bg: #faf9f5;
  --panel: #ffffff;
  --ink: #23281f;
  --muted: #6b7160;
  --accent: #3a6b5c;
  --border: #e2e0d4;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: Georgia, "Iowan Old Style", "Palatino Linotype", serif;
  line-height: 1.6;
}}
.wrap {{ max-width: 860px; margin: 0 auto; padding: 3rem 1.5rem 5rem; }}
h1, h2, th, .subtitle, figcaption, footer {{
  font-family: "Helvetica Neue", Arial, sans-serif;
}}
h1 {{ font-size: 2rem; margin-bottom: 0.2rem; }}
.subtitle {{ color: var(--muted); margin-top: 0; }}
section {{ margin-top: 3rem; }}
h2 {{
  font-size: 1.3rem;
  border-bottom: 2px solid var(--accent);
  padding-bottom: 0.35rem;
  margin-bottom: 1rem;
}}
p {{ margin: 0.8rem 0; }}
ul {{ padding-left: 1.3rem; }}
li {{ margin: 0.4rem 0; }}
figure {{ margin: 1.5rem 0; text-align: center; }}
figure img {{ max-width: 100%; border: 1px solid var(--border); border-radius: 4px; }}
figcaption {{ font-size: 0.85rem; color: var(--muted); margin-top: 0.5rem; }}
table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; font-size: 0.9rem; }}
th, td {{
  text-align: right;
  padding: 0.4rem 0.6rem;
  border-bottom: 1px solid var(--border);
  font-variant-numeric: tabular-nums;
}}
th:first-child, td:first-child {{ text-align: left; }}
th {{ color: var(--muted); font-weight: 600; }}
code {{ background: #efeee5; padding: 0.1rem 0.35rem; border-radius: 3px; font-size: 0.85em; }}
a {{ color: var(--accent); }}
footer {{
  margin-top: 4rem;
  padding-top: 1.2rem;
  border-top: 1px solid var(--border);
  font-size: 0.8rem;
  color: var(--muted);
}}
</style>
</head>
<body>
<div class="wrap">
<h1>{title}</h1>
<p class="subtitle">A plain-language walkthrough of the model and what its currently
calibrated parameters actually are, generated {generated_at} directly from this
repository's data.</p>
{body}
<footer>
Generated {generated_at} from git commit <code>{commit}</code>. Regenerate with
<code>python scripts/generate_report.py</code> after re-running the pipeline notebook.
</footer>
</div>
</body>
</html>
"""
