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
from datetime import datetime
from zoneinfo import ZoneInfo
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
from project.registry.curve_spec import SMOOTH_GRID_MAX, SMOOTH_GRID_N
from project.stochastic.backtest import run_backtest_across_horizons
from project.stochastic.hjm_model import HJMModel
from project.viz.backtest import build_backtest_horizon_figure
from project.viz.curves import build_ns_fit_slider_figure, build_static_curve_overview_figure
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
    ns_params_df = artifacts.load_ns_parameters()
    pca_model = artifacts.load_pca_result()
    ou_params = artifacts.load_ou_parameters()
    pc_sens_df = pd.read_csv(_paths.PC_FORWARD_SENSITIVITIES_CSV, index_col=0)
    model = HJMModel.from_disk()

    html = build_report_html(
        yields_df=yields_df,
        ns_params_df=ns_params_df,
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
    ns_params_df,
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
    standalone=True,
) -> str:
    """Pure rendering function -- takes already-loaded pipeline objects (so
    it's testable against small synthetic fixtures) and returns the report's
    full HTML as a string.

    `standalone=True` (the default, and what `reports/project_report.html`
    is generated as) wraps the content in its own `<!doctype html>`/`<html>`/
    `<head>`/`<body>` -- a complete file, openable directly in a browser with
    no host page needed. `standalone=False` instead returns just a `<title>`
    + `<style>` block followed by the body content, with none of those
    wrapper tags -- the shape an Artifact-hosted publish of this same report
    needs, since the Artifact tool supplies its own `<html>`/`<head>`/
    `<body>` skeleton and expects exactly a title and a style block at the
    top of the file it's given.
    """
    factor_names = list(ou_params.keys())
    scores_df = pca_model.scores[factor_names]
    cum_var = float(np.sum(pca_model.explained_variance_ratio[: len(factor_names)]))

    sections = [
        _section_abstract(yields_df, factor_names),
        _section_introduction(),
        _section_hjm_assumptions(),
        _section_curve_overview(yields_df, ns_params_df),
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
    sections.append(_section_codebase_overview())

    title = "HJM Term Structure Model — Project Report"
    if standalone:
        return _html_document(title, sections)
    return _page_style(title) + "\n" + _page_body(title, sections)


# ---------------------------------------------------------------------------
# Sections


def _section_abstract(yields_df, factor_names):
    tenors = sorted(float(c) for c in yields_df.columns)
    start, end = yields_df.index.min(), yields_df.index.max()
    return f"""
<section>
<h2>Abstract</h2>
<p>This report presents a Heath-Jarrow-Morton (HJM) model of the US Treasury
yield curve: a computer model of the entire curve — not just the handful of
maturities the government actually reports, but every point in between —
evolving realistically through time under a no-arbitrage constraint. The
curve's dimensionality is reduced in two stages: a Nelson-Siegel fit
compresses each day's {len(tenors)}-tenor curve (from {tenors[0]:g}-year to
{tenors[-1]:g}-year) down to 3 shape parameters, which are then rotated via
Principal Component Analysis (PCA) into <strong>{len(factor_names)}
statistically independent factors</strong>, ranked by how much of the
curve's combined movement each one explains. Each factor is modeled as a
mean-reverting (Ornstein-Uhlenbeck) process and calibrated on
<strong>{len(yields_df):,} daily observations</strong> spanning
<strong>{start.date()} to {end.date()}</strong>, then used to simulate
no-arbitrage forward paths for the full curve. Where noted, forecast
accuracy is validated by a walk-forward backtest against a held-out region
of that history, benchmarked against a naive random-walk forecast and a
standard machine-learning baseline, with results reported by horizon
rather than collapsed into one headline figure. Every figure and number in
this report is generated from the project's own currently calibrated
state, not illustrative examples.</p>
</section>
"""


def _section_introduction():
    return """
<section>
<h2>Introduction</h2>
<p>US Treasury yields are observed at only a handful of maturities — a
1-month bill, a 2-year note, a 10-year bond, and so on — yet most practical
work with interest rates (pricing, hedging, stress testing) needs the
<em>entire</em> curve, at every maturity, evolving realistically through
time. A curve model has to satisfy three requirements at once: it must fit
the maturities actually observed, stay smooth in between them, and evolve
over time without creating <strong>arbitrage</strong> — a situation where
two parts of the curve are priced inconsistently enough that a riskless
profit could be locked in just by trading between them. Simple
interpolation methods satisfy the first two requirements but say nothing
about the third: they describe a <em>shape</em>, not a process that stays
internally consistent as time passes and rates move.</p>
<p>The Heath-Jarrow-Morton (HJM) framework resolves this by modeling the
whole forward curve as a single evolving stochastic process, instead of
picking a curve shape and re-fitting it snapshot by snapshot. Its key
result: once the curve's volatility structure is specified — how much, and
in what correlated way, different maturities move — the drift each
maturity must follow is no longer a free choice. It is pinned down
directly by the no-arbitrage condition, rather than imposed as a separate
rule. What remains is a practical difficulty, not a theoretical one: the
forward curve has infinitely many maturities, so specifying an independent
volatility for each is infeasible — there isn't enough data, and naive
attempts overfit noise into an unstable model. The rest of this report
addresses that difficulty directly: reducing the curve to a small number
of interpretable factors, calibrating their dynamics against real data,
and validating the result against what the curve actually did next.</p>
</section>
"""


def _section_hjm_assumptions():
    return """
<section>
<h2>The HJM framework's assumptions</h2>
<p>Heath-Jarrow-Morton (1992) is a framework, not a single model — it fixes
what the forward curve's dynamics must satisfy, and leaves one piece as a
free modeling choice. Stated formally, the instantaneous forward rate
<em>f(t,T)</em> (the rate agreed today, at time <em>t</em>, for
instantaneous borrowing at future time <em>T</em>) evolves as:</p>
<p style="text-align:center; font-family: 'Iowan Old Style', Georgia, serif; font-size: 1.05rem;">
df(t,T) = &alpha;(t,T) dt + &sigma;(t,T) dW(t)
</p>
<p>for some drift <em>&alpha;</em>, volatility <em>&sigma;</em>, and
Brownian motion <em>W</em>. The framework's central result is a restriction
on <em>&alpha;</em>: absence of arbitrage (formally, the existence of an
equivalent martingale measure under which discounted bond prices are
martingales) forces the risk-neutral drift to be a specific function of the
volatility alone —</p>
<p style="text-align:center; font-family: 'Iowan Old Style', Georgia, serif; font-size: 1.05rem;">
&alpha;(t,T) = &sigma;(t,T) &int;<sub>t</sub><sup>T</sup> &sigma;(t,s) ds
</p>
<p>— the HJM drift condition. Practically, this means <em>&sigma;(t,T)</em>,
the curve's volatility structure, is the <em>only</em> free choice in the
entire framework: specify how volatile each maturity is and how those
maturities move together, and the no-arbitrage drift follows automatically
rather than needing to be separately imposed or checked. This project's
<code>HJMModel._hjm_drift</code> implements exactly this condition, applied
to the reconstructed forward curve under the risk-neutral (Q) measure.</p>
<p>Beyond that general framework, this implementation makes four specific,
concrete choices, each a real simplification worth stating plainly rather
than leaving implicit:</p>
<ul>
<li><strong>Finite-dimensional, not infinite-dimensional.</strong> The
fully general HJM setting allows <em>&sigma;(t,T)</em> to vary
independently at every maturity. This project reduces the curve to a
handful of Nelson-Siegel parameters and then a small number of PCA
factors first (see the sections above), and specifies volatility per
<em>factor</em> rather than per maturity directly — the practical
resolution HJM's own literature recommends, not a deviation from it.</li>
<li><strong>Deterministic, time-invariant volatility.</strong> Each
factor's <em>&sigma;</em> is a single constant (the OU volatility fitted
below), not a stochastic process or a function of time or the current
level of rates. This is the Gaussian-HJM special case — it rules out
volatility smiles and regime-dependent volatility, but keeps calibration
and simulation exact rather than requiring a stochastic-volatility
extension this project's data (spot yields only, no options) couldn't
identify anyway.</li>
<li><strong>Continuous paths, no jumps.</strong> Every factor moves via a
Brownian increment; the model has no mechanism for a discontinuous jump
(e.g. a surprise policy announcement's immediate effect), consistent with
using Gaussian (OU) factor dynamics throughout.</li>
<li><strong>Zero market price of risk.</strong> A fully general change of
measure from the real-world (P) to the risk-neutral (Q) measure allows an
additional risk-premium adjustment to each factor's drift
(<em>&lambda;<sub>risk</sub></em>), separate from the HJM drift condition
above. This project leaves it at zero — calibrating it honestly would need
derivative-price data this project doesn't have (see <code>TODO.md</code>)
— so the Q-measure drift used here comes entirely from the HJM condition
itself, not from an additionally-calibrated risk premium.</li>
</ul>
<p>How the next few sections' modeling choices — Nelson-Siegel, PCA, and an
Ornstein-Uhlenbeck process per factor — fit inside these assumptions
without contradicting them is addressed directly where each choice is
made, not asserted here in the abstract.</p>
</section>
"""


def _observed_yields_table(yields_df):
    """A single day's observed tenor grid, rendered as a plain table -- the
    literal raw input to everything downstream, before any curve fitting."""
    latest_date = yields_df.index.max()
    row = yields_df.loc[latest_date]
    ordered_cols = sorted(row.index, key=lambda c: float(c))
    headers = "".join(f"<th>{float(c):g}y</th>" for c in ordered_cols)
    cells = "".join(f"<td>{row[c] * 100:.2f}%</td>" for c in ordered_cols)
    return f"""<table>
<thead><tr><th>Date</th>{headers}</tr></thead>
<tbody><tr><td>{latest_date.date()}</td>{cells}</tr></tbody>
</table>"""


def _fig_to_plotly_html(fig) -> str:
    """Embed a Plotly figure as inline HTML+JS, with the full plotly.js
    library inlined (`include_plotlyjs=True`) rather than loaded from a CDN
    -- this report is meant to be a self-contained file that opens and
    works offline, and only one Plotly figure lives in it, so the ~4-5MB
    one-time library cost isn't paid more than once."""
    return fig.to_html(
        full_html=False,
        include_plotlyjs=True,
        config={"displayModeBar": False},
    )


def _section_curve_overview(yields_df, ns_params_df):
    tenors = np.array([float(c) for c in yields_df.columns])
    smooth_tenors = np.linspace(0, SMOOTH_GRID_MAX, SMOOTH_GRID_N)
    overview_fig = build_static_curve_overview_figure(yields_df)
    overview_img = _fig_to_data_uri(overview_fig)
    # A slider over the full daily history would be gigabytes of JSON (each
    # frame's "visible" array is one bool per frame, so size grows with the
    # square of frame count -- see build_yield_curve_slider_figure's
    # docstring for the ~960MB version of this mistake). Every 15th trading
    # day keeps the slider genuinely scrollable (~three weeks between steps)
    # while keeping this report's own file size sane.
    slider_fig = build_ns_fit_slider_figure(
        yields_df, ns_params_df, tenors, smooth_tenors, sample_every=15
    )
    slider_html = _fig_to_plotly_html(slider_fig)
    return f"""
<section>
<h2>The raw data: yield curves over time</h2>
<p>Each line in the figure further below is one day's observed Treasury
yield curve — the interest rate the government pays, at that moment, for
every maturity from a few months out to 30 years. The shape moves around:
it flattens, steepens, and occasionally inverts (short-term rates above
long-term ones) as the economy and monetary policy change. A single number,
like "the 10-year rate," misses almost all of this — which is the whole
reason for building a model of the <em>curve</em> rather than tracking one
point on it.</p>
<h3>What's actually observed</h3>
<p>We directly observe yields at only these maturities — the government
doesn't report a continuous curve, just these discrete points. Here is the
most recent trading day in this dataset:</p>
{_observed_yields_table(yields_df)}
<p>The five snapshots below are hand-picked, not evenly sampled, to show the
range of shapes this curve takes historically: the legend labels each one
with its curve shape — steep, flat, or inverted — computed directly from
that day's actual 10-year/2-year spread (the standard "2s10s" market
shorthand).</p>
<figure>
<img src="{overview_img}" alt="Observed Treasury yield curves at several points in history, each labeled with its shape">
<figcaption>Five historically representative curves, each labeled with its
shape in the legend.</figcaption>
</figure>
<h3>From discrete points to a continuous curve: Nelson-Siegel</h3>
<p>Everything downstream in this report — PCA, the mean-reversion model, the
simulator — needs a <em>continuous</em> curve, not 11 disconnected points,
and needs that curve reduced to a handful of numbers rather than treating
each tenor as an independent quantity. This project uses the Nelson-Siegel
(1987) parameterization, standard in both academic and central-bank
term-structure work, for three reasons: it collapses each day's curve to
just a few interpretable numbers instead of an arbitrary spline with no
economic meaning; it stays smooth and well-behaved by construction, unlike
unconstrained interpolation, which can swing wildly between sparse tenor
points; and its shape parameters correspond directly to how curves actually
move in practice — a level shift, a steepening/flattening tilt, and a
hump-shaped bend.</p>
<p>The Nelson-Siegel yield at maturity <em>&tau;</em> is:</p>
<p style="text-align:center; font-family: 'Iowan Old Style', Georgia, serif; font-size: 1.05rem;">
y(&tau;) = &beta;<sub>0</sub> + &beta;<sub>1</sub> &middot;
<span style="display:inline-block; vertical-align:middle; text-align:center;">
<span style="display:block; border-bottom:1px solid var(--ink); padding:0 0.3em;">1 &minus; e<sup>&minus;&lambda;&tau;</sup></span>
<span style="display:block; padding:0 0.3em;">&lambda;&tau;</span>
</span>
+ &beta;<sub>2</sub> &middot;
<span style="display:inline-block; vertical-align:middle; text-align:center;">
<span style="display:block; border-bottom:1px solid var(--ink); padding:0 0.3em;">1 &minus; e<sup>&minus;&lambda;&tau;</sup></span>
<span style="display:block; padding:0 0.3em;">&lambda;&tau;</span>
</span>
&minus; &beta;<sub>2</sub> e<sup>&minus;&lambda;&tau;</sup>
</p>
<p>where <em>&beta;<sub>0</sub></em> is the long-run level (the curve's
asymptote as maturity grows), <em>&beta;<sub>1</sub></em> is the slope
(its loading is concentrated at the short end, so it mostly moves short
rates relative to long ones), <em>&beta;<sub>2</sub></em> is the curvature
(a hump or trough loading concentrated at medium maturities), and
<em>&lambda;</em> controls how quickly the slope and curvature loadings
decay as maturity increases.</p>
<p>For a <em>fixed</em> &lambda;, this expression is linear in
&beta;<sub>0</sub>, &beta;<sub>1</sub>, &beta;<sub>2</sub> — fitting a
day's curve is an exact ordinary-least-squares regression against the
observed tenors, not an iterative search
(<code>project.curves.nelson_siegel.fit_ns_fixed_lambda</code>). This
project fixes &lambda; once, estimated from a single robust fit against the
panel's across-day average curve, rather than refitting it independently
every day: &lambda; is only weakly identified from a handful of tenor
quotes, and letting it float freely was tried first and introduced
optimizer-driven day-to-day noise that swamped genuine curve dynamics —
a real, diagnosed problem (see <code>calibrate_all_days</code>'s docstring
and <code>TODO.md</code>), not a hypothetical one.</p>
<h3>How well does it actually fit?</h3>
<p>The figure below overlays each day's fitted Nelson-Siegel curve (blue
line) against that day's actual observed points (red ×'s). Drag the slider
to scroll through this dataset's history and judge the fit quality
yourself, rather than taking "it fits well" on faith.</p>
{slider_html}
</section>
"""


_NS_PARAM_READABLE_NAMES = {
    "b0_level": "level",
    "b1_slope": "slope",
    "b2_curvature": "curvature",
    "lambda": "decay (lambda)",
}


def _dominant_loading_sentence(pca_model, factor_names):
    """A concrete, live-computed sentence naming which Nelson-Siegel
    parameter each of the first two factors loads on most heavily -- using
    this project's own calibrated loadings rather than asserting a generic
    "PC1 is level, PC2 is slope" claim that may not actually hold for the
    currently-fitted basis."""
    clauses = []
    for name in factor_names[:2]:
        column = pca_model.loadings[name]
        dominant_param = column.abs().idxmax()
        readable = _NS_PARAM_READABLE_NAMES.get(dominant_param, dominant_param)
        clauses.append(f"{name} loads most heavily on {readable} ({column[dominant_param]:+.2f})")
    return "; ".join(clauses) + "."


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
<h3>Reading the bottom two panels</h3>
<p><strong>Bottom left, the first two factors over time:</strong> mechanically,
this is nothing more than two time series plotted together — the actual
calibrated score for {factor_names[0]} and {factor_names[1]} on every day in
the dataset, produced by projecting that day's level/slope/curvature onto
the PCA basis above. Its significance is that this <em>is</em> the raw
material everything downstream in this report actually operates on: the
mean-reversion fit, the Bayesian posterior, and the HJM simulator all model
these score series directly, not the original curve. It also works as a
visual sanity check — a genuinely mean-reverting process should look like
fluctuation around a stable level, not a persistent drift, and a stretch
where the swings visibly widen is a direct look at exactly the kind of
regime change this project's OU calibration has already been found to
struggle with (see <code>TODO.md</code>).</p>
<p><strong>Bottom right, the loadings heatmap:</strong> mechanically, each
column is one factor, each row is one Nelson-Siegel parameter, and the
number in a cell is how much a one-unit move in that factor translates into
a move in that parameter — the rotation PCA found, written out in full
rather than left as an abstract transformation. Its significance is that
this is what makes "{factor_names[0]}" and "{factor_names[1]}" mean
something economically instead of being opaque statistical directions: on
this project's current calibration, {_dominant_loading_sentence(pca_model, factor_names)}
Reading a factor's dominant loading this way is how the sensitivities
section two steps from here translates a factor's movement back into an
actual change in the observed yield curve.</p>
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
<h3>The mathematics</h3>
<p>Each factor's evolution is modeled as an Ornstein-Uhlenbeck stochastic
differential equation:</p>
<p style="text-align:center; font-family: 'Iowan Old Style', Georgia, serif; font-size: 1.05rem;">
dX<sub>t</sub> = &kappa;(&theta; &minus; X<sub>t</sub>) dt + &sigma; dW<sub>t</sub>
</p>
<p>where <em>X<sub>t</sub></em> is the factor's score, <em>&theta;</em> its
long-run mean, <em>&kappa;</em> &gt; 0 how fast it reverts toward
<em>&theta;</em>, <em>&sigma;</em> its volatility, and <em>W<sub>t</sub></em> a
standard Brownian motion. Unusually for a stochastic differential equation,
this one has a closed-form solution: conditional on <em>X<sub>t</sub></em>,
<em>X<sub>t+dt</sub></em> is exactly Gaussian, with</p>
<p style="text-align:center; font-family: 'Iowan Old Style', Georgia, serif; font-size: 1.0rem;">
E[X<sub>t+dt</sub> | X<sub>t</sub>] = &theta; + (X<sub>t</sub> &minus; &theta;)e<sup>&minus;&kappa;dt</sup>
&nbsp;&nbsp;&nbsp;&nbsp;
Var[X<sub>t+dt</sub> | X<sub>t</sub>] =
<span style="display:inline-block; vertical-align:middle; text-align:center;">
<span style="display:block; border-bottom:1px solid var(--ink); padding:0 0.3em;">&sigma;<sup>2</sup></span>
<span style="display:block; padding:0 0.3em;">2&kappa;</span>
</span>
(1 &minus; e<sup>&minus;2&kappa;dt</sup>)
</p>
<p>That exact transition density is what makes calibration here an exact
maximum-likelihood optimization rather than a numerical approximation (the
Euler-Maruyama discretization most stochastic differential equations
require) — <code>project.calibration.ou_process.estimate_ou_parameters</code>
maximizes this Gaussian log-likelihood directly against each factor's
observed history.</p>
<h3>Why Ornstein-Uhlenbeck, and not something else</h3>
<p>OU is the standard choice for a mean-reverting rate factor for concrete
reasons, not just convention. Term-structure factors are empirically
mean-reverting — a displaced level, slope, or curvature factor is understood
to pull back toward a stable average rather than drift indefinitely, which
rules out a driftless process like geometric Brownian motion (the standard
choice for, say, an equity price, which has no such pull). OU is the
natural continuous-time process for exactly that behavior, and is the same
process Vasicek's (1977) equilibrium short-rate model uses for the short
rate itself — here applied to PCA factor scores instead. Its closed-form
Gaussian transition (above) is also what keeps calibration and simulation
exact rather than approximated. The one property OU deliberately gives up
is guaranteed positivity — unlike the CIR process, whose state-dependent
volatility keeps a rate from ever crossing zero — but that is not a real
concern here: PCA factor scores are rotated, mean-zero-ish coordinates that
can and do go negative by construction, not raw rates that need to stay
positive.</p>
<h3>Consistency with the no-arbitrage assumption</h3>
<p>Choosing OU for these factors does not conflict with the no-arbitrage
assumption in "The HJM framework's assumptions" above, because the two
operate at different levels. &kappa; and &theta; describe how the factors
evolve under the real-world (P) measure, used directly for forecasting and
the walk-forward backtest. The no-arbitrage constraint only ever concerns
the risk-neutral (Q) measure, and there it depends on exactly one thing
from this section: each factor's volatility &sigma;, which (combined with
the PCA loadings and Nelson-Siegel sensitivities from the previous
sections) defines the observed forward curve's volatility structure.
<code>HJMModel._hjm_drift</code> computes the arbitrage-free drift directly
from that volatility structure and applies it to the reconstructed forward
curve under Q — it never uses &kappa; or &theta; at all. So the
mean-reversion assumption is free to be whatever best describes real-world
dynamics without threatening the model's no-arbitrage property, which rests
entirely on the volatility structure being applied consistently between
what calibration measured and what the no-arbitrage drift condition
consumes. One piece is deliberately left incomplete, though, and disclosed
rather than glossed over: a full change of measure from P to Q for the
factor SDE itself would also need a market price of risk
(&lambda;<sub>risk</sub>), which this project leaves at zero (see
<code>TODO.md</code>) rather than calibrate without the derivative-price
data that would honestly require.</p>
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


def _describe_horizon_pattern(horizon_summary, column):
    """Turn a per-horizon skill column into an honest one-sentence summary of
    whatever sign pattern the backtest actually produced this run, rather
    than a fixed claim written once and left to go stale the next time the
    underlying data or calibration changes (see git history: extending the
    training sample back to 2001 flipped this column's sign at most horizons
    without anyone touching this function -- a hardcoded "negative past the
    short end" sentence would have quietly started lying)."""
    positive = [str(h) for h in horizon_summary.index if horizon_summary.loc[h, column] >= 0]
    negative = [str(h) for h in horizon_summary.index if horizon_summary.loc[h, column] < 0]
    if not negative:
        return "positive at every horizon tested"
    if not positive:
        return "negative at every horizon tested"

    def _clause(days, sign):
        noun = "horizon" if len(days) == 1 else "horizons"
        return f"{sign} at the {', '.join(days)}-day {noun}"

    return f"{_clause(positive, 'positive')} but {_clause(negative, 'negative')}"


def _describe_coverage_pattern(horizon_summary, nominal=0.90):
    """Same reasoning as `_describe_horizon_pattern`, for coverage: state
    whatever the backtest actually found relative to the nominal band,
    instead of asserting "at or above nominal" as a fact that a future run
    might not reproduce."""
    below = horizon_summary[horizon_summary["coverage"] < nominal]
    if below.empty:
        # A Wilson lower bound that clears nominal at every horizon is a
        # statistically meaningful signal, not just a numerically-high point
        # estimate that could plausibly be sampling noise around a
        # genuinely-nominal band -- worth a different sentence, since "wider
        # than it needs to be" is a real (if safer) miscalibration, not a
        # clean pass.
        if (horizon_summary["coverage_ci_lo"] > nominal).all():
            worst_lo = horizon_summary["coverage_ci_lo"].min()
            return (
                f"significantly above the nominal {nominal:.0%} at every horizon tested "
                f"(the 95% CI never dips below {worst_lo:.0%}) -- the simulated band looks "
                f"wider than it needs to be to hit {nominal:.0%}, not just luckily calibrated"
            )
        worst, best = horizon_summary["coverage"].min(), horizon_summary["coverage"].max()
        if worst == best:
            return f"at or above the nominal {nominal:.0%} at every horizon tested, holding steady at {worst:.0%}"
        return (
            f"at or above the nominal {nominal:.0%} at every horizon tested (as low as {worst:.0%})"
        )
    shortfall_horizons = [str(h) for h in below.index]
    noun = "horizon" if len(shortfall_horizons) == 1 else "horizons"
    worst = below["coverage"].min()
    return (
        f"below the nominal {nominal:.0%} at the {', '.join(shortfall_horizons)}-day {noun} "
        f"(as low as {worst:.0%}), though at or above nominal elsewhere"
    )


def _section_backtest(yields_df, horizons, n_paths, random_seed):
    horizon_summary = run_backtest_across_horizons(
        yields_df, horizons=horizons, start=TEST_START, n_paths=n_paths, random_seed=random_seed
    )
    fig = build_backtest_horizon_figure(horizon_summary)
    img = _fig_to_data_uri(fig)
    coverage_pattern = _describe_coverage_pattern(horizon_summary)
    naive_pattern = _describe_horizon_pattern(horizon_summary, "skill_vs_naive")
    ml_pattern = _describe_horizon_pattern(horizon_summary, "skill_vs_ml")

    rows = "".join(
        f"""<tr><td>{h}</td><td>{int(r['n_origins'])}</td><td>{r['rmse']:.4f}</td>
<td>{r['naive_rmse']:.4f}</td><td>{r['skill_vs_naive']:+.0%}</td>
<td>{r['ml_rmse']:.4f}</td><td>{r['skill_vs_ml']:+.0%}</td>
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
<p>Two baselines, so "the model's RMSE is 0.004" means something: a naive
random walk (tomorrow's curve equals today's), and a RandomForestRegressor
fit fresh at every origin on lagged rate levels -- a standard tabular ML
point forecaster, walk-forward disciplined the same way everything else here
is.</p>
<table>
<thead><tr><th>Horizon (days)</th><th>Origins</th><th>RMSE</th><th>Naive RMSE</th>
<th>Skill vs. naive</th><th>ML RMSE</th><th>Skill vs. ML</th>
<th>Coverage [95% CI]</th></tr></thead>
<tbody>{rows}</tbody>
</table>
<figure>
<img src="{img}" alt="Backtest RMSE and coverage across forecast horizons">
<figcaption>Left: RMSE of the model's median forecast against both
baselines, by horizon. Right: how often the realized rate actually fell
inside the simulated 90% band, with a 95% confidence interval on that rate,
against the nominal 90% line.</figcaption>
</figure>
<p>Three different questions, three different answers -- "skill vs. X" is the
fraction of RMSE the model removes relative to baseline X; the table above
pools across tenors, and the per-tenor breakdown (<code>scripts/run_backtest.py
--multi-horizon</code>) fills in the rest.</p>
<ul>
<li>The model's <strong>uncertainty quantification</strong>: coverage is
{coverage_pattern} -- see the per-horizon confidence interval in the table,
not just the point estimate, since a few dozen origins per horizon is a
small enough sample that it matters.</li>
<li>Its <strong>point forecast against a naive random walk</strong>
(tomorrow's curve equals today's): skill vs. naive is {naive_pattern}. Interest
rates are famously hard to beat with a point forecast -- close enough to a
random walk that a negative reading at some horizons isn't obviously a sign
of a broken model.</li>
<li>Its <strong>point forecast against a standard ML baseline</strong>: skill
vs. ML, pooled across tenors, is {ml_pattern}. That a parametric,
theory-informed model holds its own against a generic tabular learner
trained the same walk-forward way is informative regardless of which one
edges out the naive baseline -- but pooling across tenors can hide a mixed
per-tenor picture, so treat this as a headline, not the full result (see
<code>scripts/run_backtest.py --multi-horizon</code> for the per-tenor
breakdown).</li>
</ul>
<p>Reported as found rather than collapsed into one headline number: coverage,
skill vs. naive, and skill vs. ML are three different questions with three
different answers above, and a model's calibration, its edge over "assume
nothing changes," and its edge over a generic ML baseline don't have to
agree with each other -- picking just one of the three to report would
overstate or understate what this model can actually do. The confidence
interval on coverage also treats each origin as independent, which
consecutive overlapping-window origins aren't quite -- see
<code>TODO.md</code>.</p>
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


def _section_codebase_overview() -> str:
    package_rows = "".join(
        f"<tr><td><code>{pkg}</code></td><td>{role}</td><td>{files}</td></tr>"
        for pkg, role, files in _PACKAGE_TOUR
    )
    pipeline_steps = "".join(f"<li>{step}</li>" for step in _PIPELINE_STEPS)
    practice_items = "".join(f"<li>{item}</li>" for item in _ENGINEERING_PRACTICES)
    return f"""
<section>
<h2>How the code is put together</h2>
<p>Everything above is the model explained from the outside — what it does
and what it produces. This closing section explains it from the inside: how
the ~30 modules behind those results are organized, the sequence they run
in, and where deliberate engineering choices, not just modeling ones, are
doing work.</p>

<h3>Package tour</h3>
<p>Nine packages under <code>project/</code>, each with one job:</p>
<table>
<thead><tr><th>Package</th><th>Role</th><th>Key files</th></tr></thead>
<tbody>{package_rows}</tbody>
</table>
<p>Outside <code>project/</code>: <code>scripts/</code> holds three runnable
entry points (refresh the raw data, generate this report, run the backtest
from the command line); <code>notebooks/</code> holds one narrative notebook
that ties every package above together in execution order, each cell just a
few lines calling already-tested <code>project/</code> functions rather than
containing logic of its own; <code>tests/</code> holds 73 tests (one,
hitting the live FRED API, skipped by default; several more marked
<code>slow</code> for real MCMC or multi-origin backtest runs, which still
run by default); and <code>.github/workflows/</code> holds a scheduled CI
job that re-fetches Treasury data and fails loudly on a data-quality flag.</p>

<h3>The pipeline, in code</h3>
<p>The sequence every one of this report's numbers ultimately comes from:</p>
<ol>{pipeline_steps}</ol>

<h3>Where this follows good engineering practice</h3>
<ul>{practice_items}</ul>
</section>
"""


_PACKAGE_TOUR = [
    (
        "registry/",
        "Single source of truth for every tunable constant: FRED series codes, the "
        "maturity grid, Nelson-Siegel optimizer bounds, PCA factor count, and the "
        "backtest's train/validation/test split dates. Nothing else hardcodes these "
        "-- every consumer takes them as a default, not a hard constraint, so tests "
        "can still pass in tiny synthetic values.",
        "<code>paths.py</code>, <code>curve_spec.py</code>, <code>factor_spec.py</code>, "
        "<code>market_data.py</code>, <code>backtest_spec.py</code>",
    ),
    (
        "data_processing/",
        "Fetches raw Treasury yields from FRED and turns them into a clean daily "
        "panel: percentage-to-decimal conversion, forward-fill of short gaps, and a "
        "data-quality diagnostic (stale runs, outlier jumps via a robust "
        "median-absolute-deviation z-score).",
        "<code>loaders.py</code>, <code>cleaning.py</code>, <code>io.py</code>",
    ),
    (
        "curves/",
        "The Nelson-Siegel curve math: one canonical forward-rate/yield formula, two "
        "ways to fit it per day (a global optimizer, or -- with the decay parameter "
        "held fixed -- an exact linear regression), and a QuantLib benchmark comparing "
        "the fitted shape against an independent reference implementation.",
        "<code>nelson_siegel.py</code>, <code>quantlib_benchmark.py</code>",
    ),
    (
        "transform/",
        "Pure, stateless conversions between the three ways a curve gets represented "
        "here -- PCA scores, Nelson-Siegel parameters, and forward/zero curves -- "
        "shared by both calibration and simulation so each conversion exists exactly "
        "once.",
        "<code>representations.py</code>",
    ),
    (
        "calibration/",
        "Turns the daily NS-parameter panel into a compact statistical model: PCA "
        "factor extraction, Ornstein-Uhlenbeck mean-reversion fitting both as a point "
        "estimate and as a full MCMC posterior, the PC-to-forward-rate sensitivity "
        "chain rule, and calibration-quality diagnostics like out-of-sample PCA "
        "reconstruction error.",
        "<code>pca.py</code>, <code>ou_process.py</code>, <code>bayesian_ou.py</code>, "
        "<code>sensitivities.py</code>, <code>diagnostics.py</code>",
    ),
    (
        "stochastic/",
        "The model itself: <code>HJMModel</code>/<code>HJMModelParams</code> and the "
        "Monte Carlo simulation (plug-in and parameter-uncertainty-aware versions), a "
        "same-sample plausibility check, and the walk-forward, out-of-sample backtest "
        "behind the accuracy section above -- which also fits a naive random-walk and "
        "a RandomForestRegressor baseline at every origin, so the model's numbers have "
        "something to be compared against.",
        "<code>hjm_model.py</code>, <code>validation.py</code>, <code>backtest.py</code>",
    ),
    (
        "persistence/",
        "Thin save/load functions for every calibrated artifact under "
        "<code>data/ns_parameters/</code>. No computation happens here -- it exists "
        "purely so calibration code and file I/O stay separate.",
        "<code>artifacts.py</code>",
    ),
    (
        "viz/",
        "One file per domain (curves, PCA, OU, sensitivities, simulation, backtest), "
        "each returning a figure object rather than displaying or saving it -- the "
        "caller decides what happens to it.",
        "<code>curves.py</code>, <code>pca.py</code>, <code>ou.py</code>, "
        "<code>sensitivities.py</code>, <code>simulation.py</code>, "
        "<code>backtest.py</code>, <code>style.py</code>",
    ),
    (
        "reporting/",
        "This report generator -- pulls together every other package's output into "
        "the document you're reading.",
        "<code>report_builder.py</code>",
    ),
]

_PIPELINE_STEPS = [
    "<code>data_processing.loaders.fetch_treasury_yields</code> pulls raw yields "
    "from FRED; <code>cleaning.clean_treasury_yields</code> converts to decimal and "
    "forward-fills short gaps.",
    "<code>curves.nelson_siegel.calibrate_all_days_fixed_lambda</code> fits every "
    "day's curve to 3 Nelson-Siegel shape parameters via exact linear regression, "
    "lambda held fixed across the panel.",
    "<code>calibration.pca.fit_pca</code> rotates that 3-column panel into "
    "statistically independent factors.",
    "<code>calibration.ou_process.estimate_ou_parameters_for_factors</code> (point "
    "estimate) and/or <code>calibration.bayesian_ou.fit_bayesian_ou_for_factors</code> "
    "(full posterior via PyMC) fit each factor's mean-reversion.",
    "<code>calibration.sensitivities.compute_forward_sensitivities</code> chain-rules "
    "PCA loadings through the NS formula to get forward-rate sensitivity per factor.",
    "<code>persistence.artifacts.save_*</code> writes every artifact above to "
    "<code>data/ns_parameters/</code>; <code>HJMModel.from_disk()</code> reads them "
    "back into an in-memory <code>HJMModelParams</code>.",
    "<code>HJMModel.simulate()</code> (or "
    "<code>simulate_with_parameter_uncertainty()</code>) evolves the factors forward "
    "under the chosen measure and reconstructs a full curve at every step.",
    "<code>stochastic.backtest.run_backtest</code> repeats steps 2-7 at each of a "
    "series of historical origins, using only data available up to that origin, "
    "alongside a naive random-walk forecast and a fresh "
    "<code>RandomForestRegressor</code> baseline (<code>_fit_ml_baseline_at_origin</code>) "
    "fit the same way, then scores all three against what actually happened next -- "
    "the source of the accuracy section above.",
]

_ENGINEERING_PRACTICES = [
    "<strong>Separation of concerns enforced by directory structure</strong>, not just "
    "convention: pure computation (<code>calibration/</code>, <code>curves/</code>, "
    "<code>stochastic/</code>, <code>transform/</code>) never touches disk or draws a "
    "plot -- that's <code>persistence/</code>'s and <code>viz/</code>'s job "
    "respectively.",
    "<strong>Typed, structured state instead of loose dicts</strong>: "
    "<code>HJMModelParams</code>, <code>PCAFactorModel</code>, and "
    "<code>SimulationResult</code> are all dataclasses with named fields, not "
    "positional tuples or bare dictionaries passed between functions.",
    "<strong>Pure, dependency-injected constructors</strong>: <code>HJMModel</code> "
    "takes an in-memory <code>HJMModelParams</code> rather than reading files itself, "
    "specifically so it's unit-testable with tiny synthetic parameters -- "
    "<code>HJMModel.from_disk()</code> is a convenience classmethod layered on top, "
    "not the only way in.",
    "<strong>Layered, honest testing</strong>: fast synthetic-fixture unit tests are "
    "the default; a small number of tests marked <code>slow</code> run real MCMC "
    "sampling or a real multi-origin backtest and check statistical properties (e.g. "
    "credible-interval coverage across replications) rather than trusting a single "
    "lucky seed; a dedicated regression test exists specifically to catch a backtest "
    "accidentally leaking future data into a past forecast.",
    "<strong>Mistakes are documented, not hidden</strong>: <code>TODO.md</code> and "
    "several module docstrings record approaches that were tried and rejected (a "
    "day-to-day curve-smoothing penalty), and bugs that were found and fixed after "
    "they'd already produced a wrong number -- including the simulator's missing "
    "initial-state bug that this report's own backtest section exists partly to have "
    "caught.",
    "<strong>Tooling</strong>: pre-commit runs Black and basic hygiene checks (trailing "
    "whitespace, end-of-file, a large-file guard) on every commit; DVC tracks the raw "
    "data file so it can change without bloating git history; a scheduled GitHub "
    "Actions job keeps the data fresh and flags quality problems automatically.",
    '<strong>Baseline choices are specific, not "add some ML"</strong>: the backtest\'s '
    "random forest was picked over gradient boosting explicitly because it needs "
    "fewer hyperparameters tuned well to be a fair baseline with only a few hundred "
    "training rows and no separate validation split -- see "
    "<code>_fit_ml_baseline_at_origin</code>'s docstring for the full reasoning.",
    "<strong>Labels stay honest by construction, not by discipline</strong>: the "
    '"Steep/Flat/Inverted" word in the yield-curve figure\'s legend is computed from '
    "each snapshot's actual 10-year/2-year spread at render time "
    "(<code>viz.curves._classify_curve_shape</code>), not hand-typed -- so a future "
    "data refresh can't silently leave a stale shape label attached to a curve that "
    "no longer has that shape.",
]


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


def _page_style(title) -> str:
    """`<title>` + `<style>` only -- the piece that goes in `<head>` for the
    standalone document, or at the very top of the file for an Artifact
    publish (which supplies its own `<html>`/`<head>`/`<body>` wrapper and
    expects exactly this: a title tag and a style block, nothing more, at
    the top of the file)."""
    return f"""<title>{title}</title>
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
h1, h2, h3, th, .subtitle, figcaption, footer {{
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
h3 {{ font-size: 1.05rem; margin: 1.8rem 0 0.6rem; color: var(--ink); }}
p {{ margin: 0.8rem 0; }}
ul, ol {{ padding-left: 1.3rem; }}
li {{ margin: 0.5rem 0; }}
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
td:nth-child(2), td:nth-child(3) {{ text-align: left; }}
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
</style>"""


def _page_body(title, sections) -> str:
    """The `<div class="wrap">...</div>` content -- everything that goes
    inside `<body>` for the standalone document, or directly after
    `_page_style()`'s output for an Artifact publish."""
    # US Eastern, not UTC -- and %Z (not a hardcoded "EST") so this reads
    # correctly whether the report happens to be generated during Eastern
    # Standard or Eastern Daylight Time.
    generated_at = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M %Z")
    commit = _git_commit_hash()
    body = "\n".join(sections)
    return f"""<div class="wrap">
<h1>{title}</h1>
<p class="subtitle">A plain-language walkthrough of the model and what its currently
calibrated parameters actually are, generated {generated_at} directly from this
repository's data.</p>
{body}
<footer>
Generated {generated_at} from git commit <code>{commit}</code>. Regenerate with
<code>python scripts/generate_report.py</code> after re-running the pipeline notebook.
</footer>
</div>"""


def _html_document(title, sections) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
{_page_style(title)}
</head>
<body>
{_page_body(title, sections)}
</body>
</html>
"""
