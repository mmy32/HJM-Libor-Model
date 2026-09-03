"""Curve-related figures: raw yield curve slider, NS-fit slider, static overview.

Figures are returned, never shown or saved here -- the caller decides.
"""

import textwrap

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from matplotlib.patches import FancyBboxPatch
from matplotlib.transforms import Bbox

from project.curves.nelson_siegel import nelson_siegel_yield


def build_yield_curve_slider_figure(df, sample_every=10):
    """Interactive flipbook of raw observed yield curves, one frame per date.

    `sample_every` thins the date axis before building frames: each frame's
    `visible` array is one bool per frame, so the figure's JSON size grows
    quadratically with frame count (O(n^2), not O(n)) -- harmless at a few
    hundred dates, but a full daily panel spanning decades (thousands of
    rows) turns this into a multi-hundred-megabyte notebook cell. Subsampling
    keeps the flipbook's resolution reasonable (weekly-ish by default) without
    losing the ability to see how the curve evolved.
    """
    df_sampled = df.iloc[::sample_every]
    tenors = [float(c) for c in df.columns]
    fig = go.Figure()

    for date, row in df_sampled.iterrows():
        fig.add_trace(
            go.Scatter(
                visible=False,
                line=dict(color="#00ced1", width=3),
                name=str(date),
                x=tenors,
                y=row.values,
            )
        )
    fig.data[0].visible = True

    steps = []
    for i in range(len(fig.data)):
        step = dict(
            method="update",
            args=[
                {"visible": [False] * len(fig.data)},
                {"title": f"Yield Curve Snapshot: {df_sampled.index[i]}"},
            ],
            label=str(df_sampled.index[i])[:10],
        )
        step["args"][0]["visible"][i] = True
        steps.append(step)

    fig.update_layout(
        sliders=[dict(active=0, currentvalue={"prefix": "Date: "}, steps=steps)],
        title="Interactive Yield Curve Flipbook",
        xaxis_title="Tenor (Years)",
        yaxis_title="Yield (%)",
    )
    return fig


def build_ns_fit_slider_figure(df, params_df, tenors, smooth_tenors, sample_every=10):
    """Interactive flipbook comparing fitted NS curves against raw observations."""
    df_sampled = df.iloc[::sample_every]
    fig = go.Figure()

    for date, row in df_sampled.iterrows():
        p = params_df.loc[date]
        y_fitted = nelson_siegel_yield(
            smooth_tenors, p["b0_level"], p["b1_slope"], p["b2_curvature"], p["lambda"]
        )

        fig.add_trace(
            go.Scatter(
                visible=False,
                line=dict(color="#1f77b4", width=3),
                name="NS Fitted Curve",
                x=smooth_tenors,
                y=y_fitted,
            )
        )
        fig.add_trace(
            go.Scatter(
                visible=False,
                mode="markers",
                marker=dict(color="#d62728", size=8, symbol="x"),
                name="Raw FRED Data",
                x=tenors,
                y=row.values,
            )
        )

    if len(fig.data) >= 2:
        fig.data[0].visible = True
        fig.data[1].visible = True

    steps = []
    for i in range(0, len(fig.data), 2):
        idx = i // 2
        step = dict(
            method="update",
            args=[
                {"visible": [False] * len(fig.data)},
                {"title": f"Yield Curve Dynamics: {df_sampled.index[idx].date()}"},
            ],
            label=str(df_sampled.index[idx].year),
        )
        step["args"][0]["visible"][i] = True
        step["args"][0]["visible"][i + 1] = True
        steps.append(step)

    y_max = float(df.max().max()) + 0.01
    fig.update_layout(
        sliders=[dict(active=0, currentvalue={"prefix": "Date: "}, steps=steps)],
        title="Yield Curve Evolution: Theory vs. Reality",
        xaxis_title="Tenor (Years to Maturity)",
        yaxis_title="Yield (%)",
        template="plotly_white",
        yaxis=dict(range=[0, y_max], gridcolor="lightgrey"),
        xaxis=dict(range=[-1, 31], gridcolor="lightgrey"),
        showlegend=True,
        legend=dict(x=0.8, y=0.9),
    )
    return fig


# Curated, historically-grounded snapshots for `build_static_curve_overview_figure`:
# (target date -- snapped to the nearest available trading day at/before it,
# color, fixed narrative or None to build one live from the data). Dates and
# narratives are hand-picked for how clearly each one illustrates a distinct
# curve regime, not sampled automatically -- but the "Steep/Flat/Inverted"
# word attached to each is always computed from that date's actual 2s10s
# spread (see `_classify_curve_shape`), never hardcoded, so it can't drift out
# of sync with what's actually plotted if the underlying data is refreshed.
_MARKET_CONDITION_SNAPSHOTS = [
    (
        "2003-07-29",
        "#3a6b5c",
        "Emergency-low policy rate (~1%) after the dot-com bust; recovery "
        "already priced in further out the curve.",
    ),
    (
        "2006-11-15",
        "#9c5a34",
        "Fed done hiking (funds rate ~5.25%); short rates sit above long "
        "ones -- a warning later validated by the 2008 crisis.",
    ),
    (
        "2010-01-11",
        "#c9992e",
        "Post-crisis zero rates and QE -- the steepest curve in this "
        "sample, pricing in years of recovery.",
    ),
    (
        "2023-07-03",
        "#7a3b52",
        "Fastest tightening cycle in decades to fight inflation; 2s10s near "
        "its deepest inversion since 1981.",
    ),
    (
        None,  # always the most recent date in whatever df is passed in
        "#45607a",
        None,  # narrative is built live from the data below, not fixed
    ),
]


def _nearest_tenor_column(df, target_years):
    """The column whose tenor (parsed as float) is closest to `target_years`.

    Real data always has an exact "2.0"/"10.0" column (`TREASURY_SYMBOL_MAP`
    guarantees both), but this keeps the function from raising a `KeyError`
    against a caller that passes a reduced tenor grid (e.g. a fast synthetic
    test fixture) instead."""
    return min(df.columns, key=lambda c: abs(float(c) - target_years))


def _classify_curve_shape(spread_2s10s_pct):
    """Classify a curve's shape from its 10y-2y spread, in percentage points
    (e.g. 1.0 == 100bp). Below zero is a textbook inversion; above ~75bp is
    unambiguously steep; the range between reads as a genuinely flat curve,
    not confidently one or the other."""
    if spread_2s10s_pct < 0:
        return "Inverted"
    if spread_2s10s_pct < 0.75:
        return "Flat"
    return "Steep"


def _annotate_curve_snapshot(fig, ax, x, y, color, headline, narrative, wrap_width=36):
    """Draw a headline + wrapped narrative as one colored callout box, anchored
    at (x, y) in axes-fraction coordinates. Matplotlib can't mix font weights
    within a single Text object, so the headline (bold) and narrative (regular)
    are drawn as two Text objects; the box itself is a separate patch sized to
    their combined rendered extent (computed via a draw pass), so it reads as
    one seamless callout rather than two stacked boxes."""
    wrapped = textwrap.fill(narrative, width=wrap_width)
    headline_text = ax.text(
        x,
        y,
        headline,
        transform=ax.transAxes,
        fontsize=9.5,
        fontweight="bold",
        color="white",
        ha="left",
        va="top",
        zorder=6,
    )
    narrative_text = ax.text(
        x,
        y - 0.05,
        wrapped,
        transform=ax.transAxes,
        fontsize=8.2,
        color="white",
        ha="left",
        va="top",
        linespacing=1.45,
        zorder=6,
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    union = Bbox.union(
        [headline_text.get_window_extent(renderer), narrative_text.get_window_extent(renderer)]
    )
    inv = ax.transAxes.inverted()
    x0, y0 = inv.transform((union.x0, union.y0))
    x1, y1 = inv.transform((union.x1, union.y1))
    pad = 0.014
    ax.add_patch(
        FancyBboxPatch(
            (x0 - pad, y0 - pad),
            (x1 - x0) + 2 * pad,
            (y1 - y0) + 2 * pad,
            boxstyle="round,pad=0,rounding_size=0.012",
            transform=ax.transAxes,
            facecolor=color,
            edgecolor="none",
            alpha=0.94,
            zorder=5,
        )
    )


# (x, y) anchor per snapshot, in axes-fraction coordinates, hand-placed so the
# five boxes land in open space near where their own curve actually sits
# rather than on top of a line or each other -- see the values this was tuned
# against in the module docstring-adjacent comment above each snapshot.
_ANNOTATION_ANCHORS = [(0.02, 0.24), (0.02, 0.90), (0.36, 0.28), (0.55, 0.90), (0.70, 0.55)]


def build_static_curve_overview_figure(df, snapshots=None):
    """A handful of historically representative yield curves as one static
    Matplotlib figure, each labeled with its shape (steep/flat/inverted,
    computed from the data) and a short market-conditions caption -- a
    portable alternative to `build_yield_curve_slider_figure` for contexts
    (e.g. a self-contained HTML report) that need a plain image rather than
    a Plotly widget.

    `snapshots` defaults to `_MARKET_CONDITION_SNAPSHOTS`: a curated list of
    `(target_date, color, narrative)` tuples. Each `target_date` snaps to the
    nearest available trading day at/before it (`None` means "the most recent
    date in `df`"); a date entirely before `df`'s coverage is silently
    skipped rather than raising, so this still works on a shorter dataset.
    """
    snapshots = snapshots if snapshots is not None else _MARKET_CONDITION_SNAPSHOTS
    tenors = [float(c) for c in df.columns]
    col_2y = _nearest_tenor_column(df, 2.0)
    col_10y = _nearest_tenor_column(df, 10.0)

    fig, ax = plt.subplots(figsize=(10, 6.4))
    ax.set_facecolor("#fbfaf6")

    # Pass 1: resolve each snapshot's actual row and plot its curve. Collected
    # first (rather than annotating inline) so the axes limits -- and hence
    # what "0.24 of the way up" in axes-fraction space means -- are fixed
    # *before* any annotation box is placed; annotating inline would let each
    # new line's autoscale shift already-placed boxes' effective position.
    resolved = []
    for target, color, narrative in snapshots:
        if target is None:
            date = df.index.max()
        else:
            date = df.index.asof(pd.Timestamp(target))
            if pd.isna(date):
                continue

        row = df.loc[date] * 100.0
        spread = row[col_10y] - row[col_2y]
        shape = _classify_curve_shape(spread)
        if narrative is None:
            direction = "above" if spread >= 0 else "below"
            narrative = (
                f"Curve has normalized to a mild {shape.lower()} shape after "
                f"the 2022-23 inversion; the 10-year sits {abs(spread) * 100:.0f} "
                f"bp {direction} the 2-year."
            )
        headline = f"{date.year} — {shape}"
        resolved.append((date, color, headline, narrative, row))

        ax.plot(
            tenors,
            row.values,
            color=color,
            linewidth=2.2,
            marker="o",
            markersize=3,
            label=str(date.date()),
        )

    # Extra headroom below the lowest plotted point so a bottom-anchored
    # annotation box has canvas to render into instead of getting clipped by
    # the axes spine (matplotlib clips text/patches to the axes by default).
    y_min = min(row.min() for *_, row in resolved)
    ax.set_ylim(bottom=min(0.0, y_min) - 1.4)

    # Pass 2: now that ylim is final, axes-fraction anchors are stable.
    anchors = iter(_ANNOTATION_ANCHORS)
    for date, color, headline, narrative, row in resolved:
        anchor = next(anchors, (0.5, 0.5))
        _annotate_curve_snapshot(fig, ax, *anchor, color, headline, narrative)

    ax.set_xlabel("Maturity (years)")
    ax.set_ylabel("Yield (%)")
    ax.set_title(
        "Observed Treasury Yield Curves Over Time", fontsize=14, fontfamily="serif", color="#23281f"
    )
    ax.grid(True, alpha=0.4, color="#e2e0d4")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9, title="Snapshot date")
    plt.tight_layout()
    return fig
