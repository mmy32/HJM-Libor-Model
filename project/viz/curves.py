"""Curve-related figures: raw yield curve slider, NS-fit slider, static overview.

Figures are returned, never shown or saved here -- the caller decides.
"""

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go

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
# color). Hand-picked for how clearly each one illustrates a distinct curve
# regime, not sampled automatically:
#   2003-07-29 -- emergency-low policy rate (~1%) after the dot-com bust.
#   2006-11-15 -- Fed done hiking (~5.25%); short rates above long, a
#                 warning later validated by the 2008 crisis.
#   2010-01-11 -- post-crisis zero rates and QE; the steepest curve in the
#                 sample, pricing in years of recovery.
#   2023-07-03 -- fastest tightening cycle in decades; 2s10s near its
#                 deepest inversion since 1981.
#   None       -- always the most recent date in whatever df is passed in.
# The "Steep/Flat/Inverted" word attached to each in the legend is computed
# live from that date's actual 2s10s spread (see `_classify_curve_shape`),
# never hardcoded, so it can't drift out of sync with what's plotted if the
# underlying data is refreshed.
_MARKET_CONDITION_SNAPSHOTS = [
    ("2003-07-29", "#3a6b5c"),
    ("2006-11-15", "#9c5a34"),
    ("2010-01-11", "#c9992e"),
    ("2023-07-03", "#7a3b52"),
    (None, "#45607a"),
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


def build_static_curve_overview_figure(df, snapshots=None):
    """A handful of historically representative yield curves as one static
    Matplotlib figure, each legend entry labeled with its shape
    (steep/flat/inverted, computed from the data) -- a portable alternative
    to `build_yield_curve_slider_figure` for contexts (e.g. a self-contained
    HTML report) that need a plain image rather than a Plotly widget.

    `snapshots` defaults to `_MARKET_CONDITION_SNAPSHOTS`: a curated list of
    `(target_date, color)` pairs. Each `target_date` snaps to the nearest
    available trading day at/before it (`None` means "the most recent date
    in `df`"); a date entirely before `df`'s coverage is silently skipped
    rather than raising, so this still works on a shorter dataset.
    """
    snapshots = snapshots if snapshots is not None else _MARKET_CONDITION_SNAPSHOTS
    tenors = [float(c) for c in df.columns]
    col_2y = _nearest_tenor_column(df, 2.0)
    col_10y = _nearest_tenor_column(df, 10.0)

    fig, ax = plt.subplots(figsize=(10, 6.4))
    ax.set_facecolor("#fbfaf6")

    for target, color in snapshots:
        if target is None:
            date = df.index.max()
        else:
            date = df.index.asof(pd.Timestamp(target))
            if pd.isna(date):
                continue

        row = df.loc[date] * 100.0
        spread = row[col_10y] - row[col_2y]
        shape = _classify_curve_shape(spread)

        ax.plot(
            tenors,
            row.values,
            color=color,
            linewidth=2.2,
            marker="o",
            markersize=3,
            label=f"{date.date()} — {shape}",
        )

    ax.set_xlabel("Maturity (years)")
    ax.set_ylabel("Yield (%)")
    ax.set_title(
        "Observed Treasury Yield Curves Over Time", fontsize=14, fontfamily="serif", color="#23281f"
    )
    ax.grid(True, alpha=0.4, color="#e2e0d4")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9, title="Snapshot date")
    plt.tight_layout()
    return fig
