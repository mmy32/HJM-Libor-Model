"""Curve-related figures: raw yield curve slider, NS-fit slider, static overview.

Figures are returned, never shown or saved here -- the caller decides.
"""

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from project.curves.nelson_siegel import nelson_siegel_yield


def build_yield_curve_slider_figure(df):
    """Interactive flipbook of raw observed yield curves, one frame per date."""
    tenors = [float(c) for c in df.columns]
    fig = go.Figure()

    for date, row in df.iterrows():
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
                {"title": f"Yield Curve Snapshot: {df.index[i]}"},
            ],
            label=str(df.index[i])[:10],
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


def build_static_curve_overview_figure(df, n_curves=5):
    """A handful of historical yield curves, evenly spaced across the sample,
    as one static Matplotlib figure -- a portable alternative to
    `build_yield_curve_slider_figure` for contexts (e.g. a self-contained
    HTML report) that need a plain image rather than a Plotly widget."""
    tenors = [float(c) for c in df.columns]
    sample_idx = [int(i) for i in np.linspace(0, len(df) - 1, min(n_curves, len(df)))]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i in sample_idx:
        date = df.index[i]
        label = str(date.date()) if hasattr(date, "date") else str(date)
        ax.plot(tenors, df.iloc[i].values, marker="o", markersize=3, linewidth=1.8, label=label)
    ax.set_xlabel("Tenor (years)")
    ax.set_ylabel("Yield")
    ax.set_title("Observed Treasury Yield Curves Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig
