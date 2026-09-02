"""Sensitivity figures: NS-parameter sensitivities and PC-to-forward-rate sensitivities."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def build_ns_sensitivities_figure(maturities, sens):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(maturities, sens["db0"], label="df/db0", linewidth=2)
    axes[0, 0].plot(maturities, sens["db1"], label="df/db1", linewidth=2)
    axes[0, 0].plot(maturities, sens["db2"], label="df/db2", linewidth=2)
    axes[0, 0].plot(maturities, sens["dlambda"], label="df/dlambda", linewidth=2)
    axes[0, 0].set_title("NS Parameter Sensitivities")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0, color="k", linestyle="--", alpha=0.3)

    axes[0, 1].plot(maturities, sens["db1"], linewidth=2, color="C1")
    axes[0, 1].fill_between(maturities, 0, sens["db1"], alpha=0.3)
    axes[0, 1].set_title("b1 Sensitivity (Short-term Component)")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(maturities, sens["db2"], linewidth=2, color="C2")
    axes[1, 0].fill_between(maturities, 0, sens["db2"], alpha=0.3)
    axes[1, 0].set_title("b2 Sensitivity (Curvature/Hump)")
    axes[1, 0].grid(True, alpha=0.3)

    sens_matrix = np.column_stack([sens["db0"], sens["db1"], sens["db2"], sens["dlambda"]])
    sns.heatmap(
        sens_matrix.T,
        xticklabels=[f"{m:.1f}Y" if i % 3 == 0 else "" for i, m in enumerate(maturities)],
        yticklabels=["b0", "b1", "b2", "lambda"],
        cmap="RdBu_r",
        center=0,
        cbar_kws={"label": "Sensitivity"},
        ax=axes[1, 1],
    )
    axes[1, 1].set_title("Sensitivity Heatmap")

    plt.tight_layout()
    return fig


def build_pc_sensitivities_figure(pc_sens_df, factor_names=None):
    """`factor_names` defaults to `pc_sens_df`'s own columns rather than the
    registry's global FACTOR_NAMES -- the number of PCA factors actually
    used varies (e.g. 3 when NS parameters were fit with lambda held fixed,
    see curves.nelson_siegel.calibrate_all_days_fixed_lambda), so the
    figure should reflect what's actually in the data passed in."""
    factor_names = factor_names or list(pc_sens_df.columns)
    n = len(factor_names)
    ncols = 2
    nrows = (n + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5 * nrows))
    axes = np.atleast_2d(axes)

    for i, name in enumerate(factor_names):
        ax = axes[i // ncols, i % ncols]
        maturities = pc_sens_df.index.values
        ax.plot(maturities, pc_sens_df[name], linewidth=2.5, color=f"C{i}")
        ax.fill_between(maturities, 0, pc_sens_df[name], alpha=0.3, color=f"C{i}")
        ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
        ax.set_title(f"{name} Sensitivity to Forward Rates")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
