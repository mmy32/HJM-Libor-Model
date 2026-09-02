"""Simulation diagnostic figure: sample factor/curve paths, terminal distribution."""

import matplotlib.pyplot as plt
import numpy as np


def build_sample_paths_figure(
    result, n_factors, n_sample=5, plot_type="zero_rates", random_seed=None
):
    rng = np.random.default_rng(random_seed)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    time_grid = result.time_grid
    maturities = result.maturities
    sample_paths = rng.choice(result.n_paths, n_sample, replace=False)

    ax = axes[0, 0]
    for pc_idx in range(min(3, n_factors)):
        for path in sample_paths:
            ax.plot(time_grid, result.pc_paths[path, :, pc_idx], alpha=0.6, linewidth=1)
        mean_path = result.pc_paths[:, :, pc_idx].mean(axis=0)
        ax.plot(time_grid, mean_path, "k--", linewidth=2, label=f"PC{pc_idx + 1} mean")
    ax.set_title("Principal Component Evolution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="k", linestyle=":", alpha=0.3)

    curves = result.zero_curves if plot_type == "zero_rates" else result.forward_curves

    ax = axes[0, 1]
    for path in sample_paths:
        ax.plot(maturities, curves[path, -1, :] * 100, alpha=0.6, linewidth=1.5)
    ax.plot(maturities, curves[:, -1, :].mean(axis=0) * 100, "k--", linewidth=2.5, label="Mean")
    ax.set_title(f"Final Yield Curves (T={time_grid[-1]:.2f}y)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    idx_10y = int(np.argmin(np.abs(maturities - 10.0)))
    rates_10y = curves[:, :, idx_10y] * 100
    for path in sample_paths:
        ax.plot(time_grid, rates_10y[path], alpha=0.6, linewidth=1)
    mean_10y = rates_10y.mean(axis=0)
    std_10y = rates_10y.std(axis=0)
    ax.plot(time_grid, mean_10y, "k--", linewidth=2.5, label="Mean")
    ax.fill_between(
        time_grid,
        mean_10y - 2 * std_10y,
        mean_10y + 2 * std_10y,
        alpha=0.2,
        color="gray",
        label="+/-2 sigma",
    )
    ax.set_title("10-Year Rate Evolution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.hist(rates_10y[:, -1], bins=50, alpha=0.7, edgecolor="black", density=True)
    ax.axvline(mean_10y[-1], color="r", linestyle="--", linewidth=2, label="Mean")
    ax.set_title(f"Distribution at T={time_grid[-1]:.2f}y")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
