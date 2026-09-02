"""OU-fit diagnostic figure: per-factor time series, stationary-distribution histogram,
and empirical vs theoretical autocorrelation."""

import matplotlib.pyplot as plt
import numpy as np


def build_ou_diagnostics_figure(scores_df, ou_params, dt=1 / 252):
    factor_names = list(scores_df.columns)
    n = len(factor_names)
    fig, axes = plt.subplots(n, 3, figsize=(15, 4 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    for i, name in enumerate(factor_names):
        X = scores_df[name].values
        params = ou_params[name]

        axes[i, 0].plot(scores_df.index, X, alpha=0.7, linewidth=0.8)
        axes[i, 0].axhline(
            y=params["theta"], color="r", linestyle="--", label=f"theta = {params['theta']:.3f}"
        )
        axes[i, 0].axhline(y=0, color="k", linestyle=":", alpha=0.3)
        axes[i, 0].set_title(f"{name} Time Series")
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)

        axes[i, 1].hist(X, bins=50, density=True, alpha=0.7, edgecolor="black")
        stationary_std = params["sigma"] / np.sqrt(2 * params["kappa"])
        x_range = np.linspace(X.min(), X.max(), 100)
        theoretical_pdf = (1 / (stationary_std * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * ((x_range - params["theta"]) / stationary_std) ** 2
        )
        axes[i, 1].plot(
            x_range, theoretical_pdf, "r-", linewidth=2, label="Theoretical stationary dist."
        )
        axes[i, 1].set_title(f"{name} Distribution")
        axes[i, 1].legend()
        axes[i, 1].grid(True, alpha=0.3)

        max_lag = min(50, len(X) // 4)
        acf_values = [
            np.corrcoef(X[:-lag], X[lag:])[0, 1] if lag > 0 else 1.0 for lag in range(max_lag)
        ]
        theoretical_acf = [np.exp(-params["kappa"] * dt * lag) for lag in range(max_lag)]
        axes[i, 2].bar(range(max_lag), acf_values, alpha=0.7, label="Empirical")
        axes[i, 2].plot(range(max_lag), theoretical_acf, "r-", linewidth=2, label="Theoretical OU")
        axes[i, 2].set_title(f"{name} Autocorrelation")
        axes[i, 2].legend()
        axes[i, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
