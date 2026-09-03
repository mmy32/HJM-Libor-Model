"""Backtest diagnostic figure: forecast accuracy and band calibration across horizons."""

import matplotlib.pyplot as plt


def build_backtest_horizon_figure(horizon_summary_df, band=0.90):
    """Two panels, both indexed by horizon (trading days): RMSE (model vs.
    naive random-walk vs. a random-forest ML baseline) on the left, achieved
    band coverage with its Wilson confidence interval against the nominal
    band on the right.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    horizons = horizon_summary_df.index.values

    ax = axes[0]
    ax.plot(horizons, horizon_summary_df["rmse"], marker="o", linewidth=2, label="Model")
    ax.plot(
        horizons,
        horizon_summary_df["naive_rmse"],
        marker="o",
        linewidth=2,
        linestyle="--",
        label="Naive (no-change)",
    )
    ax.plot(
        horizons,
        horizon_summary_df["ml_rmse"],
        marker="o",
        linewidth=2,
        linestyle=":",
        label="ML (random forest)",
    )
    ax.set_xlabel("Horizon (trading days)")
    ax.set_ylabel("RMSE")
    ax.set_title("Forecast error vs. horizon")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    coverage = horizon_summary_df["coverage"]
    # clip at 0: a Wilson bound can land a float epsilon on the wrong side
    # of `coverage` (e.g. 1.0 - 1e-16) at the 0%/100% edges, and errorbar
    # rejects any negative length outright, however small.
    err_lo = (coverage - horizon_summary_df["coverage_ci_lo"]).clip(lower=0)
    err_hi = (horizon_summary_df["coverage_ci_hi"] - coverage).clip(lower=0)
    ax.errorbar(
        horizons,
        coverage,
        yerr=[err_lo, err_hi],
        marker="o",
        capsize=4,
        linewidth=2,
        label="Achieved coverage (95% CI)",
    )
    ax.axhline(y=band, color="r", linestyle="--", label=f"Nominal {band:.0%}")
    ax.set_xlabel("Horizon (trading days)")
    ax.set_ylabel("Coverage")
    ax.set_ylim(0, 1.05)
    ax.set_title("Band coverage vs. horizon")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
