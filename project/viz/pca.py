"""PCA diagnostic figure: scree, cumulative variance, PC time series, loadings heatmap."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def build_pca_diagnostics_figure(model):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    cumsum = np.cumsum(model.explained_variance_ratio)

    axes[0, 0].bar(
        range(1, len(model.explained_variance_ratio) + 1), model.explained_variance_ratio
    )
    axes[0, 0].set_title("Scree Plot")

    axes[0, 1].plot(range(1, len(cumsum) + 1), cumsum, marker="o")
    axes[0, 1].axhline(y=0.95, color="r", linestyle="--")
    axes[0, 1].set_title("Cumulative Explained Variance")

    for col in model.scores.columns[:2]:
        axes[1, 0].plot(model.scores.index, model.scores[col], label=col, alpha=0.7)
    axes[1, 0].legend()
    axes[1, 0].set_title("First Two Principal Components Over Time")

    sns.heatmap(model.loadings, annot=True, fmt=".3f", cmap="RdBu_r", center=0, ax=axes[1, 1])
    axes[1, 1].set_title("PCA Loadings Heatmap")

    plt.tight_layout()
    return fig
