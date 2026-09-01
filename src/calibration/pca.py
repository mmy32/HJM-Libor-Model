"""Pure PCA fit/transform for Nelson-Siegel parameter panels.

Unlike the archived version, the fitted StandardScaler is kept as part of the
returned model so new NS-parameter rows can later be projected onto the same
basis via `transform_pca`, without refitting from scratch.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.registry.factor_spec import N_PCA_FACTORS


@dataclass
class PCAFactorModel:
    scaler: StandardScaler
    pca: PCA
    scores: pd.DataFrame
    loadings: pd.DataFrame
    explained_variance_ratio: np.ndarray


def fit_pca(params_df: pd.DataFrame, n_components=None) -> PCAFactorModel:
    n_components = n_components or N_PCA_FACTORS

    scaler = StandardScaler()
    scaled = scaler.fit_transform(params_df)

    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(scaled)

    factor_names = [f"PC{i + 1}" for i in range(n_components)]
    scores_df = pd.DataFrame(scores, index=params_df.index, columns=factor_names)
    loadings_df = pd.DataFrame(
        pca.components_.T, index=params_df.columns, columns=factor_names
    )

    return PCAFactorModel(
        scaler=scaler,
        pca=pca,
        scores=scores_df,
        loadings=loadings_df,
        explained_variance_ratio=pca.explained_variance_ratio_,
    )


def transform_pca(model: PCAFactorModel, params_df: pd.DataFrame) -> pd.DataFrame:
    """Project new NS-parameter rows onto an already-fitted PCA basis."""
    scaled = model.scaler.transform(params_df)
    scores = model.pca.transform(scaled)
    return pd.DataFrame(scores, index=params_df.index, columns=model.scores.columns)
