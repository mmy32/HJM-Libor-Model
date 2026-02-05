# src/calibration/pca.py

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def run_pca_analysis(input_path='data/ns_parameters/ns_parameters.csv', 
                     output_dir='data/ns_parameters/',
                     show_plot=True):
    """
    Performs PCA on Nelson-Siegel parameters and returns results for notebook use.
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # --- 1. Load Saved Parameters ---
    params_df = pd.read_csv(input_path, index_col=0, parse_dates=True)

    # --- 2. Standardize the Data ---
    scaler = StandardScaler()
    params_scaled = scaler.fit_transform(params_df)

    # --- 3. Perform PCA ---
    pca = PCA()
    principal_components = pca.fit_transform(params_scaled)

    # Create DataFrame with principal components
    pca_df = pd.DataFrame(
        principal_components,
        index=params_df.index,
        columns=[f'PC{i+1}' for i in range(principal_components.shape[1])]
    )

    # --- 4. Analyze Results ---
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(pca.components_.shape[0])],
        index=params_df.columns
    )

    # --- 5. Visualizations ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    cumsum = np.cumsum(pca.explained_variance_ratio_)

    # Plot 1: Scree Plot
    axes[0, 0].bar(range(1, len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_)
    axes[0, 0].set_title('Scree Plot')
    
    # Plot 2: Cumulative Variance
    axes[0, 1].plot(range(1, len(cumsum)+1), cumsum, marker='o')
    axes[0, 1].axhline(y=0.95, color='r', linestyle='--')
    axes[0, 1].set_title('Cumulative Explained Variance')

    # Plot 3: PC1 vs PC2 Time Series
    axes[1, 0].plot(pca_df.index, pca_df['PC1'], label='PC1', alpha=0.7)
    axes[1, 0].plot(pca_df.index, pca_df['PC2'], label='PC2', alpha=0.7)
    axes[1, 0].legend()
    axes[1, 0].set_title('First Two Principal Components Over Time')

    # Plot 4: Component Loadings Heatmap
    sns.heatmap(loadings, annot=True, fmt='.3f', cmap='RdBu_r', center=0, ax=axes[1, 1])
    axes[1, 1].set_title('PCA Loadings Heatmap')

    plt.tight_layout()
    plt.savefig(output_path / 'pca_analysis_plot.png', dpi=300)
    
    if show_plot:
        plt.show()
    else:
        plt.close()

    # --- 6. Save PCA Results ---
    pca_df.to_csv(output_path / 'principal_components.csv')
    loadings.to_csv(output_path / 'pca_loadings.csv')

    return pca_df, loadings, pca, fig

if __name__ == "__main__":
    # This allows you to still run the file directly from the terminal
    run_pca_analysis()