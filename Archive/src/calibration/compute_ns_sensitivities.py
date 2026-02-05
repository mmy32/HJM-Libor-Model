# compute_ns_sensitivities.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

# --- 1. Nelson-Siegel Forward Rate Function ---

def nelson_siegel_forward(tau_values, beta0, beta1, beta2, tau):
    """
    Compute Nelson-Siegel instantaneous forward rate
    
    f(τ) = β₀ + β₁·exp(-τ/λ) + β₂·(τ/λ)·exp(-τ/λ)
    
    Parameters:
    -----------
    tau_values : array-like
        Time to maturity values (T - t)
    beta0, beta1, beta2, tau : float
        NS parameters
    
    Returns:
    --------
    array : Forward rates for each maturity
    """
    m = np.array(tau_values) / tau  # Normalized maturity
    
    term1 = beta0
    term2 = beta1 * np.exp(-m)
    term3 = beta2 * m * np.exp(-m)
    
    return term1 + term2 + term3

# --- 2. Analytical Sensitivities ---

def ns_sensitivities(tau_values, beta0, beta1, beta2, tau):
    """
    Compute analytical partial derivatives of NS forward rate
    
    Returns:
    --------
    dict with keys: 'dbeta0', 'dbeta1', 'dbeta2', 'dtau'
    Each is an array of sensitivities at each maturity
    """
    tau_values = np.array(tau_values)
    m = tau_values / tau  # Normalized maturity
    exp_m = np.exp(-m)
    
    # ∂f/∂β₀ = 1 (constant across all maturities)
    df_dbeta0 = np.ones_like(tau_values)
    
    # ∂f/∂β₁ = exp(-τ/λ)
    df_dbeta1 = exp_m
    
    # ∂f/∂β₂ = (τ/λ)·exp(-τ/λ)
    df_dbeta2 = m * exp_m
    
    # ∂f/∂λ = β₁·(τ/λ²)·exp(-τ/λ) + β₂·[(1/λ)·exp(-τ/λ) - (τ/λ²)·exp(-τ/λ)]
    #       = (τ/λ²)·exp(-τ/λ)·[β₁ + β₂·(λ/τ - 1)]
    df_dtau = (tau_values / (tau**2)) * exp_m * (beta1 - beta2 * (m - 1))
    
    return {
        'dbeta0': df_dbeta0,
        'dbeta1': df_dbeta1,
        'dbeta2': df_dbeta2,
        'dtau': df_dtau
    }
# --- 3. Load Your Estimated Parameters ---

print("="*70)
print("COMPUTING NELSON-SIEGEL SENSITIVITIES")
print("="*70)

# Load mean NS parameters
params_df = pd.read_csv('data/ns_parameters/ns_parameters.csv', index_col=0, parse_dates=True)

print(f"\nLoaded columns: {params_df.columns.tolist()}")

# Extract mean parameters (using YOUR actual column names)
mean_params = params_df.mean()

print("\nMean NS Parameters:")
print(f"  β₀ (level):     {mean_params['b0_level']:.4f}")
print(f"  β₁ (slope):     {mean_params['b1_slope']:.4f}")
print(f"  β₂ (curvature): {mean_params['b2_curvature']:.4f}")
print(f"  λ (decay):      {mean_params['lambda']:.4f}")

# Define maturity grid
maturities = np.array([
    0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
    12, 15, 20, 25, 30
])

print(f"\nComputing sensitivities for {len(maturities)} maturities:")
print(f"  Range: {maturities[0]}Y to {maturities[-1]}Y")

# --- 4. Compute Sensitivities ---

sensitivities = ns_sensitivities(
    maturities,
    mean_params['b0_level'],      # Changed from beta_0
    mean_params['b1_slope'],       # Changed from beta_1
    mean_params['b2_curvature'],   # Changed from beta_2
    mean_params['lambda']          # Changed from tau
)

# Create DataFrame
sens_df = pd.DataFrame({
    'maturity': maturities,
    'df_db0': sensitivities['dbeta0'],      # You can rename these too
    'df_db1': sensitivities['dbeta1'],
    'df_db2': sensitivities['dbeta2'],
    'df_dlambda': sensitivities['dtau']     # Changed from dtau
})

print("\nSample sensitivities (5Y maturity):")
row_5y = sens_df[sens_df['maturity'] == 5].iloc[0]
print(f"  ∂f/∂β₀ = {row_5y['df_db0']:.4f}")
print(f"  ∂f/∂β₁ = {row_5y['df_db1']:.4f}")
print(f"  ∂f/∂β₂ = {row_5y['df_db2']:.4f}")
print(f"  ∂f/∂λ  = {row_5y['df_dlambda']:.4f}")

# --- 5. Visualize Sensitivities ---

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: All sensitivities
axes[0, 0].plot(maturities, sensitivities['dbeta0'], label='∂f/∂β₀', linewidth=2)
axes[0, 0].plot(maturities, sensitivities['dbeta1'], label='∂f/∂β₁', linewidth=2)
axes[0, 0].plot(maturities, sensitivities['dbeta2'], label='∂f/∂β₂', linewidth=2)
axes[0, 0].plot(maturities, sensitivities['dtau'], label='∂f/∂τ', linewidth=2)
axes[0, 0].set_xlabel('Maturity (years)')
axes[0, 0].set_ylabel('Sensitivity')
axes[0, 0].set_title('NS Parameter Sensitivities')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)

# Plot 2: β₁ sensitivity (short-term component)
axes[0, 1].plot(maturities, sensitivities['dbeta1'], linewidth=2, color='C1')
axes[0, 1].fill_between(maturities, 0, sensitivities['dbeta1'], alpha=0.3)
axes[0, 1].set_xlabel('Maturity (years)')
axes[0, 1].set_ylabel('∂f/∂β₁')
axes[0, 1].set_title('β₁ Sensitivity (Short-term Component)')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].annotate('Decays exponentially', xy=(5, 0.1), fontsize=10)

# Plot 3: β₂ sensitivity (medium-term hump)
axes[1, 0].plot(maturities, sensitivities['dbeta2'], linewidth=2, color='C2')
axes[1, 0].fill_between(maturities, 0, sensitivities['dbeta2'], alpha=0.3)
axes[1, 0].set_xlabel('Maturity (years)')
axes[1, 0].set_ylabel('∂f/∂β₂')
axes[1, 0].set_title('β₂ Sensitivity (Curvature/Hump)')
axes[1, 0].grid(True, alpha=0.3)
# Find peak
peak_idx = np.argmax(sensitivities['dbeta2'])
peak_maturity = maturities[peak_idx]
axes[1, 0].annotate(f'Peak at {peak_maturity:.1f}Y', 
                    xy=(peak_maturity, sensitivities['dbeta2'][peak_idx]),
                    xytext=(peak_maturity+5, sensitivities['dbeta2'][peak_idx]),
                    arrowprops=dict(arrowstyle='->', color='red'))

# Plot 4: Heatmap of sensitivities
sens_matrix = np.column_stack([
    sensitivities['dbeta0'],
    sensitivities['dbeta1'],
    sensitivities['dbeta2'],
    sensitivities['dtau']
])
sns.heatmap(sens_matrix.T, 
            xticklabels=[f'{m:.1f}Y' if i % 3 == 0 else '' for i, m in enumerate(maturities)],
            yticklabels=['β₀', 'β₁', 'β₂', 'τ'],
            cmap='RdBu_r', center=0, 
            cbar_kws={'label': 'Sensitivity'},
            ax=axes[1, 1])
axes[1, 1].set_title('Sensitivity Heatmap')
axes[1, 1].set_xlabel('Maturity')

plt.tight_layout()
plt.savefig('data/ns_parameters/ns_sensitivities.png', dpi=300, bbox_inches='tight')
plt.show()

# --- 6. Compute PC-to-Forward-Rate Sensitivities ---

print("\n" + "="*70)
print("MAPPING PC MOVEMENTS TO FORWARD RATE CHANGES")
print("="*70)

# Load PCA loadings
loadings_df = pd.read_csv('data/ns_parameters/pca_loadings.csv', index_col=0)

print("\nPCA Loadings:")
print(loadings_df)

# Check what the index looks like
print(f"\nLoadings index (row names): {loadings_df.index.tolist()}")

n_pcs = loadings_df.shape[1]
pc_sensitivities = {}

for pc in range(1, n_pcs + 1):
    pc_name = f'PC{pc}'
    
    # Get loadings for this PC (using YOUR actual row names)
    w_b0 = loadings_df.loc['b0_level', pc_name]
    w_b1 = loadings_df.loc['b1_slope', pc_name]
    w_b2 = loadings_df.loc['b2_curvature', pc_name]
    w_lambda = loadings_df.loc['lambda', pc_name]
    
    # Compute sensitivity of forward rate to this PC
    df_dpc = (
        sensitivities['dbeta0'] * w_b0 +
        sensitivities['dbeta1'] * w_b1 +
        sensitivities['dbeta2'] * w_b2 +
        sensitivities['dtau'] * w_lambda
    )
    
    pc_sensitivities[pc_name] = df_dpc

# Create DataFrame
pc_sens_df = pd.DataFrame(pc_sensitivities, index=maturities)
pc_sens_df.index.name = 'maturity'

print("\nPC Sensitivities (how 1-unit PC move affects forward rates):")
print(pc_sens_df.head(10))

# --- 7. Visualize PC Sensitivities ---

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for i, pc in enumerate(['PC1', 'PC2', 'PC3', 'PC4']):
    ax = axes[i // 2, i % 2]
    ax.plot(maturities, pc_sens_df[pc], linewidth=2.5, color=f'C{i}')
    ax.fill_between(maturities, 0, pc_sens_df[pc], alpha=0.3, color=f'C{i}')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Maturity (years)')
    ax.set_ylabel(f'∂f/∂{pc}')
    ax.set_title(f'{pc} Sensitivity to Forward Rates')
    ax.grid(True, alpha=0.3)
    
    # Add interpretation
    max_impact = maturities[np.argmax(np.abs(pc_sens_df[pc]))]
    ax.annotate(f'Max impact at {max_impact:.1f}Y',
                xy=(max_impact, pc_sens_df[pc].iloc[np.argmax(np.abs(pc_sens_df[pc]))]),
                xytext=(max_impact+3, pc_sens_df[pc].max()*0.8),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=9)

plt.tight_layout()
plt.savefig('data/ns_parameters/pc_to_forward_sensitivities.png', dpi=300, bbox_inches='tight')
plt.show()

# --- 8. Save Results ---

# Save NS sensitivities
sens_df.to_csv('data/ns_parameters/ns_sensitivities.csv', index=False)

# Save PC-to-forward sensitivities
pc_sens_df.to_csv('data/ns_parameters/pc_forward_sensitivities.csv')

# Save as JSON for easy loading
sensitivity_data = {
    'maturities': maturities.tolist(),
    'ns_sensitivities': {
        'b0_level': sensitivities['dbeta0'].tolist(),
        'b1_slope': sensitivities['dbeta1'].tolist(),
        'b2_curvature': sensitivities['dbeta2'].tolist(),
        'lambda': sensitivities['dtau'].tolist()
    },
    'pc_sensitivities': {
        pc: pc_sens_df[pc].tolist() for pc in pc_sens_df.columns
    },
    'mean_parameters': {
        'b0_level': float(mean_params['b0_level']),
        'b1_slope': float(mean_params['b1_slope']),
        'b2_curvature': float(mean_params['b2_curvature']),
        'lambda': float(mean_params['lambda'])
    }
}

with open('data/ns_parameters/sensitivities.json', 'w') as f:
    json.dump(sensitivity_data, f, indent=2)

print("\n" + "="*70)
print("✓ NS SENSITIVITIES COMPUTED AND SAVED")
print("="*70)
print("  - NS sensitivities:         data/ns_parameters/ns_sensitivities.csv")
print("  - PC forward sensitivities: data/ns_parameters/pc_forward_sensitivities.csv")
print("  - All data (JSON):          data/ns_parameters/sensitivities.json")
print("  - Visualizations:           data/ns_parameters/ns_sensitivities.png")
print("                              data/ns_parameters/pc_to_forward_sensitivities.png")
print("="*70)

# --- 9. Summary Statistics ---

print("\nKEY INSIGHTS:")
print("-" * 70)

for pc in ['PC1', 'PC2', 'PC3', 'PC4']:
    max_sens = pc_sens_df[pc].abs().max()
    max_mat = maturities[pc_sens_df[pc].abs().argmax()]
    print(f"{pc}:")
    print(f"  Max impact: {max_sens:.4f} at {max_mat:.1f}Y maturity")
    print(f"  Interpretation: 1-std-dev move in {pc} → {max_sens*100:.2f} bps at {max_mat:.1f}Y")