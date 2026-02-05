# ============================================================================
# Ornstein-Uhlenbeck Parameter Estimation for Principal Components
# ============================================================================

import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# --- 1. Load PC Scores ---
pca_df = pd.read_csv('data/ns_parameters/principal_components.csv', index_col=0, parse_dates=True)

print("Loaded PC scores shape:", pca_df.shape)
print("\nFirst few rows:")
print(pca_df.head())

# --- 2. Ornstein-Uhlenbeck Parameter Estimation ---

def estimate_ou_parameters(time_series, dt=1/252):
    """
    Estimate OU process parameters: dX = κ(θ - X)dt + σ dW
    
    Using maximum likelihood estimation for discrete observations.
    For OU process: X(t+dt) | X(t) ~ Normal(θ + (X(t)-θ)e^(-κ·dt), σ²/(2κ)(1-e^(-2κ·dt)))
    
    Parameters:
    -----------
    time_series : array-like
        Time series of observations
    dt : float
        Time step (default: 1/252 for daily data in years)
    
    Returns:
    --------
    dict with keys: kappa (mean reversion), theta (long-run mean), sigma (volatility)
    """
    X = np.array(time_series)
    n = len(X)
    
    # Negative log-likelihood function
    def neg_log_likelihood(params):
        kappa, theta, sigma = params
        
        # Avoid numerical issues
        if kappa <= 0 or sigma <= 0:
            return 1e10
        
        # Conditional mean and variance
        exp_term = np.exp(-kappa * dt)
        mu_cond = theta + (X[:-1] - theta) * exp_term
        var_cond = (sigma**2 / (2 * kappa)) * (1 - np.exp(-2 * kappa * dt))
        
        if var_cond <= 0:
            return 1e10
        
        # Log-likelihood
        residuals = X[1:] - mu_cond
        ll = -0.5 * n * np.log(2 * np.pi * var_cond) - 0.5 * np.sum(residuals**2) / var_cond
        
        return -ll
    
    # Initial guesses
    empirical_mean = np.mean(X)
    empirical_std = np.std(X)
    
    # Use autocorrelation for initial kappa guess
    if len(X) > 1:
        acf_1 = np.corrcoef(X[:-1], X[1:])[0, 1]
        initial_kappa = -np.log(max(acf_1, 0.01)) / dt
    else:
        initial_kappa = 0.1
    
    initial_params = [initial_kappa, empirical_mean, empirical_std]
    
    # Optimize
    result = minimize(
        neg_log_likelihood,
        initial_params,
        method='L-BFGS-B',
        bounds=[(1e-6, 50), (None, None), (1e-6, None)]
    )
    
    if not result.success:
        print(f"Warning: Optimization did not converge. Message: {result.message}")
    
    kappa, theta, sigma = result.x
    
    # Calculate half-life (time to revert halfway to mean)
    half_life = np.log(2) / kappa
    
    return {
        'kappa': kappa,
        'theta': theta,
        'sigma': sigma,
        'half_life_days': half_life * 252,
        'log_likelihood': -result.fun
    }

# --- 3. Estimate Parameters for Each PC ---

print("\n" + "="*70)
print("ESTIMATING OU PARAMETERS FOR EACH PRINCIPAL COMPONENT")
print("="*70)

n_pcs = pca_df.shape[1]
ou_params = {}

for i in range(n_pcs):
    pc_name = f'PC{i+1}'
    print(f"\n{pc_name}:")
    
    # Estimate OU parameters
    params = estimate_ou_parameters(pca_df[pc_name].values)
    ou_params[pc_name] = params
    
    # Print results
    print(f"  κ (mean reversion speed): {params['kappa']:.4f}")
    print(f"  θ (long-run mean):        {params['theta']:.4f}")
    print(f"  σ (volatility):           {params['sigma']:.4f}")
    print(f"  Half-life:                {params['half_life_days']:.1f} days")
    print(f"  Log-likelihood:           {params['log_likelihood']:.2f}")

# --- 4. Diagnostic Plots ---

fig, axes = plt.subplots(n_pcs, 3, figsize=(15, 4*n_pcs))
if n_pcs == 1:
    axes = axes.reshape(1, -1)

for i in range(n_pcs):
    pc_name = f'PC{i+1}'
    X = pca_df[pc_name].values
    params = ou_params[pc_name]
    
    # Plot 1: Time series with mean
    axes[i, 0].plot(pca_df.index, X, alpha=0.7, linewidth=0.8)
    axes[i, 0].axhline(y=params['theta'], color='r', linestyle='--', 
                       label=f"θ = {params['theta']:.3f}")
    axes[i, 0].axhline(y=0, color='k', linestyle=':', alpha=0.3)
    axes[i, 0].set_title(f'{pc_name} Time Series')
    axes[i, 0].set_ylabel('Value')
    axes[i, 0].legend()
    axes[i, 0].grid(True, alpha=0.3)
    
    # Plot 2: Histogram with theoretical distribution
    axes[i, 1].hist(X, bins=50, density=True, alpha=0.7, edgecolor='black')
    
    # Theoretical stationary distribution: N(θ, σ²/(2κ))
    stationary_mean = params['theta']
    stationary_std = params['sigma'] / np.sqrt(2 * params['kappa'])
    x_range = np.linspace(X.min(), X.max(), 100)
    theoretical_pdf = (1 / (stationary_std * np.sqrt(2*np.pi))) * \
                      np.exp(-0.5 * ((x_range - stationary_mean) / stationary_std)**2)
    axes[i, 1].plot(x_range, theoretical_pdf, 'r-', linewidth=2, 
                    label=f'Theoretical N({stationary_mean:.2f}, {stationary_std:.2f}²)')
    axes[i, 1].set_title(f'{pc_name} Distribution')
    axes[i, 1].set_xlabel('Value')
    axes[i, 1].set_ylabel('Density')
    axes[i, 1].legend()
    axes[i, 1].grid(True, alpha=0.3)
    
    # Plot 3: Autocorrelation
    max_lag = min(50, len(X)//4)
    acf_values = [np.corrcoef(X[:-lag], X[lag:])[0, 1] if lag > 0 else 1.0 
                  for lag in range(max_lag)]
    
    # Theoretical ACF for OU: ρ(τ) = exp(-κ·τ)
    dt = 1/252
    theoretical_acf = [np.exp(-params['kappa'] * dt * lag) for lag in range(max_lag)]
    
    axes[i, 2].bar(range(max_lag), acf_values, alpha=0.7, label='Empirical')
    axes[i, 2].plot(range(max_lag), theoretical_acf, 'r-', linewidth=2, label='Theoretical OU')
    axes[i, 2].set_title(f'{pc_name} Autocorrelation')
    axes[i, 2].set_xlabel('Lag (days)')
    axes[i, 2].set_ylabel('ACF')
    axes[i, 2].legend()
    axes[i, 2].grid(True, alpha=0.3)

plt.tight_layout()

# Create directory if it doesn't exist
Path('data/ns_parameters').mkdir(parents=True, exist_ok=True)

plt.savefig('data/ns_parameters/ou_diagnostics.png', dpi=300, bbox_inches='tight')
plt.show()

# --- 5. Save Parameters ---

# Save as JSON
with open('data/ns_parameters/ou_parameters.json', 'w') as f:
    json.dump(ou_params, f, indent=2)

# Also save as CSV for easy viewing
ou_df = pd.DataFrame(ou_params).T
ou_df.to_csv('data/ns_parameters/ou_parameters.csv')

print("\n" + "="*70)
print("✓ OU parameters estimated and saved!")
print("  - Parameters (JSON): data/ns_parameters/ou_parameters.json")
print("  - Parameters (CSV):  data/ns_parameters/ou_parameters.csv")
print("  - Diagnostics plot:  data/ns_parameters/ou_diagnostics.png")
print("="*70)

# --- 6. Summary Statistics ---

print("\nSUMMARY:")
print("-" * 70)
print(f"{'PC':<6} {'κ (speed)':<12} {'Half-life':<15} {'σ (vol)':<12} {'Stationary σ':<15}")
print("-" * 70)

for i in range(n_pcs):
    pc_name = f'PC{i+1}'
    params = ou_params[pc_name]
    stat_std = params['sigma'] / np.sqrt(2 * params['kappa'])
    print(f"{pc_name:<6} {params['kappa']:>10.4f}  {params['half_life_days']:>12.1f} d  "
          f"{params['sigma']:>10.4f}  {stat_std:>13.4f}")

print("-" * 70)
print("\nInterpretation:")
print("  - Higher κ → faster mean reversion → less persistent")
print("  - Lower half-life → shocks dissipate quickly")
print("  - Stationary σ → long-run standard deviation of PC")