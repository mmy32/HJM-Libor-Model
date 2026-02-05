# hjm_simulator.py
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from scipy.integrate import quad
from pathlib import Path
import pickle

class HJMSimulator:
    """
    Heath-Jarrow-Morton simulator using Nelson-Siegel + PCA framework.
    
    Simulates yield curves by evolving principal components and constructing
    curves via Nelson-Siegel parameterization. Supports both physical (P) and
    risk-neutral (Q) measures.
    """
    
    def __init__(self, data_dir='data/ns_parameters'):
        """
        Initialize simulator by loading all model components.
        
        Parameters:
        -----------
        data_dir : str or Path
            Directory containing model parameters
        """
        data_dir = Path(data_dir)
        
        print("="*70)
        print("INITIALIZING HJM SIMULATOR")
        print("="*70)
        
        # Load OU parameters
        with open(data_dir / 'ou_parameters.json', 'r') as f:
            self.ou_params = json.load(f)
        print(f"✓ Loaded OU parameters for {len(self.ou_params)} PCs")
        
        # Load PCA loadings
        self.loadings = pd.read_csv(data_dir / 'pca_loadings.csv', index_col=0)
        print(f"✓ Loaded PCA loadings: {self.loadings.shape}")
        
        # Load sensitivities
        with open(data_dir / 'sensitivities.json', 'r') as f:
            sens_data = json.load(f)
        
        self.maturities = np.array(sens_data['maturities'])
        self.pc_sensitivities = {
            pc: np.array(values) 
            for pc, values in sens_data['pc_sensitivities'].items()
        }
        self.mean_params = sens_data['mean_parameters']
        print(f"✓ Loaded sensitivities for {len(self.maturities)} maturities")
        
        # Extract parameters for easier access
        self.n_pcs = len(self.ou_params)
        self.pc_names = [f'PC{i+1}' for i in range(self.n_pcs)]
        
        # Store volatilities for each PC
        self.pc_vols = np.array([
            self.ou_params[pc]['sigma'] for pc in self.pc_names
        ])
        
        print(f"✓ Model initialized with {self.n_pcs} factors")
        print("="*70)
    
    def nelson_siegel_forward(self, tau_values, b0, b1, b2, lmbda):
        """
        Compute Nelson-Siegel instantaneous forward rate.
        
        f(τ) = β₀ + β₁·exp(-τ/λ) + β₂·(τ/λ)·exp(-τ/λ)
        """
        tau_values = np.asarray(tau_values)
        m = tau_values / lmbda
        exp_m = np.exp(-m)
        
        return b0 + b1 * exp_m + b2 * m * exp_m
    
    def _compute_forward_volatility(self, maturity_idx):
        """
        Compute forward rate volatility at a given maturity index.
        
        σ_forward(T) = Σⱼ (∂f(T)/∂PCⱼ) · σⱼ
        
        Parameters:
        -----------
        maturity_idx : int
            Index into self.maturities array
        
        Returns:
        --------
        float : Forward rate volatility
        """
        vol = 0.0
        for pc in self.pc_names:
            pc_sens = self.pc_sensitivities[pc][maturity_idx]
            pc_vol = self.ou_params[pc]['sigma']
            vol += pc_sens * pc_vol
        
        return vol
    
    def _compute_hjm_drift(self, maturity_idx, t=0):
        """
        Compute HJM drift restriction at a given maturity.
        
        μ(T) = σ(T) · ∫ₜᵀ σ(s) ds
        
        Parameters:
        -----------
        maturity_idx : int
            Index into self.maturities array
        t : float
            Current time (for time-dependent drift)
        
        Returns:
        --------
        float : HJM-consistent drift
        """
        T = self.maturities[maturity_idx]
        
        # Compute volatility at maturity T
        sigma_T = self._compute_forward_volatility(maturity_idx)
        
        # Compute integral ∫ₜᵀ σ(s) ds
        # We'll use trapezoidal rule on our maturity grid
        # Find indices between t and T
        valid_indices = np.where((self.maturities >= t) & (self.maturities <= T))[0]
        
        if len(valid_indices) < 2:
            # If not enough points, approximate as zero
            return 0.0
        
        # Compute volatilities at these maturities
        vols = np.array([self._compute_forward_volatility(i) for i in valid_indices])
        mats = self.maturities[valid_indices]
        
        # Trapezoidal integration
        integral = np.trapz(vols, mats)
        
        # HJM drift
        return sigma_T * integral
    
    def _evolve_pcs_P_measure(self, alpha, dt, dW):
        """
        Evolve PCs under physical measure (historical OU dynamics).
        
        dαⱼ = κⱼ(θⱼ - αⱼ)dt + σⱼ dWⱼ
        """
        d_alpha = np.zeros(self.n_pcs)
        
        for j, pc in enumerate(self.pc_names):
            kappa = self.ou_params[pc]['kappa']
            theta = self.ou_params[pc]['theta']
            sigma = self.ou_params[pc]['sigma']
            
            # OU drift
            drift = kappa * (theta - alpha[j])
            diffusion = sigma * dW[j]
            
            d_alpha[j] = drift * dt + diffusion
        
        return d_alpha
    
    def _evolve_pcs_Q_measure(self, alpha, dt, dW, lambda_risk=None):
        """
        Evolve PCs under risk-neutral measure.
        
        dαⱼ = [κⱼ(θⱼ - αⱼ) - λⱼ]dt + σⱼ dWⱼ
        
        Parameters:
        -----------
        lambda_risk : array-like or None
            Market price of risk for each PC. If None, uses simple approximation.
        """
        if lambda_risk is None:
            # Simple approximation: adjust drift to remove risk premium
            # This is a placeholder - in practice, calibrate to market prices
            lambda_risk = np.zeros(self.n_pcs)
        
        d_alpha = np.zeros(self.n_pcs)
        
        for j, pc in enumerate(self.pc_names):
            kappa = self.ou_params[pc]['kappa']
            theta = self.ou_params[pc]['theta']
            sigma = self.ou_params[pc]['sigma']
            
            # Risk-neutral drift: remove risk premium
            drift = kappa * (theta - alpha[j]) - lambda_risk[j]
            diffusion = sigma * dW[j]
            
            d_alpha[j] = drift * dt + diffusion
        
        return d_alpha
    
    def _pcs_to_ns_params(self, alpha):
        """
        Convert PC scores to NS parameters.
        
        θ = θ̄ + Loadings · α
        """
        # Mean parameters as array [b0, b1, b2, lambda]
        mean_array = np.array([
            self.mean_params['b0_level'],
            self.mean_params['b1_slope'],
            self.mean_params['b2_curvature'],
            self.mean_params['lambda']
        ])
        
        # Loadings is 4x4, alpha is 4x1
        params = mean_array + self.loadings.values @ alpha
        
        return {
            'b0': params[0],
            'b1': params[1],
            'b2': params[2],
            'lambda': params[3]
        }
    
    def _ns_params_to_curve(self, params):
        """
        Generate forward curve from NS parameters.
        """
        return self.nelson_siegel_forward(
            self.maturities,
            params['b0'],
            params['b1'],
            params['b2'],
            params['lambda']
        )
    
    def _forward_to_zero_rates(self, forward_curve):
        """
        Convert instantaneous forward rates to zero rates.
        
        r(0,T) = (1/T) · ∫₀ᵀ f(0,s) ds
        """
        zero_rates = np.zeros(len(self.maturities))
        
        for i, T in enumerate(self.maturities):
            # Integrate forward curve from 0 to T
            mats_up_to_T = self.maturities[:i+1]
            forwards_up_to_T = forward_curve[:i+1]
            
            integral = np.trapz(forwards_up_to_T, mats_up_to_T)
            zero_rates[i] = integral / T if T > 0 else forwards_up_to_T[0]
        
        return zero_rates
    
    def simulate(self, n_paths=1000, T_horizon=1.0, dt=1/252, 
                 measure='P', lambda_risk=None, random_seed=None):
        """
        Simulate yield curve paths.
        
        Parameters:
        -----------
        n_paths : int
            Number of Monte Carlo paths
        T_horizon : float
            Simulation horizon in years
        dt : float
            Time step in years (default: 1/252 = daily)
        measure : str
            'P' for physical measure (forecasting/risk)
            'Q' for risk-neutral measure (pricing)
        lambda_risk : array-like or None
            Market price of risk for each PC (only used if measure='Q')
        random_seed : int or None
            Random seed for reproducibility
        
        Returns:
        --------
        dict with keys:
            'pc_paths': (n_paths, n_steps, n_pcs) - PC score paths
            'forward_curves': (n_paths, n_steps, n_maturities) - Forward rate curves
            'zero_curves': (n_paths, n_steps, n_maturities) - Zero rate curves
            'ns_params': (n_paths, n_steps, 4) - NS parameters
            'time_grid': (n_steps,) - Time points
            'maturities': (n_maturities,) - Maturity points
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        print(f"\n{'='*70}")
        print(f"SIMULATING {n_paths} PATHS")
        print(f"{'='*70}")
        print(f"  Horizon: {T_horizon} years")
        print(f"  Time step: {dt} years ({dt*252:.0f} days)")
        print(f"  Measure: {measure}")
        print(f"  Random seed: {random_seed}")
        
        # Time grid
        n_steps = int(T_horizon / dt) + 1
        time_grid = np.linspace(0, T_horizon, n_steps)
        
        # Preallocate arrays
        pc_paths = np.zeros((n_paths, n_steps, self.n_pcs))
        ns_params_paths = np.zeros((n_paths, n_steps, 4))
        forward_curves = np.zeros((n_paths, n_steps, len(self.maturities)))
        zero_curves = np.zeros((n_paths, n_steps, len(self.maturities)))
        
        # Initial conditions: all PCs start at 0 (mean)
        pc_paths[:, 0, :] = 0.0
        
        # Generate initial curves
        for path in range(n_paths):
            params = self._pcs_to_ns_params(pc_paths[path, 0, :])
            ns_params_paths[path, 0, :] = [params['b0'], params['b1'], 
                                            params['b2'], params['lambda']]
            forward_curves[path, 0, :] = self._ns_params_to_curve(params)
            zero_curves[path, 0, :] = self._forward_to_zero_rates(forward_curves[path, 0, :])
        
        print(f"\n  Simulating {n_steps} time steps...")
        
        # Simulation loop
        for step in range(1, n_steps):
            # Generate random shocks (independent across PCs)
            dW = np.random.randn(n_paths, self.n_pcs) * np.sqrt(dt)
            
            for path in range(n_paths):
                alpha_current = pc_paths[path, step-1, :]
                
                # Evolve PCs based on measure
                if measure == 'P':
                    d_alpha = self._evolve_pcs_P_measure(alpha_current, dt, dW[path])
                elif measure == 'Q':
                    d_alpha = self._evolve_pcs_Q_measure(alpha_current, dt, dW[path], 
                                                         lambda_risk)
                else:
                    raise ValueError(f"measure must be 'P' or 'Q', got '{measure}'")
                
                # Update PCs
                pc_paths[path, step, :] = alpha_current + d_alpha
                
                # Convert to NS parameters
                params = self._pcs_to_ns_params(pc_paths[path, step, :])
                ns_params_paths[path, step, :] = [params['b0'], params['b1'],
                                                   params['b2'], params['lambda']]
                
                # Generate forward curve
                forward_curves[path, step, :] = self._ns_params_to_curve(params)
                
                # Apply HJM drift correction if using Q-measure
                if measure == 'Q':
                    for mat_idx in range(len(self.maturities)):
                        hjm_adjustment = self._compute_hjm_drift(mat_idx, time_grid[step])
                        forward_curves[path, step, mat_idx] += hjm_adjustment * dt
                
                # Convert to zero rates
                zero_curves[path, step, :] = self._forward_to_zero_rates(
                    forward_curves[path, step, :]
                )
        
        print(f"✓ Simulation complete!")
        print(f"{'='*70}")
        
        return {
            'pc_paths': pc_paths,
            'forward_curves': forward_curves,
            'zero_curves': zero_curves,
            'ns_params': ns_params_paths,
            'time_grid': time_grid,
            'maturities': self.maturities,
            'measure': measure,
            'n_paths': n_paths
        }
    
    def plot_sample_paths(self, results, n_sample=5, plot_type='zero_rates'):
        """
        Plot sample simulated paths.
        
        Parameters:
        -----------
        results : dict
            Output from simulate()
        n_sample : int
            Number of paths to plot
        plot_type : str
            'zero_rates', 'forward_rates', or 'pcs'
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        time_grid = results['time_grid']
        maturities = results['maturities']
        
        # Select random paths
        sample_paths = np.random.choice(results['n_paths'], n_sample, replace=False)
        
        # Plot 1: PC paths
        ax = axes[0, 0]
        for pc_idx in range(min(3, self.n_pcs)):  # Plot first 3 PCs
            for path in sample_paths:
                ax.plot(time_grid, results['pc_paths'][path, :, pc_idx], 
                       alpha=0.6, linewidth=1)
            # Add mean
            mean_path = results['pc_paths'][:, :, pc_idx].mean(axis=0)
            ax.plot(time_grid, mean_path, 'k--', linewidth=2, 
                   label=f'PC{pc_idx+1} mean')
        ax.set_xlabel('Time (years)')
        ax.set_ylabel('PC Score')
        ax.set_title('Principal Component Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle=':', alpha=0.3)
        
        # Plot 2: Yield curves at final time
        ax = axes[0, 1]
        for path in sample_paths:
            if plot_type == 'zero_rates':
                ax.plot(maturities, results['zero_curves'][path, -1, :] * 100, 
                       alpha=0.6, linewidth=1.5)
            else:
                ax.plot(maturities, results['forward_curves'][path, -1, :] * 100,
                       alpha=0.6, linewidth=1.5)
        # Add mean curve
        if plot_type == 'zero_rates':
            mean_curve = results['zero_curves'][:, -1, :].mean(axis=0) * 100
        else:
            mean_curve = results['forward_curves'][:, -1, :].mean(axis=0) * 100
        ax.plot(maturities, mean_curve, 'k--', linewidth=2.5, label='Mean')
        ax.set_xlabel('Maturity (years)')
        ax.set_ylabel('Rate (%)')
        ax.set_title(f'Final Yield Curves (T={time_grid[-1]:.2f}y)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: 10Y rate evolution
        ax = axes[1, 0]
        idx_10y = np.argmin(np.abs(maturities - 10.0))
        for path in sample_paths:
            if plot_type == 'zero_rates':
                ax.plot(time_grid, results['zero_curves'][path, :, idx_10y] * 100,
                       alpha=0.6, linewidth=1)
            else:
                ax.plot(time_grid, results['forward_curves'][path, :, idx_10y] * 100,
                       alpha=0.6, linewidth=1)
        # Add confidence bands
        if plot_type == 'zero_rates':
            rates_10y = results['zero_curves'][:, :, idx_10y] * 100
        else:
            rates_10y = results['forward_curves'][:, :, idx_10y] * 100
        mean_10y = rates_10y.mean(axis=0)
        std_10y = rates_10y.std(axis=0)
        ax.plot(time_grid, mean_10y, 'k--', linewidth=2.5, label='Mean')
        ax.fill_between(time_grid, mean_10y - 2*std_10y, mean_10y + 2*std_10y,
                       alpha=0.2, color='gray', label='±2σ')
        ax.set_xlabel('Time (years)')
        ax.set_ylabel('10Y Rate (%)')
        ax.set_title('10-Year Rate Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Distribution of final 10Y rate
        ax = axes[1, 1]
        ax.hist(rates_10y[:, -1], bins=50, alpha=0.7, edgecolor='black', density=True)
        ax.axvline(mean_10y[-1], color='r', linestyle='--', linewidth=2, label='Mean')
        ax.set_xlabel('10Y Rate (%)')
        ax.set_ylabel('Density')
        ax.set_title(f'Distribution at T={time_grid[-1]:.2f}y')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


# --- Usage Example and Testing ---

if __name__ == "__main__":
    # Initialize simulator
    sim = HJMSimulator(data_dir='data/ns_parameters')
    
    print("\n" + "="*70)
    print("RUNNING TEST SIMULATIONS")
    print("="*70)
    
    # Test 1: Physical measure (forecasting)
    print("\n[TEST 1] Physical Measure (P) - Short horizon")
    results_P = sim.simulate(
        n_paths=1000,
        T_horizon=0.25,  # 3 months
        dt=1/252,
        measure='P',
        random_seed=42
    )
    
    # Test 2: Risk-neutral measure (pricing)
    print("\n[TEST 2] Risk-Neutral Measure (Q) - Medium horizon")
    results_Q = sim.simulate(
        n_paths=1000,
        T_horizon=1.0,  # 1 year
        dt=1/52,  # Weekly
        measure='Q',
        random_seed=42
    )
    
    # Plot results
    print("\nGenerating visualizations...")
    fig_P = sim.plot_sample_paths(results_P, n_sample=10)
    fig_P.savefig('data/ns_parameters/hjm_simulation_P_measure.png', 
                  dpi=300, bbox_inches='tight')
    plt.show()
    
    fig_Q = sim.plot_sample_paths(results_Q, n_sample=10)
    fig_Q.savefig('data/ns_parameters/hjm_simulation_Q_measure.png',
                  dpi=300, bbox_inches='tight')
    plt.show()
    
    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    print("\nPhysical Measure (P) at T=0.25y:")
    final_10y_P = results_P['zero_curves'][:, -1, np.argmin(np.abs(results_P['maturities'] - 10))]
    print(f"  10Y rate mean: {final_10y_P.mean()*100:.2f}%")
    print(f"  10Y rate std:  {final_10y_P.std()*100:.2f}%")
    print(f"  10Y rate range: [{final_10y_P.min()*100:.2f}%, {final_10y_P.max()*100:.2f}%]")
    
    print("\nRisk-Neutral Measure (Q) at T=1.0y:")
    final_10y_Q = results_Q['zero_curves'][:, -1, np.argmin(np.abs(results_Q['maturities'] - 10))]
    print(f"  10Y rate mean: {final_10y_Q.mean()*100:.2f}%")
    print(f"  10Y rate std:  {final_10y_Q.std()*100:.2f}%")
    print(f"  10Y rate range: [{final_10y_Q.min()*100:.2f}%, {final_10y_Q.max()*100:.2f}%]")
    
    # Save results
    print("\nSaving simulation results...")
    with open('data/ns_parameters/hjm_results_P.pkl', 'wb') as f:
        pickle.dump(results_P, f)
    with open('data/ns_parameters/hjm_results_Q.pkl', 'wb') as f:
        pickle.dump(results_Q, f)
    
    print("\n✓ All tests complete!")
    print("  - Visualizations saved to: data/ns_parameters/hjm_simulation_*.png")
    print("  - Results saved to: data/ns_parameters/hjm_results_*.pkl")
    print("="*70)