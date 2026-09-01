"""The HJM model: evolves PCA factors under the P or Q measure and reconstructs yield curves.

Unlike the archived HJMSimulator, the constructor takes an in-memory
HJMModelParams object rather than reading files itself, so the model is
unit-testable with tiny synthetic parameters. Use `HJMModel.from_disk` for
the old load-everything-and-go convenience path.
"""
from dataclasses import dataclass, field

import numpy as np

from src.transform.representations import (
    forward_to_zero_rates,
    ns_params_to_curve,
    pcs_to_ns_params,
)


@dataclass
class HJMModelParams:
    ou_params: dict  # {factor_name: {"kappa", "theta", "sigma", ...}}
    loadings: np.ndarray  # (n_ns_params, n_factors), NS params <- PC scores
    pc_sensitivities: np.ndarray  # (n_maturities, n_factors), forward rate <- PC scores
    mean_params: dict  # NS_PARAM_NAMES -> float
    maturities: np.ndarray
    factor_names: list = field(default_factory=list)

    def __post_init__(self):
        if not self.factor_names:
            self.factor_names = list(self.ou_params.keys())
        self.maturities = np.asarray(self.maturities, dtype=float)
        self.loadings = np.asarray(self.loadings, dtype=float)
        self.pc_sensitivities = np.asarray(self.pc_sensitivities, dtype=float)


@dataclass
class SimulationResult:
    pc_paths: np.ndarray
    forward_curves: np.ndarray
    zero_curves: np.ndarray
    ns_params: np.ndarray
    time_grid: np.ndarray
    maturities: np.ndarray
    measure: str
    n_paths: int


class HJMModel:
    """Simulates yield curve paths by evolving PCA factors and reconstructing
    Nelson-Siegel forward/zero curves at each step."""

    def __init__(self, params: HJMModelParams):
        self.params = params
        self.n_factors = len(params.factor_names)
        self.pc_vols = np.array([params.ou_params[name]["sigma"] for name in params.factor_names])
        self.pc_kappas = np.array([params.ou_params[name]["kappa"] for name in params.factor_names])
        self.pc_thetas = np.array([params.ou_params[name]["theta"] for name in params.factor_names])

    @classmethod
    def from_disk(cls, data_dir=None):
        """Convenience constructor: load calibrated parameters from the artifact directory."""
        from pathlib import Path

        from src.persistence import artifacts
        from src.registry.factor_spec import NS_PARAM_NAMES

        ou_path = pca_path = sens_path = None
        if data_dir is not None:
            data_dir = Path(data_dir)
            ou_path = data_dir / "ou_parameters.json"
            pca_path = data_dir / "pca_model.pkl"
            sens_path = data_dir / "sensitivities.json"

        ou_params = artifacts.load_ou_parameters(ou_path)
        pca_model = artifacts.load_pca_result(pca_path)
        sens = artifacts.load_sensitivities(sens_path)

        factor_names = list(ou_params.keys())
        loadings = pca_model.loadings.loc[NS_PARAM_NAMES, factor_names].values
        pc_sensitivities = np.column_stack([sens["pc_sensitivities"][name] for name in factor_names])

        params = HJMModelParams(
            ou_params=ou_params,
            loadings=loadings,
            pc_sensitivities=pc_sensitivities,
            mean_params=sens["mean_parameters"],
            maturities=np.array(sens["maturities"]),
            factor_names=factor_names,
        )
        return cls(params)

    def _forward_volatility(self):
        """sigma_forward(T_i) = sum_j pc_sensitivities[i, j] * pc_vol[j], for every maturity at once."""
        return self.params.pc_sensitivities @ self.pc_vols

    def _hjm_drift(self, forward_vol, t):
        """mu(T) = sigma(T) * integral_t^T sigma(s) ds, trapezoidal over the maturity grid."""
        maturities = self.params.maturities
        drift = np.zeros_like(forward_vol)
        for i, T in enumerate(maturities):
            mask = (maturities >= t) & (maturities <= T)
            if mask.sum() < 2:
                continue
            drift[i] = forward_vol[i] * np.trapezoid(forward_vol[mask], maturities[mask])
        return drift

    def _evolve_pcs(self, alpha, dt, dW, measure="P", lambda_risk=None):
        """dalpha = [kappa*(theta - alpha) - risk_adj] dt + sigma dW, vectorized across paths.

        `alpha`/`dW` are (n_paths, n_factors); risk_adj is zero under the P
        measure and defaults to zero under Q unless `lambda_risk` (market
        price of risk per factor) is supplied.
        """
        if measure not in ("P", "Q"):
            raise ValueError(f"measure must be 'P' or 'Q', got {measure!r}")

        risk_adj = np.zeros(self.n_factors) if lambda_risk is None else np.asarray(lambda_risk)
        drift = self.pc_kappas * (self.pc_thetas - alpha)
        if measure == "Q":
            drift = drift - risk_adj

        return drift * dt + self.pc_vols * dW

    def _reconstruct_curve(self, alpha, drift_adjustment=None):
        """NS params -> forward curve -> zero curve, for one factor vector."""
        ns_params = pcs_to_ns_params(alpha, self.params.mean_params, self.params.loadings)
        forward = ns_params_to_curve(ns_params, self.params.maturities)
        if drift_adjustment is not None:
            forward = forward + drift_adjustment
        zero = forward_to_zero_rates(forward, self.params.maturities)
        return ns_params, forward, zero

    def simulate(
        self, n_paths=1000, T_horizon=1.0, dt=1 / 252, measure="P", lambda_risk=None, random_seed=None
    ) -> SimulationResult:
        """Monte Carlo simulation of yield curve paths under the P or Q measure.

        The PC (factor) SDE step is vectorized across all paths at once.
        Curve reconstruction (NS params -> forward -> zero rates) is still a
        per-path loop -- a further vectorization opportunity noted in
        TODO.md, left for a follow-up since the maturity-grid math involved
        is not yet batch-shaped.
        """
        rng = np.random.default_rng(random_seed)

        maturities = self.params.maturities
        n_maturities = len(maturities)
        n_steps = int(T_horizon / dt) + 1
        time_grid = np.linspace(0, T_horizon, n_steps)

        forward_vol = self._forward_volatility()

        pc_paths = np.zeros((n_paths, n_steps, self.n_factors))
        ns_params_paths = np.zeros((n_paths, n_steps, 4))
        forward_curves = np.zeros((n_paths, n_steps, n_maturities))
        zero_curves = np.zeros((n_paths, n_steps, n_maturities))

        for path in range(n_paths):
            ns_params, forward, zero = self._reconstruct_curve(pc_paths[path, 0, :])
            ns_params_paths[path, 0, :] = [ns_params[k] for k in ("b0", "b1", "b2", "lambda")]
            forward_curves[path, 0, :] = forward
            zero_curves[path, 0, :] = zero

        for step in range(1, n_steps):
            dW = rng.standard_normal((n_paths, self.n_factors)) * np.sqrt(dt)
            alpha_prev = pc_paths[:, step - 1, :]
            d_alpha = self._evolve_pcs(alpha_prev, dt, dW, measure=measure, lambda_risk=lambda_risk)
            pc_paths[:, step, :] = alpha_prev + d_alpha

            drift_adjustment = None
            if measure == "Q":
                drift_adjustment = self._hjm_drift(forward_vol, time_grid[step]) * dt

            for path in range(n_paths):
                ns_params, forward, zero = self._reconstruct_curve(pc_paths[path, step, :], drift_adjustment)
                ns_params_paths[path, step, :] = [ns_params[k] for k in ("b0", "b1", "b2", "lambda")]
                forward_curves[path, step, :] = forward
                zero_curves[path, step, :] = zero

        return SimulationResult(
            pc_paths=pc_paths,
            forward_curves=forward_curves,
            zero_curves=zero_curves,
            ns_params=ns_params_paths,
            time_grid=time_grid,
            maturities=maturities,
            measure=measure,
            n_paths=n_paths,
        )
