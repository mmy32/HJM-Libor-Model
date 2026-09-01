# TODO

## Data
- [x] Split raw fetch / cleaning / persistence into `src/data_processing/{loaders,cleaning,io}.py`
- [ ] Add data quality checks/diagnostics (stale quotes, forward-fill runs, outliers) as their own step in `cleaning.py`
- [ ] Add a script/notebook cell to refresh `data/treasury_yields.csv` on a schedule, and track it with DVC

## Curve construction (`src/curves/`)
- [x] Canonical Nelson-Siegel yield/forward formulas and per-day calibration (`src/curves/nelson_siegel.py`)

## Factor extraction / calibration (`src/calibration/`)
- [x] PCA on Nelson-Siegel parameters, with the fitted scaler retained for projecting new data (`src/calibration/pca.py`)
- [x] OU parameter estimation per PCA factor (`src/calibration/ou_process.py`)
- [x] NS-parameter and PC-to-forward-rate sensitivities (`src/calibration/sensitivities.py`)
- [ ] Add calibration diagnostics beyond the notebook plots (residual analysis, factor stability over time, out-of-sample checks)

## Stochastic simulation (`src/stochastic/`)
- [x] HJM no-arbitrage drift, P/Q-measure factor evolution, and Monte Carlo simulation (`src/stochastic/hjm_model.py`)
- [ ] Calibrate a real market price of risk (`lambda_risk`) for the Q-measure -- it currently defaults to zero, an inherited placeholder, not a calibrated quantity
- [ ] Fully vectorize `HJMModel.simulate` across paths -- the PC (factor) SDE step is already vectorized, but curve reconstruction (NS params -> forward -> zero rates) is still a per-path loop
- [ ] Validate simulated curves against historical behavior (beyond the notebook's sample-path diagnostics)

## Testing & tooling
- [x] Unit tests for cleaning, Nelson-Siegel math, PCA, OU estimation, sensitivities, and the HJM model (`tests/`)
- [ ] Set up `pre-commit` hooks (black, etc.) per Requirements.txt
- [ ] Add QuantLib-based benchmark comparison for the scratch-built model
- [ ] Add a network-marked integration test for `loaders.fetch_treasury_yields` (skipped by default, since it hits FRED live)

## Documentation
- [x] Notebook sections 4-8 (NS curve fitting, PCA, OU processes, sensitivities, HJM simulation) written, calling into `src/` rather than defining logic inline
- [ ] Fill in repository link and commit hash placeholders in notebook Section 0
- [ ] Document how to reproduce results end-to-end (setup, data pull, run notebook/scripts)
