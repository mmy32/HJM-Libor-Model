# TODO

## Stochastic simulation (`project/stochastic/`)
- [ ] `lambda_risk` (Q-measure market price of risk) is deliberately left at its zero default rather than calibrated. Two blockers, documented in `hjm_model.py::_evolve_pcs`: (1) this project only has spot Treasury yields, not derivative prices, so there's no clean data to calibrate a real price of risk against without building an empirical term-premium estimator (e.g. Fama-Bliss-style excess-return regressions); (2) it's unclear whether `lambda_risk` is even a free parameter in this construction, or redundant with the `_hjm_drift` no-arbitrage adjustment `simulate()` already applies directly to the forward curve under Q -- that relationship hasn't been derived. Needs a methodology decision before any calibration is trustworthy.

## Testing & tooling
- [ ] `scripts/refresh_treasury_data.py` fetches, cleans, quality-checks, and DVC-tracks a fresh pull, but nothing schedules it -- wiring an actual cron/CI schedule is a machine/environment-level choice left to whoever deploys this.
