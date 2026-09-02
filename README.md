# HJM-Libor-Model
Building a no-arbitrage yield curve model from US Treasury data

## The Problem
US Treasury Interest rates are only observed at a handful of maturities (1 month, 1 year, 10 years, etc.). But most financial work — pricing, risk management, stress testing — needs the *entire* curve, at every maturity, evolving realistically through time.

That's harder than it sounds. A curve-fitting method needs to:
- fit the maturities we actually observe,
- stay smooth in between them (no jagged or implausible shapes),
- and evolve over time without creating **arbitrage** — a situation where you could lock in risk-free profit just because two parts of the curve are priced inconsistently with each other.

Simple curve-fitting methods (e.g. spline interpolation) can satisfy the first two but say nothing about the third: they describe a *shape*, not a process that stays internally consistent as time passes and rates move.

## The Solution: HJM
The **Heath–Jarrow–Morton (HJM)** framework solves this by modeling the whole **forward curve** (the market's implied future interest rates, not just the rate today) as a single evolving random process, instead of picking a curve shape and re-fitting it snapshot by snapshot.

The key result that makes HJM special: once you specify how volatile each part of the forward curve is (its **volatility structure** — how much and in what correlated way different maturities jiggle around), the *average direction* those rates must drift in (the **drift**) is no longer a free choice. It's mathematically pinned down by the no-arbitrage condition. You don't have to separately enforce "no arbitrage" as a rule — it falls out of the math automatically.

This is why HJM is generally regarded as superior to more common industry curve models for this purpose:
- **Nelson-Siegel / AFNS** (see `documents/`) fit a smooth curve *shape* with a few interpretable parameters (level, slope, curvature), but Nelson-Siegel itself has no built-in no-arbitrage guarantee — it's a good statistical fit, not a guaranteed-consistent process. (AFNS patches this, but at the cost of extra structure.)
- **Short-rate models** (e.g. Vasicek, CIR, Hull-White) only model a single point on the curve (the short-term rate) and derive the rest from it — simple, but the whole curve's shape is constrained by one process, limiting realism.
- **HJM** models every maturity's dynamics directly and gets no-arbitrage for free, making it the more theoretically complete choice.

## The Catch — and the Workaround
HJM's flexibility is also its problem: the forward curve has *infinitely* many maturities, so specifying a volatility for every single one independently is infeasible — there isn't enough data, and naive attempts overfit noise and become unstable.

**The workaround: PCA.** Empirically, almost all of the day-to-day movement in yield curves can be explained by a handful of common patterns — commonly interpreted as the curve shifting up/down (**level**), tilting (**slope**), and bending (**curvature**). **Principal Component Analysis (PCA)** is a statistical technique that extracts exactly these dominant patterns from historical data. Using just 2–3 PCA factors instead of infinite independent maturities gives HJM a volatility structure that is:
- small enough to estimate reliably from real data,
- interpretable (each factor has an economic meaning),
- and still expressive enough to capture how the curve actually moves.

## Pipeline
1. Ingest and clean historical Treasury yield data
2. Extract dominant factors via PCA
3. Map those factors into an HJM-consistent volatility structure
4. Compute the no-arbitrage drift implied by that volatility structure
5. Simulate future yield curve scenarios and validate them
6. Quantify how uncertain the calibrated volatility/mean-reversion parameters actually are, and propagate that uncertainty into the simulated scenarios instead of hiding it

## A Number Isn't the Same as Knowing the Number
Step 6 above exists because of something this project found the hard way, not something planned from the start.

Each PCA factor's dynamics (how fast it mean-reverts, how volatile it is) are estimated from ~2,260 daily observations via maximum likelihood — a standard, well-understood method that reports a single best-fit number, e.g. "this factor's half-life is 620 days." Early in this project, those numbers came out implausible (some factors reverting in under a week), and two different fixes — smoothing the underlying curve fits, then re-estimating on rolling time windows instead of the full history — were tried and rejected: both were tested directly against the real data rather than assumed to work, and both turned out to move the problem rather than solve it (documented in this repo's git history). The actual fix was upstream, in how the daily curve itself was fit, not in the mean-reversion estimation step.

That process left a residual question, though: even after the real fix, every mean-reversion number reported by this pipeline is still a single point estimate with no attached uncertainty. "Half-life is 620 days" sounds precise. It isn't — it's a best guess from a finite, noisy sample, same as the numbers that turned out to be implausible earlier.

`project/calibration/bayesian_ou.py` addresses this directly: it fits the same underlying model via MCMC (Markov Chain Monte Carlo, using [PyMC](https://www.pymc.io/)) instead of a single-answer optimizer, producing a *distribution* of plausible parameter values (with convergence diagnostics that have to check out before any of it is trusted) instead of one number. `HJMModel.simulate_with_parameter_uncertainty` then draws from that distribution when generating scenarios, so the simulated range of future rates reflects both how uncertain the future is (the usual Monte Carlo randomness) *and* how uncertain the calibration itself is — two different sources of "we don't know," previously conflated into one.

The honest result, tested against real data rather than assumed: propagating parameter uncertainty widens the simulated 1-year-ahead 10-year-rate range by roughly 5–10% (the exact figure moves run to run — both the MCMC sampling and the Monte Carlo simulation are stochastic), with a bigger effect in the tail (the low end shifts down by roughly 20–25 basis points). Modest, not dramatic — reported as found. The point-estimate pipeline (Sections 1–5) remains the default; the Bayesian version is available alongside it, not a silent replacement.

## Scope
This is intended as a transparent, reproducible, interpretable research model — not a production pricing system — useful as a foundation for scenario generation, stress testing, and further fixed-income research.

## Reproducing results
```bash
python -m venv venv && source venv/bin/activate
pip install -r Requirements.txt
pytest                                       # unit tests (network-marked FRED test skipped by default)
python scripts/refresh_treasury_data.py      # pull latest Treasury yields from FRED, re-track with DVC
jupyter nbconvert --to notebook --execute --inplace \
  "notebooks/HJM Term Structure Modeling of U.S. Interest Rates.ipynb"
```
The notebook runs the full pipeline end to end -- data cleaning, Nelson-Siegel
curve fitting, PCA, OU factor calibration, sensitivities, and HJM Monte Carlo
simulation -- writing its artifacts to `data/ns_parameters/`. Each stage is
also directly importable from `project/` (see `project/` package layout)
for use outside the notebook.
