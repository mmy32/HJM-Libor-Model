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
