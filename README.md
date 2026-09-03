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

## The Data
The panel is 11 US Treasury constant-maturity yield series (1 month through 30 years), pulled directly from [FRED](https://fred.stlouisfed.org/), spanning every trading day from 2001-07-31 — as far back as the shortest tenor (the 1-month bill) exists in FRED at all — through the present. That range covers the dot-com bust, the 2004-06 hiking cycle, the 2008 financial crisis and its zero-rate aftermath, the 2015-18 hiking cycle, COVID, and the 2022-23 tightening cycle, not just the calmer 2018-onward window this project started with. `reports/project_report.html`'s "yield curves over time" figure picks five of those regimes and labels each with its actual shape — steep, flat, or inverted, computed from that day's real 10-year/2-year spread, not asserted — and the Fed policy backdrop behind it, rather than five arbitrarily-spaced dates.

## Pipeline
1. Ingest and clean historical Treasury yield data
2. Extract dominant factors via PCA
3. Map those factors into an HJM-consistent volatility structure
4. Compute the no-arbitrage drift implied by that volatility structure
5. Simulate future yield curve scenarios and validate them
6. Quantify how uncertain the calibrated volatility/mean-reversion parameters actually are, and propagate that uncertainty into the simulated scenarios instead of hiding it

## A Number Isn't the Same as Knowing the Number
Step 6 above exists because of something this project found the hard way, not something planned from the start.

Each PCA factor's dynamics (how fast it mean-reverts, how volatile it is) are estimated from ~6,546 daily observations (2001-2026) via maximum likelihood — a standard, well-understood method that reports a single best-fit number, e.g. "this factor's half-life is 620 days." Early in this project, those numbers came out implausible (some factors reverting in under a week), and two different fixes — smoothing the underlying curve fits, then re-estimating on rolling time windows instead of the full history — were tried and rejected: both were tested directly against the real data rather than assumed to work, and both turned out to move the problem rather than solve it (documented in this repo's git history). The actual fix was upstream, in how the daily curve itself was fit, not in the mean-reversion estimation step.

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
python scripts/generate_report.py            # plain-language report for a reader new to the project -> reports/
python scripts/run_backtest.py               # walk-forward forecast-accuracy backtest, held-out test region by default
```
The notebook runs the full pipeline end to end -- data cleaning, Nelson-Siegel
curve fitting, PCA, OU factor calibration, sensitivities, and HJM Monte Carlo
simulation -- writing its artifacts to `data/ns_parameters/`. Each stage is
also directly importable from `project/` (see `project/` package layout)
for use outside the notebook.

`scripts/generate_report.py` (`project/reporting/`) turns those artifacts into
a single self-contained HTML file, `reports/project_report.html`: the same
figures and calibrated numbers as the notebook, but with the explanatory
narrative aimed at someone who hasn't seen the project before, rather than
someone already following a derivation cell by cell. It's regenerated from
whatever is currently in `data/ns_parameters/`, so it goes stale (not silently
wrong, just out of date) if you re-run the notebook without also re-running
the report -- it always states the git commit and timestamp it was built from
so that's easy to check.

`scripts/run_backtest.py` (`project/stochastic/backtest.py`) answers a
question none of the above do: how accurate are this model's forecasts,
actually? It's a walk-forward backtest -- at each of a series of historical
origin dates, the *entire* pipeline (Nelson-Siegel, PCA, OU) is refit using
only data available up to that date, simulated forward, and compared to what
the curve actually did next. `project/registry/backtest_spec.py` documents
the train/validation/test split this draws on: origins before 2022 are where
this project's existing modeling choices (fixed-lambda NS, 3-factor PCA)
were already decided by looking at the data, so they're informal-use only;
2022-2023 is reserved for tuning backtest-specific settings (horizon, step
size); the accuracy numbers actually reported come only from 2024 onward,
untouched until the backtest ran once, for real, against it.

The first honest result was bad: the simulated 90% band covered the realized
rate only 4-43% of the time (tenor-dependent) on the 2024-2026 test region,
and worse -- 4-12% -- on the 2022-2023 hiking cycle used for tuning. That
turned out to be a real bug, not a modeling limitation: `HJMModel.simulate()`
always started every path at PC score 0 -- `mean_params`, the *training
window's average* curve -- rather than the origin date's actual curve, so
every backtest forecast was already offset before a single step of
mean-reversion ran. Worst exactly when it mattered most: during a trending
period, the average curve and the actual one are far apart. Fixed by adding
an `initial_alpha` argument to `simulate()` (see its docstring) and having
the backtest pass each origin's own PC scores instead of the default.

With that fixed, on this project's original 2018-2026 sample, the 2024-2026
test region's coverage came out at **93-97%** against a 90% nominal target,
with RMSE down roughly 5-10x from the buggy version. The 2022-2023 hiking
cycle improved just as much in absolute terms but still fell short --
**52-76%** coverage, worst at short maturities -- which looked like a genuine
limitation, not a bug: 2022-23 was the fastest, most front-loaded tightening
cycle in the sample, and OU `theta` is a trailing-history average with no
mechanism for "policy is currently moving hard in one direction." Both rounds
of this result were reported as found, not smoothed over -- in the same
spirit as "A Number Isn't the Same as Knowing the Number" above: a number
that looks precise (a calibrated `theta`, a 90% band) isn't the same as a
number that's right, and the only way to find out which one you have is to
actually check it against data the model never saw -- twice, in this case,
since the first check itself turned out to have a bug.

That coverage number was also missing two things any accuracy claim needs: a
reference point, and an honest error bar on itself. Both are in now.
`run_backtest` scores a naive random-walk forecast ("tomorrow's curve equals
today's") alongside the model's, and `summarize_backtest` reports a Wilson
confidence interval around the coverage rate rather than a bare percentage
(with a caveat worth stating plainly: it treats each origin as independent,
which consecutive origins from overlapping windows aren't quite -- see
`TODO.md`). `scripts/run_backtest.py --multi-horizon` also now runs the
backtest at several horizons (21/63/126/252 trading days) instead of one, to
show how accuracy actually decays with how far out the forecast reaches.

The training sample was later extended back to 2001 (see "The Data" above),
adding roughly seventeen more years -- the dot-com bust, the 2004-06 hiking
cycle, the 2008 crisis and its zero-rate aftermath, 2015-18 -- to what OU
calibration draws on. Nobody targeted the 2022-23 coverage gap specifically,
but re-running the same walk-forward backtest against the wider history moved
it anyway: validation-region coverage is now **74-87%**, up from 52-76%.
Still short of the 90% nominal target, so the underlying limitation (`theta`
having no notion of "policy is currently moving hard in one direction") isn't
resolved -- but meaningfully closer, plausibly because a longer history gives
`theta` more regime diversity to average over. Worth stating plainly: this
wasn't a deliberate fix, just an observed side effect, and hasn't been
isolated from other things that changed at the same time (more PCA/OU
training rows in general, not specifically more regimes). The numbers below
are from this current, extended sample -- reproducing the pipeline today
gives these, not the original post-fix numbers above.

The result, on the 2024-2026 test region: coverage is **significantly above**
the nominal 90% at every horizon tested (100%, with a 95% CI that never dips
below 98%) -- not just numerically high, but a band that looks wider than it
needs to be to hit 90%, a different (safer, but not free) miscalibration
direction than the under-coverage this project found and fixed the first
time. `skill_vs_naive` -- the fraction of RMSE the model removes relative to
the no-change forecast -- is now **positive at the 63, 126, and 252-day
horizons** (+2% to +30%) and negative only at the shortest, 21-day horizon
(-17%). That's close to a reversal of the original finding on the smaller
sample, where naive won at essentially every horizon and maturity past the
very short end. Interest rates being hard to beat with a point forecast is
still a well-known property of theirs (and exchange rates') -- a negative
21-day reading isn't obviously a sign of a broken model -- but "the point
forecast mostly isn't better than doing nothing" no longer describes this
model's current, wider-trained state as cleanly as it once did. The full
breakdown (by horizon and by tenor) is in `reports/project_report.html`'s
"How accurate is this, really?" section, generated the same way as
everything else in this README: from the model's own results, not asserted,
and its prose is now generated directly from the sign of each horizon's
numbers rather than hand-written, so it can't go stale like the paragraph
above just did.

That naive baseline raised an obvious follow-up: is a naive forecast simply a
low bar, or would a standard ML model do meaningfully better? `run_backtest`
now also fits a `RandomForestRegressor` at every origin (on lagged rate
levels, walk-forward disciplined the same way as everything else --
`stochastic.backtest._fit_ml_baseline_at_origin`), and the answer is more
interesting than either "yes" or "no": `skill_vs_ml`, pooled across tenors,
is **positive at every horizon tested** (+22% to +42%) -- the HJM model beats
the ML baseline on average, even though maturity-by-maturity it's a mixed
picture (losing in the belly of the curve, roughly 3-20 years, winning at the
short and long ends). Combined with the naive result above, the ranking in
RMSE is no longer as clean as "naive best, HJM model second, random forest
worst" -- it now depends on horizon: naive wins at 21 days, but the HJM model
wins at 63 days and beyond, with random forest last throughout. What holds
regardless of horizon is that the theory-informed model holds up against a
generic tabular learner trained the same walk-forward way. Random forest over
gradient boosting was a deliberate, stated choice, not a default -- see that
function's docstring for why a handful of training rows and no separate
validation split makes GBM's extra hyperparameters a real risk of an
unfairly-tuned baseline.

## Docker
```bash
docker build -t hjm-libor-model .
docker run --rm hjm-libor-model                                              # runs the test suite
docker run --rm hjm-libor-model python scripts/run_backtest.py --region test # or any other entry point
docker run --rm -v "$(pwd)/reports:/app/reports" hjm-libor-model \
  python scripts/generate_report.py                                         # mount reports/ to keep the output
```
Pins the Python version and system dependencies this project actually needs
(including QuantLib's compiled bindings) into one reproducible image, so
"works on my machine" stops being a real risk -- which it has been: adding
`pymc`/`arviz` silently downgraded `numpy` in a local dev environment mid-way
through this project and broke an unrelated function until it was caught
(see `TODO.md`/git history). `data/` ships in the image (so the pipeline runs
without a live FRED fetch); `documents/` is excluded via `.dockerignore` --
it's gitignored and deliberately never distributed anywhere, including in a
built image.
