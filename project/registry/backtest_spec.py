"""Walk-forward backtest configuration: the rolling-origin schedule, and the
train/validation/test boundary dates that govern which origins' results are
allowed to inform which decisions.

This project's full sample runs 2001-07-31 onward (the cleaned panel's real
start -- see `registry/market_data.py`; data was pulled back to 2000-01-01
but the shortest tenor, DGS1MO, has nothing on FRED before mid-2001). Every
one of this project's existing modeling choices -- holding lambda fixed in
the Nelson-Siegel fit, using 3 PCA factors, the Bayesian OU priors -- was
decided by looking at diagnostics computed across the *2018-2026* slice of
that history, back when that was the entire sample (see TODO.md and the
notebook's Section 4). The 2001-2018 stretch was added later, purely to
widen the backtest's TRAIN region with more regimes (the dot-com bust, the
2008 financial crisis, the zero-rate 2010s) for OU calibration to draw on --
it was never itself looked at while making those design choices, so it's on
the same footing as the rest of TRAIN below: fine to fit through, not a
license to re-litigate decisions that predate it. Reusing the span a model
was designed against to also report its "accuracy" would be close to grading
your own exam. These constants draw an explicit, disclosed line instead:

- TRAIN region (start of data -> VALIDATION_START): the span those earlier
  design decisions were made by looking at. Backtest origins here exist for
  informal development/debugging of the backtest mechanism itself, never for
  a reported number.
- VALIDATION region (VALIDATION_START -> TEST_START): reserved for tuning
  backtest-specific choices that haven't been decided yet -- horizon length,
  step size, minimum training window -- by comparing coverage/RMSE across
  candidate settings. Still not the number that gets reported.
- TEST region (TEST_START -> end of data): touched only to compute the
  final, reported backtest accuracy statistics. Origins here are refit from
  only the data available up to each origin (which reaches back through the
  train/validation regions -- that's normal, not leakage, since it's what
  "fit through today" means for a walk-forward origin actually sitting in
  the test region); what's reserved is *evaluating and reporting results*
  from this span, not the historical data feeding each fit.
"""

from datetime import date

VALIDATION_START = date(2022, 1, 1)
TEST_START = date(2024, 1, 1)

MIN_TRAIN_WINDOW_DAYS = (
    504  # ~2 trading years -- OU mean-reversion needs enough history to identify kappa
)
DEFAULT_HORIZON_DAYS = 63  # ~3 months ahead
DEFAULT_STEP_DAYS = 21  # ~1 month between successive walk-forward origins
DEFAULT_N_PCA_FACTORS = (
    3  # matches the fixed-lambda NS fit's 3 free shape parameters (see curve_spec.py)
)
