"""Run the walk-forward backtest and print forecast-accuracy statistics.

By default, restricts origins to the held-out test region
(registry.backtest_spec.TEST_START onward) -- the only origins whose results
this project treats as a reportable accuracy number (see
registry/backtest_spec.py for why). Pass --region to look at the other
regions instead, e.g. for backtest-mechanism development or tuning.
"""

import argparse

import pandas as pd

from project.data_processing.io import load_yield_matrix
from project.registry.backtest_spec import (
    DEFAULT_HORIZON_DAYS,
    DEFAULT_STEP_DAYS,
    TEST_START,
    VALIDATION_START,
)
from project.stochastic.backtest import generate_backtest_origins, run_backtest, summarize_backtest

_REGIONS = {
    "train": (None, VALIDATION_START),
    "validation": (VALIDATION_START, TEST_START),
    "test": (TEST_START, None),
    "all": (None, None),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", choices=sorted(_REGIONS), default="test")
    parser.add_argument("--horizon-days", type=int, default=DEFAULT_HORIZON_DAYS)
    parser.add_argument("--step-days", type=int, default=DEFAULT_STEP_DAYS)
    parser.add_argument("--n-paths", type=int, default=300)
    parser.add_argument("--random-seed", type=int, default=0)
    args = parser.parse_args()

    start, end = _REGIONS[args.region]

    yields_df = load_yield_matrix()
    origins = generate_backtest_origins(
        yields_df.index, start=start, end=end, step=args.step_days, horizon=args.horizon_days
    )
    if not origins:
        print(f"No eligible origins in the '{args.region}' region with these settings.")
        return
    print(
        f"Backtesting {len(origins)} origins in the '{args.region}' region "
        f"({origins[0].date()} to {origins[-1].date()}), {args.horizon_days}-day horizon..."
    )

    results = run_backtest(
        yields_df,
        origins=origins,
        horizon_days=args.horizon_days,
        n_paths=args.n_paths,
        random_seed=args.random_seed,
    )
    summary = summarize_backtest(results)

    pd.set_option("display.width", 120)
    print()
    print(summary.round(4))


if __name__ == "__main__":
    main()
