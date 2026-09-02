"""Generate the project's self-contained HTML report from the currently
calibrated artifacts under data/ns_parameters/.

Run manually (`python scripts/generate_report.py`) after re-running the
pipeline notebook, whenever the calibration has changed and the report
should reflect it. Includes a live Bayesian OU refit per factor by default
(a few tens of seconds); pass --no-bayesian to skip it for a faster report.
"""

import argparse

from project.reporting.report_builder import build_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-bayesian",
        action="store_true",
        help="Skip the live Bayesian OU refit section (faster, but omits parameter uncertainty).",
    )
    parser.add_argument(
        "--n-sim-paths",
        type=int,
        default=200,
        help="Number of Monte Carlo paths to simulate for the report's scenario figure.",
    )
    args = parser.parse_args()

    print("Building project report from data/ns_parameters/ ...")
    output_path = build_report(
        include_bayesian=not args.no_bayesian,
        n_sim_paths=args.n_sim_paths,
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
