"""Refresh data/treasury_yields.csv from FRED and re-track it with DVC.

Run manually (`python scripts/refresh_treasury_data.py`) or on a schedule --
e.g. a daily cron entry or CI job:
    0 18 * * 1-5 cd "/path/to/HJM Libor Model" && venv/bin/python scripts/refresh_treasury_data.py
This project doesn't install that schedule itself, since that's a
machine-level choice for whoever's running it.

After a refresh, downstream artifacts (NS parameters, PCA, OU, sensitivities,
HJM simulation) are stale until the notebook is re-run against the new data.
"""

import subprocess
import sys

from project.data_processing.cleaning import clean_treasury_yields, diagnose_yield_quality
from project.data_processing.io import save_yield_matrix
from project.data_processing.loaders import fetch_treasury_yields
from project.registry import paths


def main():
    print("Fetching live Treasury yields from FRED...")
    raw = fetch_treasury_yields()
    cleaned = clean_treasury_yields(raw)
    save_yield_matrix(cleaned)
    print(f"Saved {len(cleaned)} rows to {paths.RAW_YIELDS_PATH}")

    diagnostics = diagnose_yield_quality(cleaned)
    flagged = {
        tenor: diag
        for tenor, diag in diagnostics.items()
        if diag["stale_flag"] or diag["n_outliers"] > 0
    }
    if flagged:
        print("Data quality flags:")
        for tenor, diag in flagged.items():
            print(
                f"  {tenor}: max_stale_run={diag['max_stale_run']}, n_outliers={diag['n_outliers']}"
            )
    else:
        print("No data quality flags.")

    result = subprocess.run(
        ["dvc", "add", str(paths.RAW_YIELDS_PATH)], capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"dvc add failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    print(result.stdout)
    print(
        f"Tracked with DVC. Commit {paths.RAW_YIELDS_PATH}.dvc (and data/.gitignore if changed) to record this version."
    )


if __name__ == "__main__":
    main()
