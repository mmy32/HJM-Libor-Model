# Reproducible environment for this project's tests, notebook pipeline,
# backtest, and report generator -- see README's "Reproducing results" for
# the equivalent bare-venv instructions this mirrors. Built to remove
# exactly the kind of "works on my machine" drift this project actually hit
# once already: adding pymc/arviz silently downgraded numpy in the
# maintainer's local venv and broke an unrelated function (see TODO.md /
# git history) until it was caught. A pinned base image doesn't make that
# class of problem impossible, but it makes it reproducible and diffable
# instead of "well it works for me."
FROM python:3.9-slim

# A C/C++ toolchain, in case pip falls back to building any package (e.g.
# QuantLib's Python bindings, or a scientific-stack package without a
# prebuilt wheel for this exact platform) from source instead of finding one.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Installed before the rest of the source is copied in, so an unrelated code
# change doesn't force every dependency to reinstall from scratch.
COPY Requirements.txt .
RUN pip install --no-cache-dir -r Requirements.txt

COPY . .

# Default: run the test suite (the network-marked live-FRED test is skipped
# by default, per pytest.ini -- same behavior as running pytest locally).
# Override at `docker run` time for anything else, e.g.:
#   docker run --rm hjm-libor-model python scripts/run_backtest.py --region test
#   docker run --rm -v "$(pwd)/reports:/app/reports" hjm-libor-model python scripts/generate_report.py
CMD ["pytest"]
