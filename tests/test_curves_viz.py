import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from project.viz.curves import (
    _classify_curve_shape,
    build_static_curve_overview_figure,
)


def _synthetic_full_history_df():
    """Spans every hardcoded snapshot in `_MARKET_CONDITION_SNAPSHOTS`
    (2003 through 2023) plus a later "latest" date, on a coarse-but-covering
    tenor grid matching the real project's columns."""
    dates = pd.date_range("2002-01-01", "2026-01-01", freq="7D")
    tenors = ["0.0833", "0.25", "0.5", "1.0", "2.0", "3.0", "5.0", "7.0", "10.0", "20.0", "30.0"]
    rng = np.random.default_rng(0)
    base = np.linspace(0.01, 0.05, len(tenors))
    data = base + rng.normal(scale=0.002, size=(len(dates), len(tenors)))
    return pd.DataFrame(data, index=dates, columns=tenors)


def test_classify_curve_shape_thresholds():
    assert _classify_curve_shape(-0.5) == "Inverted"
    assert _classify_curve_shape(-0.01) == "Inverted"
    assert _classify_curve_shape(0.0) == "Flat"
    assert _classify_curve_shape(0.5) == "Flat"
    assert _classify_curve_shape(0.75) == "Steep"
    assert _classify_curve_shape(2.5) == "Steep"


def test_build_static_curve_overview_figure_plots_one_line_per_resolved_snapshot():
    df = _synthetic_full_history_df()
    fig = build_static_curve_overview_figure(df)
    ax = fig.axes[0]

    # All 5 curated snapshots (2003, 2006, 2010, 2023, "latest") fall inside
    # this fixture's date range, so each should resolve to a real row and get
    # plotted -- none silently skipped.
    assert len(ax.get_lines()) == 5


def test_build_static_curve_overview_figure_skips_snapshots_before_the_data_starts():
    dates = pd.date_range("2025-01-01", "2026-01-01", freq="7D")
    tenors = ["0.0833", "0.25", "0.5", "1.0", "2.0", "3.0", "5.0", "7.0", "10.0", "20.0", "30.0"]
    rng = np.random.default_rng(1)
    data = 0.03 + rng.normal(scale=0.002, size=(len(dates), len(tenors)))
    df = pd.DataFrame(data, index=dates, columns=tenors)

    fig = build_static_curve_overview_figure(df)
    ax = fig.axes[0]

    # 2003/2006/2010/2023 all predate this fixture -- only the "latest" (None)
    # snapshot, which always resolves to df.index.max(), should get plotted.
    assert len(ax.get_lines()) == 1


def test_build_static_curve_overview_figure_labels_each_line_with_shape_and_year():
    df = _synthetic_full_history_df()
    fig = build_static_curve_overview_figure(df)
    ax = fig.axes[0]

    texts = [t.get_text() for t in ax.texts]
    headlines = [t for t in texts if "—" in t]
    assert len(headlines) == 5
    for headline in headlines:
        year_part, shape_part = headline.split("—")
        assert year_part.strip().isdigit()
        assert shape_part.strip() in {"Steep", "Flat", "Inverted"}
