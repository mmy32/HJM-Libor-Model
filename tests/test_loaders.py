import pytest

from project.data_processing.loaders import fetch_treasury_yields


@pytest.mark.network
def test_fetch_treasury_yields_returns_recent_data():
    """Hits live FRED. Skipped by default (pytest.ini: `-m "not network"`);
    run explicitly with `pytest -m network` when you want to confirm the
    live fetch path still works (e.g. after a pandas-datareader/FRED change)."""
    df = fetch_treasury_yields(start_date="2024-01-01")
    assert len(df) > 0
    assert not df.empty
    assert df.iloc[-1].notna().any()
