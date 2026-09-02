"""Raw data fetch from FRED -- no cleaning, no file writes."""

import datetime
import time
import warnings

import pandas as pd
import pandas_datareader.data as web

from project.registry.market_data import DEFAULT_START_DATE, TREASURY_SYMBOL_MAP


def fetch_treasury_yields(start_date=None, end_date=None, symbol_map=None, sleep_seconds=0.2):
    """Fetch raw Treasury constant-maturity yields from FRED, in percentage terms.

    Returns a DataFrame indexed by date with tenor-in-years columns, exactly as
    reported by FRED (not yet converted to decimal or cleaned of gaps).
    """
    symbol_map = symbol_map or TREASURY_SYMBOL_MAP
    start = datetime.datetime.strptime(start_date or DEFAULT_START_DATE, "%Y-%m-%d")
    end = end_date or datetime.date.today()

    series = []
    for symbol, tenor in symbol_map.items():
        try:
            df_symbol = web.DataReader(symbol, "fred", start, end)
        except Exception as exc:
            warnings.warn(f"Failed to fetch {symbol}: {exc}")
            continue
        df_symbol.columns = [tenor]
        series.append(df_symbol)
        if sleep_seconds:
            time.sleep(sleep_seconds)

    return pd.concat(series, axis=1)
