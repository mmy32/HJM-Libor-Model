"""Thin CSV read/write for the raw yield matrix."""
from pathlib import Path

import pandas as pd

from src.registry.paths import RAW_YIELDS_PATH


def save_yield_matrix(df: pd.DataFrame, path=None) -> None:
    path = Path(path) if path else RAW_YIELDS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)


def load_yield_matrix(path=None) -> pd.DataFrame:
    path = Path(path) if path else RAW_YIELDS_PATH
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index)
    return df
