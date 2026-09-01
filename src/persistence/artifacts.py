"""Thin save/load adapters for pipeline artifacts. No computation happens here."""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.calibration.pca import PCAFactorModel
from src.registry import paths as _paths


def save_ns_parameters(params_df: pd.DataFrame, path=None) -> None:
    path = Path(path) if path else _paths.NS_PARAMETERS_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    params_df.to_csv(path)


def load_ns_parameters(path=None) -> pd.DataFrame:
    path = Path(path) if path else _paths.NS_PARAMETERS_CSV
    return pd.read_csv(path, index_col=0, parse_dates=True)


def save_pca_result(model: PCAFactorModel, path=None) -> None:
    path = Path(path) if path else _paths.PCA_MODEL_PKL
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    model.scores.to_csv(path.parent / _paths.PCA_SCORES_CSV.name)
    model.loadings.to_csv(path.parent / _paths.PCA_LOADINGS_CSV.name)


def load_pca_result(path=None) -> PCAFactorModel:
    path = Path(path) if path else _paths.PCA_MODEL_PKL
    with open(path, "rb") as f:
        return pickle.load(f)


def save_ou_parameters(ou_params: dict, path=None) -> None:
    path = Path(path) if path else _paths.OU_PARAMETERS_JSON
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(ou_params, f, indent=2)


def load_ou_parameters(path=None) -> dict:
    path = Path(path) if path else _paths.OU_PARAMETERS_JSON
    with open(path, "r") as f:
        return json.load(f)


def save_sensitivities(mean_params: dict, maturities, pc_sensitivities: pd.DataFrame, path=None) -> None:
    path = Path(path) if path else _paths.SENSITIVITIES_JSON
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "maturities": [float(m) for m in np.asarray(maturities, dtype=float)],
        "mean_parameters": {k: float(v) for k, v in mean_params.items()},
        "pc_sensitivities": {col: pc_sensitivities[col].tolist() for col in pc_sensitivities.columns},
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    pc_sensitivities.to_csv(path.parent / _paths.PC_FORWARD_SENSITIVITIES_CSV.name)


def load_sensitivities(path=None) -> dict:
    path = Path(path) if path else _paths.SENSITIVITIES_JSON
    with open(path, "r") as f:
        return json.load(f)
