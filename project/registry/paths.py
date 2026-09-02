"""Single source of truth for on-disk artifact locations.

Anchored to the repo root (not left as a bare relative path) because the
caller's working directory isn't reliable: pytest/CLI usage runs from the
repo root, but Jupyter/nbconvert runs a notebook's kernel with cwd set to
the notebook's own directory -- a bare relative "data/..." path would
silently resolve to notebooks/data/ when run from the notebook, diverging
from what every other entry point sees.
"""

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = _PROJECT_ROOT / "data"
RAW_YIELDS_PATH = DATA_DIR / "treasury_yields.csv"

ARTIFACT_DIR = DATA_DIR / "ns_parameters"
NS_PARAMETERS_CSV = ARTIFACT_DIR / "ns_parameters.csv"
PCA_SCORES_CSV = ARTIFACT_DIR / "principal_components.csv"
PCA_LOADINGS_CSV = ARTIFACT_DIR / "pca_loadings.csv"
PCA_MODEL_PKL = ARTIFACT_DIR / "pca_model.pkl"
OU_PARAMETERS_JSON = ARTIFACT_DIR / "ou_parameters.json"
SENSITIVITIES_JSON = ARTIFACT_DIR / "sensitivities.json"
PC_FORWARD_SENSITIVITIES_CSV = ARTIFACT_DIR / "pc_forward_sensitivities.csv"
