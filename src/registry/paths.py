"""Single source of truth for on-disk artifact locations."""
from pathlib import Path

DATA_DIR = Path("data")
RAW_YIELDS_PATH = DATA_DIR / "treasury_yields.csv"

ARTIFACT_DIR = DATA_DIR / "ns_parameters"
NS_PARAMETERS_CSV = ARTIFACT_DIR / "ns_parameters.csv"
PCA_SCORES_CSV = ARTIFACT_DIR / "principal_components.csv"
PCA_LOADINGS_CSV = ARTIFACT_DIR / "pca_loadings.csv"
PCA_MODEL_PKL = ARTIFACT_DIR / "pca_model.pkl"
OU_PARAMETERS_JSON = ARTIFACT_DIR / "ou_parameters.json"
SENSITIVITIES_JSON = ARTIFACT_DIR / "sensitivities.json"
PC_FORWARD_SENSITIVITIES_CSV = ARTIFACT_DIR / "pc_forward_sensitivities.csv"
