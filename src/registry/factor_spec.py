"""PCA factor-count and Nelson-Siegel parameter-name conventions."""

N_PCA_FACTORS = 4
FACTOR_NAMES = [f"PC{i + 1}" for i in range(N_PCA_FACTORS)]
NS_PARAM_NAMES = ["b0_level", "b1_slope", "b2_curvature", "lambda"]
