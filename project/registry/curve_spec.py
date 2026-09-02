"""Nelson-Siegel curve-fitting constants shared across the calibration pipeline."""

import numpy as np

MATURITY_GRID = np.array([0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30])

NS_PARAM_BOUNDS = [(0, 0.15), (-0.1, 0.1), (-0.1, 0.1), (0.01, 2.0)]
NS_OPTIMIZER_SEED = 42

SMOOTH_GRID_MAX = 30.0
SMOOTH_GRID_N = 150
