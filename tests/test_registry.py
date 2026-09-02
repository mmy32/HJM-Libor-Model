from project.registry.curve_spec import NS_PARAM_BOUNDS
from project.registry.factor_spec import FACTOR_NAMES, N_PCA_FACTORS
from project.registry.market_data import TREASURY_SYMBOL_MAP


def test_symbol_map_tenors_are_positive():
    assert all(tenor > 0 for tenor in TREASURY_SYMBOL_MAP.values())


def test_ns_bounds_are_well_formed():
    assert len(NS_PARAM_BOUNDS) == 4
    for lo, hi in NS_PARAM_BOUNDS:
        assert lo < hi


def test_factor_names_match_factor_count():
    assert len(FACTOR_NAMES) == N_PCA_FACTORS
