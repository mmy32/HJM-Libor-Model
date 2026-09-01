"""Explicit, opt-in plotting style. Importing this package must not mutate global state --
call apply_default_style() yourself if you want it."""
import matplotlib.pyplot as plt
import seaborn as sns


def apply_default_style():
    plt.style.use("seaborn-v0_8-darkgrid")
    sns.set_palette("husl")
