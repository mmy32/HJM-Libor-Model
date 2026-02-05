# --- src/curves/nelson_siegel_curves.py ---

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import differential_evolution
from tqdm.auto import tqdm


def nelson_siegel(T, b0, b1, b2, lmbda):
    """
    Nelson–Siegel yield curve:
        y(T) = b0 + b1 * ((1 - e^{-λT})/(λT)) + b2 * (((1 - e^{-λT})/(λT)) - e^{-λT})
    """
    T = np.asarray(T, dtype=float)
    T = np.where(T == 0, 1e-6, T)
    return (
        b0
        + b1 * ((1 - np.exp(-lmbda * T)) / (lmbda * T))
        + b2 * (((1 - np.exp(-lmbda * T)) / (lmbda * T)) - np.exp(-lmbda * T))
    )


def fit_ns_robust(yields, tenors, seed=42):
    """
    Fit Nelson–Siegel parameters via global optimization.
    Returns: np.array([b0, b1, b2, lambda])
    """
    yields = np.asarray(yields, dtype=float)
    tenors = np.asarray(tenors, dtype=float)

    bounds = [(0, 0.15), (-0.1, 0.1), (-0.1, 0.1), (0.01, 2.0)]

    def obj(p):
        return np.sum((yields - nelson_siegel(tenors, *p)) ** 2)

    result = differential_evolution(obj, bounds, seed=seed)
    return result.x


def calibrate_all_days(df, tenors, seed=42):
    """
    Calibrate NS params for every date in df.
    Returns:
      params_df: DataFrame indexed by date with columns [b0_level, b1_slope, b2_curvature, lambda]
      params_dict: dict[Timestamp] -> dict with keys b0,b1,b2,lambda
    """
    print(f"--- Calibrating {len(df)} frames (all days) ---")

    params_dict = {}
    params_list = []
    dates_list = []

    for date, row in tqdm(df.iterrows(), total=len(df), desc="Processing All Days"):
        params = fit_ns_robust(row.values, tenors, seed=seed)

        params_dict[date] = {
            "b0": float(params[0]),
            "b1": float(params[1]),
            "b2": float(params[2]),
            "lambda": float(params[3]),
        }

        params_list.append(params)
        dates_list.append(date)

    params_df = pd.DataFrame(
        params_list,
        index=dates_list,
        columns=["b0_level", "b1_slope", "b2_curvature", "lambda"],
    )

    return params_df, params_dict


def save_parameters(params_df, params_dict, tenors, smooth_tenors, output_dir=Path("data/ns_parameters")):
    """
    Save parameter outputs in CSV / Pickle / NumPy formats.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = output_dir / "ns_parameters.csv"
    params_df.to_csv(csv_path)
    print(f"\n✓ Saved CSV to: {csv_path}")

    # Pickle
    pkl_path = output_dir / "ns_parameters.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(
            {
                "params_df": params_df,
                "params_dict": params_dict,
                "tenors": tenors,
                "smooth_tenors": smooth_tenors,
            },
            f,
        )
    print(f"✓ Saved Pickle to: {pkl_path}")

    # NumPy arrays
    np.save(output_dir / "ns_parameters_array.npy", params_df.values)
    np.save(output_dir / "dates_array.npy", params_df.index.to_numpy())
    print(f"✓ Saved NumPy arrays to: {output_dir / 'ns_parameters_array.npy'}")


def build_visualization(df, params_dict, tenors, smooth_tenors, sample_every=10):
    """
    Build and display an interactive Plotly slider visualization.
    """
    df_sampled = df.iloc[::sample_every]
    fig = go.Figure()

    print(f"\n--- Building visualization with {len(df_sampled)} frames ---")

    for date, row in tqdm(df_sampled.iterrows(), total=len(df_sampled), desc="Building Viz"):
        p = params_dict[date]
        params = [p["b0"], p["b1"], p["b2"], p["lambda"]]
        y_fitted = nelson_siegel(smooth_tenors, *params)

        # Trace A: fitted curve
        fig.add_trace(
            go.Scatter(
                visible=False,
                line=dict(color="#1f77b4", width=3),
                name="NS Fitted Curve",
                x=smooth_tenors,
                y=y_fitted,
            )
        )

        # Trace B: raw points
        fig.add_trace(
            go.Scatter(
                visible=False,
                mode="markers",
                marker=dict(color="#d62728", size=8, symbol="x"),
                name="Raw FRED Data",
                x=tenors,
                y=row.values,
            )
        )

    # Make first frame visible
    if len(fig.data) >= 2:
        fig.data[0].visible = True
        fig.data[1].visible = True

    # Slider logic
    steps = []
    for i in range(0, len(fig.data), 2):
        idx = i // 2
        step = dict(
            method="update",
            args=[
                {"visible": [False] * len(fig.data)},
                {"title": f"Yield Curve Dynamics: {df_sampled.index[idx].date()}"},
            ],
            label=str(df_sampled.index[idx].year),
        )
        step["args"][0]["visible"][i] = True
        step["args"][0]["visible"][i + 1] = True
        steps.append(step)

    y_min = 0
    y_max = float(df.max().max()) + 0.01

    fig.update_layout(
        sliders=[dict(active=0, currentvalue={"prefix": "Date: "}, steps=steps)],
        title="Yield Curve Evolution: Theory vs. Reality",
        xaxis_title="Tenor (Years to Maturity)",
        yaxis_title="Yield (%)",
        template="plotly_white",
        yaxis=dict(range=[y_min, y_max], gridcolor="lightgrey"),
        xaxis=dict(range=[-1, 31], gridcolor="lightgrey"),
        showlegend=True,
        legend=dict(x=0.8, y=0.9),
    )

    fig.show()
    return fig


def main(
    data_path="data/treasury_yields.csv",
    start_date=None,
    output_dir="data/ns_parameters",
    smooth_grid_max=30.0,
    smooth_grid_n=150,
    seed=42,
    viz_sample_every=10,
    build_viz=True,
):
    """
    End-to-end Nelson–Siegel calibration pipeline.

    Parameters
    ----------
    data_path : str
        Path to the Treasury yield CSV.
    start_date : str or None
        If provided, filter df to dates >= start_date (YYYY-MM-DD).
    output_dir : str
        Directory to save fitted parameters.
    smooth_grid_max : float
        Max maturity for smooth evaluation grid (years).
    smooth_grid_n : int
        Number of points on smooth evaluation grid.
    seed : int
        Random seed for differential evolution.
    viz_sample_every : int
        Plot every k-th day for visualization.
    build_viz : bool
        Whether to display Plotly slider visualization.

    Returns
    -------
    params_df : pd.DataFrame
    params_dict : dict
    """
    df = pd.read_csv(data_path, index_col=0)
    df.index = pd.to_datetime(df.index)

    if start_date is not None:
        df = df.loc[df.index >= pd.to_datetime(start_date)]

    tenors = np.array([float(c) for c in df.columns], dtype=float)
    smooth_tenors = np.linspace(0, smooth_grid_max, smooth_grid_n)

    params_df, params_dict = calibrate_all_days(df, tenors, seed=seed)
    save_parameters(params_df, params_dict, tenors, smooth_tenors, output_dir=Path(output_dir))

    fig = None # Initialize fig
    if build_viz:
        # Capture the returned figure object
        fig = build_visualization(df, params_dict, tenors, smooth_tenors, sample_every=viz_sample_every)

    print("\n" + "=" * 60)
    print("PARAMETER SUMMARY STATISTICS")
    print("=" * 60)
    print(params_df.describe())
    print("\n" + "=" * 60)
    print("Data saved and ready for PCA analysis!")
    print("=" * 60)

    return params_df, params_dict, fig


if __name__ == "__main__":
    main()
