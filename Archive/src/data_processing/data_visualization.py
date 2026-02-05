import plotly.graph_objects as go
import pandas as pd

# Load your data
df = pd.read_csv('data/treasury_yields.csv', index_col=0)
tenors = [float(c) for c in df.columns]

# Create the figure
fig = go.Figure()

# Add a trace for each day (but keep them hidden)
for date, row in df.iterrows():
    fig.add_trace(
        go.Scatter(
            visible=False,
            line=dict(color="#00ced1", width=3),
            name=str(date),
            x=tenors,
            y=row.values
        )
    )

# Make the first day visible
fig.data[0].visible = True

# Create the slider steps
steps = []
for i in range(len(fig.data)):
    step = dict(
        method="update",
        args=[{"visible": [False] * len(fig.data)},
              {"title": f"Yield Curve Snapshot: {df.index[i]}"}],
        label=str(df.index[i])[:10] # Show only YYYY-MM-DD
    )
    step["args"][0]["visible"][i] = True
    steps.append(step)

sliders = [dict(active=0, currentvalue={"prefix": "Date: "}, steps=steps)]
fig.update_layout(sliders=sliders, title="Interactive Yield Curve Flipbook",
                  xaxis_title="Tenor (Years)", yaxis_title="Yield (%)")

fig.show()