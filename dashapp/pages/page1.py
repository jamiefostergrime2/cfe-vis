import dash
import numpy as np
import plotly.graph_objects as go
from dash import dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
from pathlib import Path

from components.figures import build_heatmap, build_scatter
from components.theme import TEMPLATE, CHART_HEIGHT

dash.register_page(
    __name__,
    path="/page-1",
    name="Page 1",
    title="Page 1 | CFE-Vis",
    order=2,
)

# Load data once at import time
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
all_deltas = pd.read_parquet(DATA_DIR / "all_deltas.parquet", engine="fastparquet")

# Compute global scatter scales so axes and colourbar stay fixed across features
norm_cols = [col for col in all_deltas.columns if col.endswith("_norm")]
lr_avg = all_deltas[all_deltas["model"] == "logistic_regression"].groupby("patient_idx")[norm_cols].mean()
en_avg = all_deltas[all_deltas["model"] == "elastic_net"].groupby("patient_idx")[norm_cols].mean()
_shared = lr_avg.index.intersection(en_avg.index)
lr_avg = lr_avg.loc[_shared]
en_avg = en_avg.loc[_shared]

_all_vals = np.concatenate([lr_avg.values.flatten(), en_avg.values.flatten()])
_global_lim = float(np.abs(_all_vals).max()) * 1.05
SCATTER_AXIS_RANGE = [-_global_lim, _global_lim]

_diffs = (en_avg - lr_avg).abs() / np.sqrt(2)
SCATTER_CMAX = float(_diffs.values.max())


def _placeholder_scatter():
    return go.Figure(
        layout=go.Layout(
            template=TEMPLATE,
            height=CHART_HEIGHT,
            title="Click a feature column in the heatmap to explore",
            xaxis=dict(title="Logistic Regression: Normalised delta", range=SCATTER_AXIS_RANGE),
            yaxis=dict(title="Elastic Net: Normalised delta", range=SCATTER_AXIS_RANGE),
        )
    )


def layout():
    return dbc.Container([
        dcc.Store(id="selected-feature"),

        dbc.Row([
            dbc.Col([
                dcc.Graph(id="heatmap", figure=build_heatmap(all_deltas)),
            ], width=7),

            dbc.Col([
                dcc.Graph(id="scatter", figure=_placeholder_scatter()),
            ], width=5),
        ]),
    ], className="pt-0", fluid=True)


@callback(
    Output("scatter", "figure"),
    Output("selected-feature", "data"),
    Input("heatmap", "clickData"),
    prevent_initial_call=True,
)
def update_scatter(heatmap_click):
    feature = heatmap_click["points"][0]["x"]
    return (
        build_scatter(all_deltas, feature, axis_range=SCATTER_AXIS_RANGE, cmax=SCATTER_CMAX),
        feature,
    )
