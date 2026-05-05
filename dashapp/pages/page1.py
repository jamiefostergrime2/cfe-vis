import pickle
import dash
import numpy as np
import plotly.graph_objects as go
from dash import dcc, callback, Input, Output, ctx
import dash_bootstrap_components as dbc
from pathlib import Path

from components.figures import build_heatmap, build_scatter
from components.theme import TEMPLATE

dash.register_page(
    __name__,
    path="/page-1",
    name="Page 1",
    title="Page 1 | CFE-Vis",
    order=2,
)

# Load data once at import time
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
with open(DATA_DIR / "cfe_data.pkl", "rb") as _f:
    _cfe_data = pickle.load(_f)
all_deltas = _cfe_data["all_deltas"]

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

_diffs = (en_avg - lr_avg).abs()
SCATTER_CMAX = float(_diffs.values.max())

# Feature ranking by total disagreement — same ordering as the heatmap columns
FEATURE_ORDER = [col.replace("_norm", "") for col in _diffs.sum().sort_values(ascending=False).index]


def _placeholder_scatter():
    return go.Figure(
        layout=go.Layout(
            template=TEMPLATE,
            title="Click a feature column in the heatmap to explore",
            xaxis=dict(title="Logistic Regression: Normalised delta", range=SCATTER_AXIS_RANGE),
            yaxis=dict(title="Elastic Net: Normalised delta", range=SCATTER_AXIS_RANGE),
        )
    )


_GRAPH_HEIGHT = "calc(100vh - 140px)"


def layout():
    return dbc.Container([
        dcc.Store(id="selected-feature"),

        dbc.Row([
            dbc.Col([
                dbc.Button(
                    "Show per-patient detail",
                    id="heatmap-toggle",
                    size="sm",
                    color="secondary",
                    className="mb-1",
                ),
                dcc.Graph(
                    id="heatmap",
                    figure=build_heatmap(all_deltas, condensed=True),
                    style={"height": _GRAPH_HEIGHT},
                ),
            ], width=7),

            dbc.Col([
                dbc.ButtonGroup([
                    dbc.Button("Top 1", id="top1-btn", size="sm", color="secondary"),
                    dbc.Button("Top 3", id="top3-btn", size="sm", color="secondary"),
                    dbc.Button("Top 5", id="top5-btn", size="sm", color="secondary"),
                ], className="mb-1"),
                dcc.Graph(
                    id="scatter",
                    figure=_placeholder_scatter(),
                    style={"height": _GRAPH_HEIGHT},
                ),
            ], width=5),
        ]),
    ], className="pt-0", fluid=True)


@callback(
    Output("heatmap", "figure"),
    Output("heatmap-toggle", "children"),
    Input("heatmap-toggle", "n_clicks"),
    prevent_initial_call=True,
)
def toggle_heatmap(n_clicks):
    condensed = (n_clicks or 0) % 2 == 0
    label = "Show per-patient detail" if condensed else "Show summary"
    return build_heatmap(all_deltas, condensed=condensed), label


@callback(
    Output("scatter", "figure"),
    Output("selected-feature", "data"),
    Input("heatmap", "clickData"),
    Input("top1-btn", "n_clicks"),
    Input("top3-btn", "n_clicks"),
    Input("top5-btn", "n_clicks"),
    prevent_initial_call=True,
)
def update_scatter(heatmap_click, n1, n3, n5):
    triggered = ctx.triggered_id
    if triggered == "heatmap":
        feature = heatmap_click["points"][0]["x"]
    else:
        n = {"top1-btn": 1, "top3-btn": 3, "top5-btn": 5}[triggered]
        feature = FEATURE_ORDER[:n]
    return (
        build_scatter(all_deltas, feature, axis_range=SCATTER_AXIS_RANGE, cmax=SCATTER_CMAX),
        feature,
    )
