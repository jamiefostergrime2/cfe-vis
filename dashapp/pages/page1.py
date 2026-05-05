import pickle
import dash
import numpy as np
import plotly.graph_objects as go
from dash import dcc, callback, Input, Output, State, ctx, no_update
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

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

with open(DATA_DIR / "cfe_data.pkl", "rb") as _f:
    _cfe_data = pickle.load(_f)

# Support both old flat structure (stopgap) and new nested structure
if "pairs" in _cfe_data:
    _ALL_PAIRS = _cfe_data["pairs"]
else:
    # Wrap flat structure in the expected nested format
    _ALL_PAIRS = {
        "LR vs EN": {
            "all_deltas": _cfe_data["all_deltas"],
            "model_a_name": "logistic_regression",
            "model_b_name": "elastic_net",
        }
    }

_DEFAULT_PAIR = list(_ALL_PAIRS.keys())[0]


def _compute_pair_meta(pair_name: str) -> dict:
    """Pre-compute the derived axis/scale values for a given pair."""
    pair = _ALL_PAIRS[pair_name]
    ad = pair["all_deltas"]
    ma = pair["model_a_name"]
    mb = pair["model_b_name"]

    norm_cols = [c for c in ad.columns if c.endswith("_norm")]
    a_avg = ad[ad["model"] == ma].groupby("patient_idx")[norm_cols].mean()
    b_avg = ad[ad["model"] == mb].groupby("patient_idx")[norm_cols].mean()
    shared = a_avg.index.intersection(b_avg.index)
    a_avg = a_avg.loc[shared]
    b_avg = b_avg.loc[shared]

    all_vals = np.concatenate([a_avg.values.flatten(), b_avg.values.flatten()])
    global_lim = float(np.abs(all_vals).max()) * 1.05

    diffs = (b_avg - a_avg).abs()
    feature_order = [c.replace("_norm", "") for c in diffs.sum().sort_values(ascending=False).index]

    return {
        "axis_range": [-global_lim, global_lim],
        "scatter_cmax": float(diffs.values.max()),
        "feature_order": feature_order,
    }


# Pre-compute metadata for all pairs at import time (fast — no model calls)
_PAIR_META = {name: _compute_pair_meta(name) for name in _ALL_PAIRS}


def _placeholder_scatter(axis_range: list) -> go.Figure:
    return go.Figure(
        layout=go.Layout(
            template=TEMPLATE,
            title="Click a feature column in the heatmap to explore",
            xaxis=dict(title="Model A: Normalised delta", range=axis_range),
            yaxis=dict(title="Model B: Normalised delta", range=axis_range),
        )
    )


_GRAPH_HEIGHT = "calc(100vh - 140px)"


def layout():
    meta = _PAIR_META[_DEFAULT_PAIR]
    pair = _ALL_PAIRS[_DEFAULT_PAIR]
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
                    figure=build_heatmap(
                        pair["all_deltas"], condensed=True,
                        model_a_name=pair["model_a_name"],
                        model_b_name=pair["model_b_name"],
                    ),
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
                    figure=_placeholder_scatter(meta["axis_range"]),
                    style={"height": _GRAPH_HEIGHT},
                ),
            ], width=5),
        ]),
    ], className="pt-0", fluid=True)


@callback(
    Output("heatmap", "figure"),
    Output("heatmap-toggle", "children"),
    Output("scatter", "figure"),
    Output("selected-feature", "data"),
    Input("heatmap-toggle", "n_clicks"),
    Input("selected-pair", "data"),
    Input("heatmap", "clickData"),
    Input("top1-btn", "n_clicks"),
    Input("top3-btn", "n_clicks"),
    Input("top5-btn", "n_clicks"),
    State("selected-feature", "data"),
    prevent_initial_call=True,
)
def update_page(toggle_clicks, selected_pair, heatmap_click, n1, n3, n5, current_feature):
    triggered = ctx.triggered_id
    pair_name = selected_pair or _DEFAULT_PAIR
    pair = _ALL_PAIRS[pair_name]
    meta = _PAIR_META[pair_name]
    ad = pair["all_deltas"]
    ma = pair["model_a_name"]
    mb = pair["model_b_name"]

    condensed = (toggle_clicks or 0) % 2 == 0

    if triggered == "selected-pair":
        heatmap_fig = build_heatmap(ad, condensed=True, model_a_name=ma, model_b_name=mb)
        scatter_fig = _placeholder_scatter(meta["axis_range"])
        return heatmap_fig, "Show per-patient detail", scatter_fig, None

    if triggered == "heatmap-toggle":
        label = "Show per-patient detail" if condensed else "Show summary"
        heatmap_fig = build_heatmap(ad, condensed=condensed, model_a_name=ma, model_b_name=mb)
        if current_feature:
            scatter_fig = build_scatter(
                ad, current_feature,
                axis_range=meta["axis_range"], cmax=meta["scatter_cmax"],
                model_a_name=ma, model_b_name=mb,
            )
        else:
            scatter_fig = _placeholder_scatter(meta["axis_range"])
        return heatmap_fig, label, scatter_fig, current_feature

    # Feature selection triggers (heatmap click or top-N buttons)
    if triggered == "heatmap":
        feature = heatmap_click["points"][0]["x"]
    else:
        n = {"top1-btn": 1, "top3-btn": 3, "top5-btn": 5}[triggered]
        feature = meta["feature_order"][:n]

    scatter_fig = build_scatter(
        ad, feature,
        axis_range=meta["axis_range"], cmax=meta["scatter_cmax"],
        model_a_name=ma, model_b_name=mb,
    )
    return no_update, no_update, scatter_fig, feature
