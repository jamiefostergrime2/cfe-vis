import joblib
import pandas as pd
import dash
import dash_bootstrap_components as dbc
from dash import dcc, callback, Input, Output, State, callback_context
from pathlib import Path

from components.figures import precompute_boundary_data, assemble_boundary_fig

dash.register_page(
    __name__,
    path="/page-2",
    name="Decision Boundaries",
    title="Decision Boundaries | CFE-Vis",
    order=3,
)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"

all_deltas = pd.read_parquet(DATA_DIR / "all_deltas.parquet", engine="fastparquet")
X = pd.read_parquet(DATA_DIR / "X.parquet", engine="fastparquet")
pca_results = pd.read_parquet(DATA_DIR / "cfe_pca_results.parquet", engine="fastparquet")
lr_pipeline = joblib.load(MODELS_DIR / "a-lr.pkl")
en_pipeline = joblib.load(MODELS_DIR / "a-en.pkl")
batch_lr = joblib.load(DATA_DIR / "cfe_batch_lr_20.pkl")
batch_en = joblib.load(DATA_DIR / "cfe_batch_en_20.pkl")

# Expensive computation once at import time — callbacks only call the fast assembler
_DATA = precompute_boundary_data(all_deltas, X, lr_pipeline, en_pipeline, batch_lr=batch_lr, batch_en=batch_en)

# Traces with patient customdata: arrows (1,2,6,7), CFE markers (3,8), originals (4,9),
# PCA scatter bg (14) and highlight (15) — indices shifted by the two legend dummy traces (10,11)
_PATIENT_TRACE_INDICES = {1, 2, 3, 4, 6, 7, 8, 9, 14, 15}


def layout():
    return dbc.Container([
        dcc.Store(id="p2-selected-patient", data=None),

        dbc.Row([
            dbc.Col([
                dbc.Button(
                    "Show Counterfactuals",
                    id="p2-cfe-toggle",
                    color="secondary",
                    outline=True,
                    size="sm",
                    n_clicks=0,
                ),
            ], width="auto"),
            dbc.Col([
                dbc.Label("Direction:", className="me-2 mb-0"),
                dbc.RadioItems(
                    id="p2-direction-filter",
                    options=[
                        {"label": "0→1", "value": "0→1"},
                        {"label": "Both", "value": "Both"},
                        {"label": "1→0", "value": "1→0"},
                    ],
                    value="Both",
                    inline=True,
                ),
            ], width="auto", className="d-flex align-items-center ms-3"),
            dbc.Col([
                dbc.Label("Individual CFE view:", className="me-2 mb-0"),
                dbc.RadioItems(
                    id="p2-indiv-model-filter",
                    options=[
                        {"label": "Both", "value": "Both"},
                        {"label": "LR", "value": "LR"},
                        {"label": "EN", "value": "EN"},
                    ],
                    value="Both",
                    inline=True,
                ),
            ], width="auto", className="d-flex align-items-center ms-3"),
        ], className="mb-2 align-items-center"),

        dcc.Graph(
            id="p2-boundary-graph",
            figure=assemble_boundary_fig(_DATA, pca_results=pca_results),
            style={"height": "calc(100vh - 180px)"},
        ),

    ], fluid=True, className="pt-0")


@callback(
    Output("p2-boundary-graph", "figure"),
    Output("p2-selected-patient", "data"),
    Output("p2-cfe-toggle", "children"),
    Input("p2-cfe-toggle", "n_clicks"),
    Input("p2-direction-filter", "value"),
    Input("p2-boundary-graph", "clickData"),
    Input("p2-indiv-model-filter", "value"),
    State("p2-selected-patient", "data"),
)
def update_boundary(cfe_n_clicks, direction, click_data, indiv_model, selected_patient):
    show_cfe = (cfe_n_clicks or 0) % 2 == 1
    btn_text = "Hide Counterfactuals" if show_cfe else "Show Counterfactuals"

    new_selected = selected_patient
    if callback_context.triggered_id == "p2-boundary-graph" and click_data:
        patient_pts = [
            p for p in click_data["points"]
            if p.get("curveNumber") in _PATIENT_TRACE_INDICES
            and p.get("customdata") is not None
        ]
        if patient_pts:
            clicked = int(patient_pts[0]["customdata"])
            new_selected = None if clicked == selected_patient else clicked

    fig = assemble_boundary_fig(_DATA, show_cfe, direction, new_selected, pca_results, indiv_model)
    return fig, new_selected, btn_text
