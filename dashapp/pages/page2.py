import pickle
import joblib
import warnings
import dash
import dash_bootstrap_components as dbc
from dash import dcc, callback, Input, Output, State, callback_context, no_update
from pathlib import Path

from components.figures import (
    precompute_boundary_data,
    assemble_boundary_fig,
    make_predict_fns,
    _MODEL_DISPLAY,
)

dash.register_page(
    __name__,
    path="/page-2",
    name="Decision Boundaries",
    title="Decision Boundaries | CFE-Vis",
    order=3,
)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"

with open(DATA_DIR / "cfe_data.pkl", "rb") as _f:
    _cfe_data = pickle.load(_f)

# Support both old flat structure (stopgap) and new nested structure
if "pairs" in _cfe_data:
    _ALL_PAIRS = _cfe_data["pairs"]
    _BASE_FEATURE_COLS = _cfe_data.get("base_feature_cols", None)
else:
    _ALL_PAIRS = {
        "LR vs EN": {
            "all_deltas": _cfe_data["all_deltas"],
            "X": _cfe_data["X"],
            "pca_results": _cfe_data["pca_results"],
            "indiv_cfe_batch": _cfe_data["indiv_cfe_batch"],
            "model_a_name": "logistic_regression",
            "model_b_name": "elastic_net",
            "spec_a": None,
            "spec_b": None,
        }
    }
    _BASE_FEATURE_COLS = None

_DEFAULT_PAIR = list(_ALL_PAIRS.keys())[0]

# Load all pipelines once at import time
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    _PIPELINES = {
        "logistic_regression": joblib.load(MODELS_DIR / "a-lr.pkl"),
        "elastic_net": joblib.load(MODELS_DIR / "a-en.pkl"),
    }
    _ac_svm_path = MODELS_DIR / "ac-svm.pkl"
    if _ac_svm_path.exists():
        _PIPELINES["ac_svm"] = joblib.load(_ac_svm_path)

# Precomputed boundary data cache (populated on first access per pair)
_DATA_CACHE: dict = {}

# Trace indices in assemble_boundary_fig that carry patient customdata
_PATIENT_TRACE_INDICES = {1, 2, 3, 4, 6, 7, 8, 9, 14, 15}


def _get_boundary_data(pair_name: str) -> tuple[dict, dict]:
    """Return (precomputed_data, pair_dict) for the given pair, with caching."""
    if pair_name not in _DATA_CACHE:
        pair = _ALL_PAIRS[pair_name]
        X = pair["X"]
        ad = pair["all_deltas"]
        ma = pair["model_a_name"]
        mb = pair["model_b_name"]

        spec_a = pair.get("spec_a")
        spec_b = pair.get("spec_b")
        pipeline_a = _PIPELINES.get(ma)
        pipeline_b = _PIPELINES.get(mb)

        base_cols = _BASE_FEATURE_COLS or list(X.columns)

        if spec_a is not None and pipeline_a is not None:
            proba_a, class_a = make_predict_fns(spec_a, pipeline_a, base_cols)
        else:
            # Fallback: pipeline accepts X directly
            def proba_a(df):
                return pipeline_a.predict_proba(df[base_cols])

            def class_a(df):
                return pipeline_a.predict(df[base_cols]).astype(int)

        if spec_b is not None and pipeline_b is not None:
            proba_b, class_b = make_predict_fns(spec_b, pipeline_b, base_cols)
        else:
            def proba_b(df):
                return pipeline_b.predict_proba(df[base_cols])

            def class_b(df):
                return pipeline_b.predict(df[base_cols]).astype(int)

        _DATA_CACHE[pair_name] = precompute_boundary_data(
            ad, X,
            proba_a, class_a, proba_b, class_b,
            ma, mb,
            indiv_cfe_batch=pair.get("indiv_cfe_batch"),
        )

    return _DATA_CACHE[pair_name], _ALL_PAIRS[pair_name]


def _make_model_options(pair_name: str) -> list:
    pair = _ALL_PAIRS[pair_name]
    disp_a = _MODEL_DISPLAY.get(pair["model_a_name"], pair["model_a_name"])
    disp_b = _MODEL_DISPLAY.get(pair["model_b_name"], pair["model_b_name"])
    return [
        {"label": "Both", "value": "Both"},
        {"label": disp_a, "value": "A"},
        {"label": disp_b, "value": "B"},
    ]


def layout():
    options = _make_model_options(_DEFAULT_PAIR)
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
                    options=options,
                    value="Both",
                    inline=True,
                ),
            ], width="auto", className="d-flex align-items-center ms-3"),
        ], className="mb-2 align-items-center"),

        dcc.Graph(
            id="p2-boundary-graph",
            figure=assemble_boundary_fig(
                _get_boundary_data(_DEFAULT_PAIR)[0],
                pca_results=_ALL_PAIRS[_DEFAULT_PAIR].get("pca_results"),
            ),
            style={"height": "calc(100vh - 180px)"},
        ),

    ], fluid=True, className="pt-0")


@callback(
    Output("p2-boundary-graph", "figure"),
    Output("p2-selected-patient", "data"),
    Output("p2-cfe-toggle", "children"),
    Output("p2-indiv-model-filter", "options"),
    Output("p2-indiv-model-filter", "value"),
    Input("p2-cfe-toggle", "n_clicks"),
    Input("p2-direction-filter", "value"),
    Input("p2-boundary-graph", "clickData"),
    Input("p2-indiv-model-filter", "value"),
    Input("selected-pair", "data"),
    State("p2-selected-patient", "data"),
)
def update_boundary(cfe_n_clicks, direction, click_data, indiv_model, selected_pair, selected_patient):
    pair_name = selected_pair or _DEFAULT_PAIR
    triggered_id = callback_context.triggered_id

    show_cfe = (cfe_n_clicks or 0) % 2 == 1
    btn_text = "Hide Counterfactuals" if show_cfe else "Show Counterfactuals"

    # Pair switch: reset patient selection, model filter
    if triggered_id == "selected-pair":
        data, pair = _get_boundary_data(pair_name)
        opts = _make_model_options(pair_name)
        fig = assemble_boundary_fig(
            data, show_cfe, direction, None,
            pair.get("pca_results"), "Both",
        )
        return fig, None, btn_text, opts, "Both"

    data, pair = _get_boundary_data(pair_name)
    pca_results = pair.get("pca_results")
    model_options = no_update  # options don't change within a pair

    new_selected = selected_patient
    if triggered_id == "p2-boundary-graph" and click_data:
        patient_pts = [
            p for p in click_data["points"]
            if p.get("curveNumber") in _PATIENT_TRACE_INDICES
            and p.get("customdata") is not None
        ]
        if patient_pts:
            clicked = int(patient_pts[0]["customdata"])
            new_selected = None if clicked == selected_patient else clicked

    fig = assemble_boundary_fig(data, show_cfe, direction, new_selected, pca_results, indiv_model)
    return fig, new_selected, btn_text, model_options, no_update
