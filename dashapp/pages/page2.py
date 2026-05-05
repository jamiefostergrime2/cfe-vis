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
    _select_boundary_features,
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

_ALL_PAIRS = _cfe_data["pairs"]
_BASE_FEATURE_COLS = _cfe_data.get("base_feature_cols", None)

_DEFAULT_PAIR = list(_ALL_PAIRS.keys())[0]

# Load all pipelines once
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    _PIPELINES = {
        "logistic_regression": joblib.load(MODELS_DIR / "a-lr.pkl"),
        "elastic_net": joblib.load(MODELS_DIR / "a-en.pkl"),
    }
    _ac_svm_path = MODELS_DIR / "ac-svm.pkl"
    if _ac_svm_path.exists():
        _PIPELINES["ac_svm"] = joblib.load(_ac_svm_path)

# Build predict function closures and auto-select default feature pairs once at import time.
_PREDICT_FNS: dict = {}
_DEFAULT_FEATURES: dict = {}
for _pn, _pr in _ALL_PAIRS.items():
    _base = _BASE_FEATURE_COLS or list(_pr["X"].columns)
    _ma, _mb = _pr["model_a_name"], _pr["model_b_name"]
    _spa, _spb = _pr.get("spec_a"), _pr.get("spec_b")
    _pla, _plb = _PIPELINES.get(_ma), _PIPELINES.get(_mb)

    if _spa and _pla:
        _fa, _ca = make_predict_fns(_spa, _pla, _base)
    else:
        def _fa(df, _pl=_pla, _bc=_base): return _pl.predict_proba(df[_bc])
        def _ca(df, _pl=_pla, _bc=_base): return _pl.predict(df[_bc]).astype(int)

    if _spb and _plb:
        _fb, _cb = make_predict_fns(_spb, _plb, _base)
    else:
        def _fb(df, _pl=_plb, _bc=_base): return _pl.predict_proba(df[_bc])
        def _cb(df, _pl=_plb, _bc=_base): return _pl.predict(df[_bc]).astype(int)

    _PREDICT_FNS[_pn] = (_fa, _ca, _fb, _cb)
    _DEFAULT_FEATURES[_pn] = _select_boundary_features(_pr["X"], _fa, _fb)

# Boundary data cache keyed by (pair_name, feature_x, feature_y)
_DATA_CACHE: dict = {}

# Trace indices in assemble_boundary_fig that carry patient customdata
_PATIENT_TRACE_INDICES = {1, 2, 3, 4, 6, 7, 8, 9, 14, 15}

# Feature options shared across all dropdowns (11 base features)
_FEATURE_OPTIONS = [
    {"label": f, "value": f}
    for f in (_BASE_FEATURE_COLS or list(list(_ALL_PAIRS.values())[0]["X"].columns))
]


def _get_boundary_data(pair_name: str, feature_x: str, feature_y: str) -> tuple[dict, dict]:
    """Return (precomputed_data, pair_dict), caching by (pair, feature_x, feature_y)."""
    cache_key = (pair_name, feature_x, feature_y)
    if cache_key not in _DATA_CACHE:
        pair = _ALL_PAIRS[pair_name]
        proba_a, class_a, proba_b, class_b = _PREDICT_FNS[pair_name]
        _DATA_CACHE[cache_key] = precompute_boundary_data(
            pair["all_deltas"], pair["X"],
            proba_a, class_a, proba_b, class_b,
            pair["model_a_name"], pair["model_b_name"],
            feature_x=feature_x, feature_y=feature_y,
            indiv_cfe_batch=pair.get("indiv_cfe_batch"),
        )
    return _DATA_CACHE[cache_key], _ALL_PAIRS[pair_name]


def _make_model_options(pair_name: str) -> list:
    pair = _ALL_PAIRS[pair_name]
    disp_a = _MODEL_DISPLAY.get(pair["model_a_name"], pair["model_a_name"])
    disp_b = _MODEL_DISPLAY.get(pair["model_b_name"], pair["model_b_name"])
    return [
        {"label": "Both", "value": "Both"},
        {"label": disp_a, "value": "A"},
        {"label": disp_b, "value": "B"},
    ]


_SELECT_STYLE = {"width": "130px"}


def layout():
    fx, fy = _DEFAULT_FEATURES[_DEFAULT_PAIR]
    options = _make_model_options(_DEFAULT_PAIR)
    data, pair = _get_boundary_data(_DEFAULT_PAIR, fx, fy)

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
            dbc.Col([
                dbc.Label("X axis:", className="me-2 mb-0"),
                dbc.Select(
                    id="p2-feature-x",
                    options=_FEATURE_OPTIONS,
                    value=fx,
                    size="sm",
                    style=_SELECT_STYLE,
                ),
            ], width="auto", className="d-flex align-items-center ms-3"),
            dbc.Col([
                dbc.Label("vs", className="mb-0"),
            ], width="auto", className="d-flex align-items-center ms-2"),
            dbc.Col([
                dbc.Label("Y axis:", className="me-2 mb-0"),
                dbc.Select(
                    id="p2-feature-y",
                    options=_FEATURE_OPTIONS,
                    value=fy,
                    size="sm",
                    style=_SELECT_STYLE,
                ),
            ], width="auto", className="d-flex align-items-center ms-2"),
        ], className="mb-2 align-items-baseline"),

        dcc.Graph(
            id="p2-boundary-graph",
            figure=assemble_boundary_fig(data, pca_results=pair.get("pca_results")),
            style={"height": "calc(100vh - 180px)"},
        ),

    ], fluid=True, className="pt-0")


@callback(
    Output("p2-boundary-graph", "figure"),
    Output("p2-selected-patient", "data"),
    Output("p2-cfe-toggle", "children"),
    Output("p2-indiv-model-filter", "options"),
    Output("p2-indiv-model-filter", "value"),
    Output("p2-feature-x", "value"),
    Output("p2-feature-y", "value"),
    Input("p2-cfe-toggle", "n_clicks"),
    Input("p2-direction-filter", "value"),
    Input("p2-boundary-graph", "clickData"),
    Input("p2-indiv-model-filter", "value"),
    Input("selected-pair", "data"),
    Input("p2-feature-x", "value"),
    Input("p2-feature-y", "value"),
    State("p2-selected-patient", "data"),
)
def update_boundary(
    cfe_n_clicks, direction, click_data, indiv_model,
    selected_pair, feat_x, feat_y, selected_patient,
):
    pair_name = selected_pair or _DEFAULT_PAIR
    triggered_id = callback_context.triggered_id

    show_cfe = (cfe_n_clicks or 0) % 2 == 1
    btn_text = "Hide Counterfactuals" if show_cfe else "Show Counterfactuals"

    # Pair switch: reset features to the best pair for the new pair, reset patient/model filter
    if triggered_id == "selected-pair":
        fx, fy = _DEFAULT_FEATURES.get(pair_name, _DEFAULT_FEATURES[_DEFAULT_PAIR])
        data, pair = _get_boundary_data(pair_name, fx, fy)
        opts = _make_model_options(pair_name)
        fig = assemble_boundary_fig(
            data, show_cfe, direction, None, pair.get("pca_results"), "Both",
        )
        return fig, None, btn_text, opts, "Both", fx, fy

    # Resolve feature values (may be None on initial load edge cases)
    fx = feat_x or _DEFAULT_FEATURES[pair_name][0]
    fy = feat_y or _DEFAULT_FEATURES[pair_name][1]

    data, pair = _get_boundary_data(pair_name, fx, fy)
    pca_results = pair.get("pca_results")

    new_selected = selected_patient
    if triggered_id == "p2-boundary-graph" and click_data:
        patient_pts = [
            p for p in click_data["points"]
            if p.get("curveNumber") in _PATIENT_TRACE_INDICES
            and p.get("customdata") is not None
        ]
        if patient_pts:
            clicked = int(patient_pts[0]["customdata"][0])
            new_selected = None if clicked == selected_patient else clicked

    # Feature change: also clear selected patient (they were selected in a different projection)
    if triggered_id in ("p2-feature-x", "p2-feature-y"):
        new_selected = None

    fig = assemble_boundary_fig(data, show_cfe, direction, new_selected, pca_results, indiv_model)
    return fig, new_selected, btn_text, no_update, no_update, no_update, no_update
