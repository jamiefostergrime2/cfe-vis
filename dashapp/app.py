import pickle
from pathlib import Path

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

with open(DATA_DIR / "cfe_data.pkl", "rb") as _f:
    _cfe_meta = pickle.load(_f)

_PAIR_NAMES = list(_cfe_meta["pairs"].keys())
_DEFAULT_PAIR = _PAIR_NAMES[0]

app = dash.Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True,
)

navbar = dbc.Navbar(
    dbc.Container([
        dbc.NavbarBrand("BIOFLARE-CFE-vis", href="/", className="fw-semibold"),
        dbc.NavbarToggler(id="navbar-toggler"),
        dbc.Collapse(
            dbc.Nav([
                dbc.NavLink(
                    "Disagreement Overview",
                    href="/page-1",
                    active="exact",
                    style={"fontSize": "15px", "padding": "0.6rem 1.1rem"},
                ),
                dbc.NavLink(
                    "Disagreement Geometry",
                    href="/page-2",
                    active="exact",
                    style={"fontSize": "15px", "padding": "0.6rem 1.1rem"},
                ),
            ], navbar=True, className="mx-auto"),
            id="navbar-collapse", navbar=True,
        ),
        html.Div([
            html.Span(
                "Model selector:",
                className="text-light me-2",
                style={"fontSize": "13px", "whiteSpace": "nowrap"},
            ),
            dcc.Dropdown(
                id="pair-dropdown",
                options=[{"label": p, "value": p} for p in _PAIR_NAMES],
                value=_DEFAULT_PAIR,
                clearable=False,
                style={"width": "180px", "color": "#000", "fontSize": "13px"},
            ),
        ], className="d-flex align-items-center"),
    ], fluid=True),
    color="#2b2b2b", dark=True, sticky="top",
)

app.layout = html.Div([
    dcc.Store(id="selected-pair", data=_DEFAULT_PAIR, storage_type="session"),
    navbar,
    dbc.Container(dash.page_container, fluid=True, className="py-4"),
])


@app.callback(
    Output("selected-pair", "data"),
    Input("pair-dropdown", "value"),
)
def update_pair(pair):
    return pair


@app.callback(
    Output("navbar-collapse", "is_open"),
    Input("navbar-toggler", "n_clicks"),
    State("navbar-collapse", "is_open"),
    prevent_initial_call=True,
)
def toggle_navbar(_n, is_open):
    return not is_open


if __name__ == "__main__":
    app.run(debug=True)
