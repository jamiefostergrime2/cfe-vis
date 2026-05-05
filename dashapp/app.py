import pickle
from pathlib import Path

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

with open(DATA_DIR / "cfe_data.pkl", "rb") as _f:
    _cfe_meta = pickle.load(_f)

# Support both the old flat structure (stopgap) and the new nested structure
if "pairs" in _cfe_meta:
    _PAIR_NAMES = list(_cfe_meta["pairs"].keys())
else:
    _PAIR_NAMES = ["LR vs EN"]
_DEFAULT_PAIR = _PAIR_NAMES[0]

app = dash.Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True,
)

navbar = dbc.Navbar(
    dbc.Container([
        dbc.NavbarBrand("CFE-Vis", href="/"),
        dbc.NavbarToggler(id="navbar-toggler"),
        dbc.Collapse(
            dbc.Nav([
                dbc.NavLink("Page 1", href="/page-1", active="exact"),
                dbc.NavLink("Page 2", href="/page-2", active="exact"),
                dbc.NavLink("Page 3", href="/page-3", active="exact"),
                dbc.NavLink("Page 4", href="/page-4", active="exact"),
                dbc.NavLink("Page 5", href="/page-5", active="exact"),
            ], navbar=True, className="mx-auto"),
            id="navbar-collapse", navbar=True,
        ),
        dcc.Dropdown(
            id="pair-dropdown",
            options=[{"label": p, "value": p} for p in _PAIR_NAMES],
            value=_DEFAULT_PAIR,
            clearable=False,
            style={"width": "180px", "color": "#000", "fontSize": "13px"},
        ),
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
def toggle_navbar(n, is_open):
    return not is_open


if __name__ == "__main__":
    app.run(debug=True)
