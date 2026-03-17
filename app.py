import dash
from dash import html, Input, Output, State
import dash_bootstrap_components as dbc

app = dash.Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[dbc.themes.FLATLY],
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
    ], fluid=True),
    color="primary", dark=True, sticky="top",
)

app.layout = html.Div([
    navbar,
    dbc.Container(dash.page_container, fluid=True, className="py-4"),
])


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
