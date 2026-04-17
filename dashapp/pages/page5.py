import dash
from dash import html
import dash_bootstrap_components as dbc

dash.register_page(
    __name__,
    path="/page-5",
    name="Page 5",
    title="Page 5 | CFE-Vis",
    order=6,
)


def layout():
    return dbc.Container([
        html.H1("Page 5"),
        html.P("Placeholder content."),
    ])
