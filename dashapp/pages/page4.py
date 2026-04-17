import dash
from dash import html
import dash_bootstrap_components as dbc

dash.register_page(
    __name__,
    path="/page-4",
    name="Page 4",
    title="Page 4 | CFE-Vis",
    order=5,
)


def layout():
    return dbc.Container([
        html.H1("Page 4"),
        html.P("Placeholder content."),
    ])
