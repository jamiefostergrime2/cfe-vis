import dash
from dash import html
import dash_bootstrap_components as dbc

dash.register_page(
    __name__,
    path="/page-1",
    name="Page 1",
    title="Page 1 | CFE-Vis",
    order=2,
)


def layout():
    return dbc.Container([
        html.H1("Page 1"),
        html.P("Placeholder content."),
    ])
