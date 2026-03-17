import dash
from dash import html
import dash_bootstrap_components as dbc

dash.register_page(
    __name__,
    path="/page-2",
    name="Page 2",
    title="Page 2 | CFE-Vis",
    order=3,
)


def layout():
    return dbc.Container([
        html.H1("Page 2"),
        html.P("Placeholder content."),
    ])
