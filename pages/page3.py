import dash
from dash import html
import dash_bootstrap_components as dbc

dash.register_page(
    __name__,
    path="/page-3",
    name="Page 3",
    title="Page 3 | CFE-Vis",
    order=4,
)


def layout():
    return dbc.Container([
        html.H1("Page 3"),
        html.P("Placeholder content."),
    ])
