import dash
from dash import html
import dash_bootstrap_components as dbc

dash.register_page(
    __name__,
    path="/",
    name="Home",
    title="CFE-Vis",
    order=1,
)


def layout():
    return dbc.Container([
        html.H1("Welcome to CFE-Vis"),
        html.P("A tool for visualising and comparing counterfactual explanations across machine learning models."),
    ])
