"""
theme.py - Shared Plotly template for the CFE-Vis dash app.

Usage:
    from components.theme import TEMPLATE
    fig = go.Figure(data=..., layout=go.Layout(template=TEMPLATE))

    or:
    fig.update_layout(template=TEMPLATE)
"""

import plotly.graph_objects as go

BG_COLOR = "#2b2b2b"
TEXT_COLOR = "#d4d4d4"
GRID_COLOR = "#555555"
CHART_HEIGHT = 800

TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(color=TEXT_COLOR),
        title=dict(font=dict(color=TEXT_COLOR)),
        xaxis=dict(
            gridcolor=GRID_COLOR,
            linecolor=GRID_COLOR,
            tickfont=dict(color=TEXT_COLOR),
        ),
        yaxis=dict(
            gridcolor=GRID_COLOR,
            linecolor=GRID_COLOR,
            tickfont=dict(color=TEXT_COLOR),
        ),
        coloraxis=dict(
            colorbar=dict(
                tickfont=dict(color=TEXT_COLOR),
                title=dict(font=dict(color=TEXT_COLOR)),
                outlinecolor=GRID_COLOR,
            )
        ),
    )
)