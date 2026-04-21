import dash
import joblib
import pandas as pd
from dash import dcc
import dash_bootstrap_components as dbc
from pathlib import Path

from components.figures import build_boundary_view

dash.register_page(
    __name__,
    path="/page-2",
    name="Decision Boundaries",
    title="Decision Boundaries | CFE-Vis",
    order=3,
)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"

all_deltas = pd.read_parquet(DATA_DIR / "all_deltas.parquet", engine="fastparquet")
X = pd.read_parquet(DATA_DIR / "X.parquet", engine="fastparquet")
lr_pipeline = joblib.load(MODELS_DIR / "a-lr.pkl")
en_pipeline = joblib.load(MODELS_DIR / "a-en.pkl")

_fig = build_boundary_view(all_deltas, X, lr_pipeline, en_pipeline)


def layout():
    return dbc.Container([
        dcc.Graph(figure=_fig),
    ], fluid=True, className="pt-0")
