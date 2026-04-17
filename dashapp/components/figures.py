"""
figures.py - Plotly figure-building functions for the CFE-Vis dash app.

Each function takes data in and returns a Plotly figure. No Dash logic,
no callbacks — just data in, figure out.

Usage:
    from components.figures import build_heatmap, build_scatter, build_bar
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from components.theme import TEMPLATE, TEXT_COLOR, GRID_COLOR, CHART_HEIGHT


def build_heatmap(all_deltas: pd.DataFrame) -> go.Figure:
    """
    Divergence heatmap: every patient × every feature.

    Clicking a column (feature) will drive the drill-down to the scatter.
    """

    norm_cols = [col for col in all_deltas.columns if col.endswith("_norm")]

    lr_avg = (all_deltas[all_deltas["model"] == "logistic_regression"]
              .groupby("patient_idx")[norm_cols].mean())
    en_avg = (all_deltas[all_deltas["model"] == "elastic_net"]
              .groupby("patient_idx")[norm_cols].mean())

    # Guard against patients where one model failed — keeps indices aligned
    shared = lr_avg.index.intersection(en_avg.index)
    lr_avg = lr_avg.loc[shared]
    en_avg = en_avg.loc[shared]

    divergence = (en_avg - lr_avg).abs()

    # Sort columns by total disagreement (most disagreed features on the left)
    col_order = divergence.sum().sort_values(ascending=False).index
    divergence = divergence[col_order]

    # Sort rows by total disagreement (most disagreed patients at the top)
    divergence = divergence.loc[divergence.sum(axis=1).sort_values(ascending=True).index]

    # Clean feature labels: "acpa_norm" → "acpa"
    feature_labels = [col.replace("_norm", "") for col in col_order]

    fig = go.Figure(
        data=go.Heatmap(
            z=divergence.values,
            x=feature_labels,
            y=[f"Patient {i+1}" for i in divergence.index],
            colorscale="OrRd",
            zmin=0,
            zmax=1,
            colorbar=dict(title="Absolute<br>disagreement"),
            hovertemplate=(
                "Feature: %{x}<br>"
                "Patient: %{y}<br>"
                "Disagreement: %{z:.3f}"
                "<extra></extra>"
            ),
        ),
        layout=go.Layout(
            template=TEMPLATE,
            title="Where do Logistic Regression and Elastic Net diverge?",
            height=CHART_HEIGHT,
            xaxis=dict(title="", tickangle=-45),
            yaxis=dict(
                title=dict(
                    text="Patients (Least to most model disagreement)",
                    standoff=10,
                ),
                showticklabels=False,
            ),
        ),
    )

    fig.update_layout(clickmode="event")

    return fig


def build_scatter(
    all_deltas: pd.DataFrame,
    feature: str,
    height: int = CHART_HEIGHT,
    axis_range: list | None = None,
    cmax: float | None = None,
) -> go.Figure:
    """
    Scatter plot: LR delta vs EN delta for one feature, across all patients.

    Parameters
    ----------
    feature    : clean feature name (e.g. "acpa", not "acpa_norm")
    axis_range : fixed [min, max] for both axes — pass to keep scale consistent
                 across features. If None, auto-scales to the current feature.
    cmax       : fixed colorbar maximum — pass to keep the colour scale
                 consistent across features. If None, uses per-feature max.
    """

    norm_col = f"{feature}_norm"

    lr_deltas = (all_deltas[all_deltas["model"] == "logistic_regression"]
                 .groupby("patient_idx")[norm_col].mean())
    en_deltas = (all_deltas[all_deltas["model"] == "elastic_net"]
                 .groupby("patient_idx")[norm_col].mean())

    shared = lr_deltas.index.intersection(en_deltas.index)
    lr_vals = lr_deltas.loc[shared].values
    en_vals = en_deltas.loc[shared].values
    patient_indices = shared.values

    distance_from_diagonal = np.abs(en_vals - lr_vals) / np.sqrt(2)

    effective_cmax = cmax if cmax is not None else distance_from_diagonal.max()

    fig = go.Figure(
        data=go.Scatter(
            x=lr_vals,
            y=en_vals,
            mode="markers",
            marker=dict(
                size=8,
                color=distance_from_diagonal,
                colorscale="OrRd",
                cmin=0,
                cmax=effective_cmax,
                colorbar=dict(title="Absolute<br>disagreement"),
                line=dict(width=0.5, color="grey"),
                opacity=0.8,
            ),

            customdata=patient_indices.tolist(),

            hovertemplate=(
                "Patient: %{customdata}<br>"
                "LR delta: %{x:.3f}<br>"
                "EN delta: %{y:.3f}"
                "<extra></extra>"
            ),
        ),
        layout=go.Layout(
            template=TEMPLATE,
            height=height,
            title=f"How do the models disagree about {feature}?",
            xaxis=dict(title="Logistic Regression: Normalised delta", range=axis_range),
            yaxis=dict(title="Elastic Net: Normalised delta", range=axis_range),
        ),
    )

    diag_bound = axis_range[1] if axis_range is not None else np.abs(np.concatenate([lr_vals, en_vals])).max()

    # Diagonal reference line
    fig.add_shape(
        type="line",
        x0=-diag_bound, y0=-diag_bound,
        x1=diag_bound, y1=diag_bound,
        line=dict(color="grey", dash="dash", width=1),
    )

    # Zero reference lines
    fig.add_hline(y=0, line=dict(color=GRID_COLOR, width=0.5))
    fig.add_vline(x=0, line=dict(color=GRID_COLOR, width=0.5))

    fig.update_layout(clickmode="event")

    return fig


def build_bar(
    all_deltas: pd.DataFrame,
    X: pd.DataFrame,
    feature: str,
    patient: int,
) -> go.Figure:
    """
    Bar chart: original value vs LR and EN counterfactual values
    for one patient and one feature.

    """

    original_val = X[feature].iloc[patient]

    # Reconstruct counterfactual values from original + raw delta
    raw_col = f"{feature}_raw"

    lr_cf_val = original_val + (all_deltas[
        (all_deltas["patient_idx"] == patient) &
        (all_deltas["model"] == "logistic_regression")
    ][raw_col].mean())

    en_cf_val = original_val + (all_deltas[
        (all_deltas["patient_idx"] == patient) &
        (all_deltas["model"] == "elastic_net")
    ][raw_col].mean())

    labels = ["Original", "Logistic Regression<br>Counterfactual", "Elastic Net<br>Counterfactual"]
    values = [original_val, lr_cf_val, en_cf_val]
    colors = ["#d4d4d4", "#2a9d8f", "#e76f51"]

    fig = go.Figure(
        data=go.Bar(
            x=labels,
            y=values,
            marker=dict(
                color=colors,
                line=dict(width=0.5, color="grey"),
            ),
            text=[f"{v:.2f}" for v in values],
            textposition="outside",
            textfont=dict(color=TEXT_COLOR, size=11),
            width=0.5,

            hovertemplate=(
                "%{x}<br>"
                "Value: %{y:.2f}"
                "<extra></extra>"
            ),
        ),
        layout=go.Layout(
            template=TEMPLATE,
            height = CHART_HEIGHT,
            title=f"Patient {patient}: Counterfactual changes to {feature}",
            yaxis=dict(title=f"{feature} value"),
        ),
    )

    return fig