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
from scipy.cluster.hierarchy import linkage, leaves_list, optimal_leaf_ordering, dendrogram
from scipy.spatial.distance import pdist

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

    # Cluster columns (features) independently
    col_dist = pdist(divergence.values.T, metric="euclidean")
    col_linkage = optimal_leaf_ordering(linkage(col_dist, method="ward"), col_dist)
    col_order_idx = leaves_list(col_linkage)

    # Cluster rows (patients) independently
    row_dist = pdist(divergence.values, metric="euclidean")
    row_linkage = optimal_leaf_ordering(linkage(row_dist, method="ward"), row_dist)
    row_order_idx = leaves_list(row_linkage)

    divergence = divergence.iloc[row_order_idx, :].iloc[:, col_order_idx]

    # Clean feature labels: "acpa_norm" → "acpa"
    feature_labels = [col.replace("_norm", "") for col in divergence.columns]

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
                    text="Patients (Hierarchically clustered)",
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


def build_col_dendrogram(all_deltas: pd.DataFrame) -> go.Figure:
    """
    Column (feature) dendrogram matching the heatmap's cluster ordering.
    Intended to sit below the heatmap for experimental review.
    """
    norm_cols = [col for col in all_deltas.columns if col.endswith("_norm")]

    lr_avg = (all_deltas[all_deltas["model"] == "logistic_regression"]
              .groupby("patient_idx")[norm_cols].mean())
    en_avg = (all_deltas[all_deltas["model"] == "elastic_net"]
              .groupby("patient_idx")[norm_cols].mean())
    shared = lr_avg.index.intersection(en_avg.index)
    divergence = (en_avg.loc[shared] - lr_avg.loc[shared]).abs()

    col_dist = pdist(divergence.values.T, metric="euclidean")
    col_linkage = optimal_leaf_ordering(linkage(col_dist, method="ward"), col_dist)

    dend = dendrogram(col_linkage, no_plot=True)

    n = len(dend["leaves"])
    feature_labels = [norm_cols[i].replace("_norm", "") for i in dend["leaves"]]
    leaf_x = [10 * i + 5 for i in range(n)]

    fig = go.Figure()
    for xs, ys in zip(dend["icoord"], dend["dcoord"]):
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="lines",
            line=dict(color=TEXT_COLOR, width=1),
            showlegend=False,
        ))

    fig.update_layout(
        template=TEMPLATE,
        height=600,
        margin=dict(t=20, b=80, l=60, r=10),
        title="Feature dendrogram",
        xaxis=dict(tickvals=leaf_x, ticktext=feature_labels, tickangle=-45),
        yaxis=dict(title="Distance"),
    )

    return fig


def build_row_dendrogram(all_deltas: pd.DataFrame) -> go.Figure:
    """
    Row (patient) dendrogram matching the heatmap's cluster ordering.
    Intended to sit below the feature dendrogram for experimental review.
    """
    norm_cols = [col for col in all_deltas.columns if col.endswith("_norm")]

    lr_avg = (all_deltas[all_deltas["model"] == "logistic_regression"]
              .groupby("patient_idx")[norm_cols].mean())
    en_avg = (all_deltas[all_deltas["model"] == "elastic_net"]
              .groupby("patient_idx")[norm_cols].mean())
    shared = lr_avg.index.intersection(en_avg.index)
    divergence = (en_avg.loc[shared] - lr_avg.loc[shared]).abs()

    row_dist = pdist(divergence.values, metric="euclidean")
    row_linkage = optimal_leaf_ordering(linkage(row_dist, method="ward"), row_dist)

    dend = dendrogram(row_linkage, no_plot=True)

    fig = go.Figure()
    for xs, ys in zip(dend["icoord"], dend["dcoord"]):
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="lines",
            line=dict(color=TEXT_COLOR, width=1),
            showlegend=False,
        ))

    fig.update_layout(
        template=TEMPLATE,
        height=900,
        margin=dict(t=20, b=40, l=60, r=10),
        title="Patient dendrogram",
        xaxis=dict(showticklabels=False, title="Patients"),
        yaxis=dict(title="Distance"),
    )

    return fig