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
from plotly.subplots import make_subplots

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


def build_boundary_view(
    all_deltas: pd.DataFrame,
    X: pd.DataFrame,
    lr_pipeline,
    en_pipeline,
    feature_x: str | None = None,
    feature_y: str | None = None,
    grid_n: int = 60,
) -> go.Figure:
    """
    Side-by-side decision boundary plots for LR and EN in original feature space.

    Shows all patients as original points with arrows to their mean CFE destination.
    The two axes are the top-2 most-disagreed features (by normalised delta divergence);
    all other features are held at their population median when computing the boundary.
    """

    # --- Auto-pick top-2 features by model disagreement (same logic as build_heatmap) ---
    if feature_x is None or feature_y is None:
        norm_cols = [c for c in all_deltas.columns if c.endswith("_norm")]
        lr_avg = (all_deltas[all_deltas["model"] == "logistic_regression"]
                  .groupby("patient_idx")[norm_cols].mean())
        en_avg = (all_deltas[all_deltas["model"] == "elastic_net"]
                  .groupby("patient_idx")[norm_cols].mean())
        shared = lr_avg.index.intersection(en_avg.index)
        divergence = (en_avg.loc[shared] - lr_avg.loc[shared]).abs()
        top2 = divergence.sum().sort_values(ascending=False).index[:2]
        feature_x = top2[0].replace("_norm", "")
        feature_y = top2[1].replace("_norm", "")

    # --- Build 60×60 prediction grid in original feature units ---
    margin = 0.05
    x_min, x_max = X[feature_x].min(), X[feature_x].max()
    y_min, y_max = X[feature_y].min(), X[feature_y].max()
    x_pad = (x_max - x_min) * margin
    y_pad = (y_max - y_min) * margin
    x_vals = np.linspace(x_min - x_pad, x_max + x_pad, grid_n)
    y_vals = np.linspace(y_min - y_pad, y_max + y_pad, grid_n)
    xx, yy = np.meshgrid(x_vals, y_vals)

    medians = X.median()
    grid_df = pd.DataFrame(
        np.tile(medians.values, (grid_n * grid_n, 1)),
        columns=X.columns,
    )
    grid_df[feature_x] = xx.ravel()
    grid_df[feature_y] = yy.ravel()

    z_lr = lr_pipeline.predict_proba(grid_df)[:, 1].reshape(grid_n, grid_n)
    z_en = en_pipeline.predict_proba(grid_df)[:, 1].reshape(grid_n, grid_n)

    # --- Compute patient original points and mean CFE endpoints ---
    patients = np.sort(all_deltas["patient_idx"].unique())
    orig_x = X.loc[patients, feature_x].values
    orig_y = X.loc[patients, feature_y].values

    def _cfe_delta(model_str, feat):
        return (
            all_deltas[all_deltas["model"] == model_str]
            .groupby("patient_idx")[f"{feat}_raw"].mean()
            .reindex(patients)
            .values
        )

    lr_cfe_x = orig_x + _cfe_delta("logistic_regression", feature_x)
    lr_cfe_y = orig_y + _cfe_delta("logistic_regression", feature_y)
    en_cfe_x = orig_x + _cfe_delta("elastic_net", feature_x)
    en_cfe_y = orig_y + _cfe_delta("elastic_net", feature_y)

    # --- Shared axis range: union of all points + grid bounds ---
    all_x = np.concatenate([orig_x, lr_cfe_x, en_cfe_x, [x_vals[0], x_vals[-1]]])
    all_y = np.concatenate([orig_y, lr_cfe_y, en_cfe_y, [y_vals[0], y_vals[-1]]])
    finite_x = all_x[np.isfinite(all_x)]
    finite_y = all_y[np.isfinite(all_y)]
    x_range = [float(finite_x.min()) * 0.98, float(finite_x.max()) * 1.02]
    y_range = [float(finite_y.min()) * 0.98, float(finite_y.max()) * 1.02]

    # --- NaN-separated arrow traces (original → CFE endpoint) ---
    def _arrow_trace(ox, oy, cx, cy):
        xs, ys = [], []
        for o_xi, o_yi, c_xi, c_yi in zip(ox, oy, cx, cy):
            xs += [float(o_xi), float(c_xi), None]
            ys += [float(o_yi), float(c_yi), None]
        return xs, ys

    lr_ax, lr_ay = _arrow_trace(orig_x, orig_y, lr_cfe_x, lr_cfe_y)
    en_ax, en_ay = _arrow_trace(orig_x, orig_y, en_cfe_x, en_cfe_y)

    # --- Assemble figure ---
    fig = make_subplots(
        rows=1, cols=2,
        shared_xaxes=True, shared_yaxes=True,
        subplot_titles=["Logistic Regression", "Elastic Net"],
        horizontal_spacing=0.04,
    )

    for col, (z, ax, ay, cfe_x, cfe_y) in enumerate(
        [
            (z_lr, lr_ax, lr_ay, lr_cfe_x, lr_cfe_y),
            (z_en, en_ax, en_ay, en_cfe_x, en_cfe_y),
        ],
        start=1,
    ):
        show = col == 1

        fig.add_trace(
            go.Contour(
                x=x_vals,
                y=y_vals,
                z=z,
                contours=dict(
                    coloring="none",
                    showlines=True,
                    start=0.5,
                    end=0.5,
                    size=1,
                ),
                line=dict(color="#aaaaaa", width=2, dash="dash"),
                showscale=False,
                opacity=0.6,
                name="Decision boundary",
                showlegend=show,
                hoverinfo="skip",
            ),
            row=1, col=col,
        )

        fig.add_trace(
            go.Scatter(
                x=ax,
                y=ay,
                mode="lines",
                line=dict(color="#888888", width=0.8),
                opacity=0.5,
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1, col=col,
        )

        fig.add_trace(
            go.Scatter(
                x=cfe_x,
                y=cfe_y,
                mode="markers",
                marker=dict(color="#f4a261", size=5, opacity=0.7,
                            line=dict(width=0.5, color="grey")),
                name="CFE destination",
                showlegend=show,
                hovertemplate=(
                    f"{feature_x}: %{{x:.3f}}<br>"
                    f"{feature_y}: %{{y:.3f}}"
                    "<extra>CFE</extra>"
                ),
            ),
            row=1, col=col,
        )

        fig.add_trace(
            go.Scatter(
                x=orig_x,
                y=orig_y,
                mode="markers",
                marker=dict(color="#d4d4d4", size=6, opacity=0.85,
                            line=dict(width=0.5, color="grey")),
                name="Original",
                showlegend=show,
                customdata=patients,
                hovertemplate=(
                    "Patient: %{customdata}<br>"
                    f"{feature_x}: %{{x:.3f}}<br>"
                    f"{feature_y}: %{{y:.3f}}"
                    "<extra>Original</extra>"
                ),
            ),
            row=1, col=col,
        )

    fig.update_layout(
        template=TEMPLATE,
        height=CHART_HEIGHT,
        title=f"Decision boundaries: {feature_x} vs {feature_y}",
        showlegend=True,
        legend=dict(x=1.02, y=1),
        margin=dict(b=80),
    )
    fig.update_xaxes(title_text=feature_x, range=x_range)
    fig.update_yaxes(title_text=feature_y, range=y_range)

    fig.add_annotation(
        text="Boundaries computed with all other features held at population median.",
        xref="paper", yref="paper",
        x=0.5, y=-0.12,
        showarrow=False,
        font=dict(color=TEXT_COLOR, size=11),
    )

    return fig