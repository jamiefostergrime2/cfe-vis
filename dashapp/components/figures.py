"""
figures.py - Plotly figure-building functions for the CFE-Vis dash app.

Each function takes data in and returns a Plotly figure. No Dash logic,
no callbacks — just data in, figure out.

Usage:
    from components.figures import (
        build_heatmap, build_scatter, build_bar,
        precompute_boundary_data, assemble_boundary_fig,
    )
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


def precompute_boundary_data(
    all_deltas: pd.DataFrame,
    X: pd.DataFrame,
    lr_pipeline,
    en_pipeline,
    feature_x: str | None = None,
    feature_y: str | None = None,
    grid_n: int = 60,
) -> dict:
    """
    Run all expensive computation (grid prediction, CFE aggregation) once.

    Returns a plain dict consumed by assemble_boundary_fig. Separating this
    from the figure builder means model calls happen only at page load, not
    on every user interaction.
    """

    # Auto-pick top-2 features by model disagreement (same logic as build_heatmap)
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

    # Build grid in original feature units — all other features at population median
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

    # Patient original points and mean CFE endpoints
    patients = np.sort(all_deltas["patient_idx"].unique())
    orig_x = X.loc[patients, feature_x].values
    orig_y = X.loc[patients, feature_y].values

    def _mean_cfe_delta(model_str: str, feat: str) -> np.ndarray:
        return (
            all_deltas[all_deltas["model"] == model_str]
            .groupby("patient_idx")[f"{feat}_raw"].mean()
            .reindex(patients)
            .values
        )

    lr_cfe_x = orig_x + _mean_cfe_delta("logistic_regression", feature_x)
    lr_cfe_y = orig_y + _mean_cfe_delta("logistic_regression", feature_y)
    en_cfe_x = orig_x + _mean_cfe_delta("elastic_net", feature_x)
    en_cfe_y = orig_y + _mean_cfe_delta("elastic_net", feature_y)

    # Shared axis range: union of all finite points + grid bounds
    all_x = np.concatenate([orig_x, lr_cfe_x, en_cfe_x, [x_vals[0], x_vals[-1]]])
    all_y = np.concatenate([orig_y, lr_cfe_y, en_cfe_y, [y_vals[0], y_vals[-1]]])
    finite_x = all_x[np.isfinite(all_x)]
    finite_y = all_y[np.isfinite(all_y)]
    x_range = [float(finite_x.min()) * 0.98, float(finite_x.max()) * 1.02]
    y_range = [float(finite_y.min()) * 0.98, float(finite_y.max()) * 1.02]

    # Original class predictions — used for direction filter (0→1 vs 1→0)
    lr_preds = lr_pipeline.predict(X.loc[patients]).astype(int)
    en_preds = en_pipeline.predict(X.loc[patients]).astype(int)

    # Individual (unaggregated) CFE positions for the detail subplot
    orig_x_ser = X[feature_x]
    orig_y_ser = X[feature_y]
    indiv = all_deltas[["patient_idx", "model", f"{feature_x}_raw", f"{feature_y}_raw"]].copy()
    indiv["orig_x"] = indiv["patient_idx"].map(orig_x_ser)
    indiv["orig_y"] = indiv["patient_idx"].map(orig_y_ser)
    indiv["cfe_x"] = indiv["orig_x"] + indiv[f"{feature_x}_raw"]
    indiv["cfe_y"] = indiv["orig_y"] + indiv[f"{feature_y}_raw"]

    return dict(
        feature_x=feature_x,
        feature_y=feature_y,
        x_vals=x_vals,
        y_vals=y_vals,
        z_lr=z_lr,
        z_en=z_en,
        patients=patients,
        orig_x=orig_x,
        orig_y=orig_y,
        lr_cfe_x=lr_cfe_x,
        lr_cfe_y=lr_cfe_y,
        en_cfe_x=en_cfe_x,
        en_cfe_y=en_cfe_y,
        lr_preds=lr_preds,
        en_preds=en_preds,
        x_range=x_range,
        y_range=y_range,
        indiv_cfe=indiv[["patient_idx", "model", "orig_x", "orig_y", "cfe_x", "cfe_y"]].reset_index(drop=True),
    )


def _direction_mask(preds: np.ndarray, direction: str) -> np.ndarray:
    if direction == "0→1":
        return preds == 0
    if direction == "1→0":
        return preds == 1
    return np.ones(len(preds), dtype=bool)


def _arrow_segments(
    ox: np.ndarray,
    oy: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    mask: np.ndarray,
    patients: np.ndarray,
) -> tuple[list, list, list]:
    """Build NaN-separated line arrays for patients where mask is True.

    Returns xs, ys, and a parallel customdata list (patient index repeated
    for origin and destination, None for separators) so click events on line
    traces carry patient identity.
    """
    xs: list = []
    ys: list = []
    cdata: list = []
    for i in range(len(ox)):
        if mask[i] and np.isfinite(cx[i]) and np.isfinite(cy[i]):
            p = int(patients[i])
            xs += [float(ox[i]), float(cx[i]), None]
            ys += [float(oy[i]), float(cy[i]), None]
            cdata += [p, p, None]
    return xs, ys, cdata


def _point_opacities(
    mask: np.ndarray,
    patients: np.ndarray,
    selected_patient: int | None,
    full: float,
    dim: float,
) -> list[float]:
    """Per-point opacity list for a Scatter trace matching the patients array."""
    result = []
    for i, p in enumerate(patients):
        if not mask[i]:
            result.append(0.0)
        elif selected_patient is not None and p != selected_patient:
            result.append(dim)
        else:
            result.append(full)
    return result


def _point_sizes(
    patients: np.ndarray,
    selected_patient: int | None,
    normal: int,
    highlighted: int,
) -> list[int]:
    """Per-point size list — highlighted patient gets a larger marker."""
    if selected_patient is None:
        return [normal] * len(patients)
    return [highlighted if p == selected_patient else normal for p in patients]


def assemble_boundary_fig(
        data: dict,
        show_cfe: bool = False,
        direction: str = "Both",
        selected_patient: int | None = None,
) -> go.Figure:
    """
    Build the decision boundary figure from pre-computed data.
    """
    feature_x = data["feature_x"]
    feature_y = data["feature_y"]
    patients = data["patients"]

    # FIX: Create a single unified mask that only includes patients
    # where BOTH models agree on the starting prediction direction.
    unified_mask = _direction_mask(data["lr_preds"], direction) & \
                   _direction_mask(data["en_preds"], direction)

    _LR_COLOR = "#2a9d8f"
    _EN_COLOR = "#e76f51"

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Logistic Regression", "Elastic Net",
            "Individual counterfactuals (both models)", "",
        ],
        horizontal_spacing=0.04,
        vertical_spacing=0.10,
        row_heights=[0.5, 0.5],
    )

    # Standard hover templates...
    hover_cfe = "Patient: %{customdata}<br>Value: %{x:.3f}, %{y:.3f}<extra>CFE</extra>"
    hover_arrow = "Patient: %{customdata}<extra>Arrow</extra>"
    hover_orig = "Patient: %{customdata}<br>Value: %{x:.3f}, %{y:.3f}<extra>Original</extra>"

    # We iterate through the two models, but use the SAME unified_mask for both
    for col, (z, cfe_x, cfe_y) in enumerate(
            [
                (data["z_lr"], data["lr_cfe_x"], data["lr_cfe_y"]),
                (data["z_en"], data["en_cfe_x"], data["en_cfe_y"]),
            ],
            start=1,
    ):
        show = col == 1
        model_color = _LR_COLOR if col == 1 else _EN_COLOR
        orig_x = data["orig_x"]
        orig_y = data["orig_y"]

        # Find selected patient's positional index
        sel_pos = None
        if selected_patient is not None:
            idxs = np.where(patients == selected_patient)[0]
            # Use unified_mask to ensure selection only happens if point is visible
            if idxs.size and unified_mask[idxs[0]]:
                sel_pos = idxs[0]

        # Background mask for arrows (excludes selected patient)
        bg_mask = unified_mask.copy()
        if sel_pos is not None:
            bg_mask[sel_pos] = False

        bg_xs, bg_ys, bg_cdata = _arrow_segments(orig_x, orig_y, cfe_x, cfe_y, bg_mask, patients)

        if sel_pos is not None:
            hl_xs = [float(orig_x[sel_pos]), float(cfe_x[sel_pos]), None]
            hl_ys = [float(orig_y[sel_pos]), float(cfe_y[sel_pos]), None]
            hl_cdata = [int(selected_patient), int(selected_patient), None]
        else:
            hl_xs, hl_ys, hl_cdata = [], [], []

        # Trace 0/5 — Decision boundary
        fig.add_trace(
            go.Contour(
                x=data["x_vals"], y=data["y_vals"], z=z,
                contours=dict(coloring="none", showlines=True, start=0.5, end=0.5, size=1),
                line=dict(color=model_color, width=2, dash="dash"),
                showscale=False, opacity=0.6, name="Decision boundary",
                showlegend=False, hoverinfo="skip",
            ),
            row=1, col=col,
        )

        # Trace 1/6 — Arrow background
        fig.add_trace(
            go.Scatter(
                x=bg_xs, y=bg_ys, mode="lines",
                line=dict(color="#888888", width=0.8), opacity=0.35,
                customdata=bg_cdata, hovertemplate=hover_arrow,
                showlegend=False, visible=show_cfe,
            ),
            row=1, col=col,
        )

        # Trace 2/7 — Arrow highlight
        fig.add_trace(
            go.Scatter(
                x=hl_xs, y=hl_ys, mode="lines",
                line=dict(color="#ffffff", width=2), opacity=0.9,
                customdata=hl_cdata, hovertemplate=hover_arrow,
                showlegend=False, visible=show_cfe and sel_pos is not None,
            ),
            row=1, col=col,
        )

        # Trace 3/8 — CFE destination markers
        # patients.tolist() forces plain JSON ints; numpy arrays get binary-encoded
        # by Plotly 6.x, which breaks clickData scalar extraction in Dash.
        patients_list = patients.tolist()
        cfe_opacities = _point_opacities(unified_mask, patients, selected_patient, full=0.75, dim=0.1)
        cfe_sizes = _point_sizes(patients, selected_patient, normal=5, highlighted=11)
        fig.add_trace(
            go.Scatter(
                x=cfe_x, y=cfe_y, mode="markers",
                marker=dict(color=model_color, size=cfe_sizes, opacity=cfe_opacities, line=dict(width=0.5, color="grey")),
                name="CFE destination", showlegend=False,
                customdata=patients_list, hovertemplate=hover_cfe, visible=show_cfe,
            ),
            row=1, col=col,
        )

        # Trace 4/9 — Original scatter markers
        orig_opacities = _point_opacities(unified_mask, patients, selected_patient, full=0.85, dim=0.15)
        orig_sizes = _point_sizes(patients, selected_patient, normal=6, highlighted=11)
        orig_colors = ["#ffffff" if (selected_patient is not None and p == selected_patient) else "#d4d4d4" for p in
                       patients]
        fig.add_trace(
            go.Scatter(
                x=orig_x, y=orig_y, mode="markers",
                marker=dict(color=orig_colors, size=orig_sizes, opacity=orig_opacities,
                            line=dict(width=0.5, color="grey")),
                name="Original", showlegend=show, customdata=patients_list, hovertemplate=hover_orig,
            ),
            row=1, col=col,
        )

    fig.update_xaxes(title_text=feature_x, range=data["x_range"], row=1, col=1)
    fig.update_xaxes(title_text=feature_x, range=data["x_range"], row=1, col=2)
    fig.update_yaxes(title_text=feature_y, range=data["y_range"], row=1, col=1)
    fig.update_yaxes(range=data["y_range"], row=1, col=2)

    # --- Lower-left subplot: individual CFEs for selected patient ---
    fig.add_trace(
        go.Contour(
            x=data["x_vals"], y=data["y_vals"], z=data["z_lr"],
            contours=dict(coloring="none", showlines=True, start=0.5, end=0.5, size=1),
            line=dict(color=_LR_COLOR, width=2, dash="dash"),
            showscale=False, opacity=0.7, name="LR boundary",
            showlegend=True, hoverinfo="skip",
        ),
        row=2, col=1,
    )
    fig.add_trace(
        go.Contour(
            x=data["x_vals"], y=data["y_vals"], z=data["z_en"],
            contours=dict(coloring="none", showlines=True, start=0.5, end=0.5, size=1),
            line=dict(color=_EN_COLOR, width=2, dash="dash"),
            showscale=False, opacity=0.7, name="EN boundary",
            showlegend=True, hoverinfo="skip",
        ),
        row=2, col=1,
    )

    if selected_patient is not None and "indiv_cfe" in data:
        indiv_df = data["indiv_cfe"]
        pat_df = indiv_df[indiv_df["patient_idx"] == selected_patient]
        lr_df = pat_df[pat_df["model"] == "logistic_regression"].dropna(subset=["cfe_x", "cfe_y"])
        en_df = pat_df[pat_df["model"] == "elastic_net"].dropna(subset=["cfe_x", "cfe_y"])

        if not pat_df.empty:
            orig_x_pt = float(pat_df["orig_x"].iloc[0])
            orig_y_pt = float(pat_df["orig_y"].iloc[0])

            if not lr_df.empty:
                lr_xs, lr_ys = [], []
                for _, r in lr_df.iterrows():
                    lr_xs += [orig_x_pt, float(r["cfe_x"]), None]
                    lr_ys += [orig_y_pt, float(r["cfe_y"]), None]
                fig.add_trace(
                    go.Scatter(x=lr_xs, y=lr_ys, mode="lines",
                               line=dict(color=_LR_COLOR, width=1), opacity=0.4,
                               showlegend=False, hoverinfo="skip"),
                    row=2, col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=lr_df["cfe_x"].tolist(), y=lr_df["cfe_y"].tolist(), mode="markers",
                        marker=dict(color=_LR_COLOR, size=8, line=dict(width=0.5, color="grey")),
                        name="LR CFE", showlegend=True,
                        hovertemplate="LR CFE: (%{x:.3f}, %{y:.3f})<extra></extra>",
                    ),
                    row=2, col=1,
                )

            if not en_df.empty:
                en_xs, en_ys = [], []
                for _, r in en_df.iterrows():
                    en_xs += [orig_x_pt, float(r["cfe_x"]), None]
                    en_ys += [orig_y_pt, float(r["cfe_y"]), None]
                fig.add_trace(
                    go.Scatter(x=en_xs, y=en_ys, mode="lines",
                               line=dict(color=_EN_COLOR, width=1), opacity=0.4,
                               showlegend=False, hoverinfo="skip"),
                    row=2, col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=en_df["cfe_x"].tolist(), y=en_df["cfe_y"].tolist(), mode="markers",
                        marker=dict(color=_EN_COLOR, size=8, line=dict(width=0.5, color="grey")),
                        name="EN CFE", showlegend=True,
                        hovertemplate="EN CFE: (%{x:.3f}, %{y:.3f})<extra></extra>",
                    ),
                    row=2, col=1,
                )

            fig.add_trace(
                go.Scatter(
                    x=[orig_x_pt], y=[orig_y_pt], mode="markers",
                    marker=dict(color="#ffffff", size=11, line=dict(width=0.5, color="grey")),
                    showlegend=False, hoverinfo="skip",
                ),
                row=2, col=1,
            )
    else:
        fig.add_annotation(
            text="Select a patient above to see their individual counterfactuals.",
            xref="x3 domain", yref="y3 domain",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(color=TEXT_COLOR, size=12),
        )

    fig.update_xaxes(title_text=feature_x, range=data["x_range"], row=2, col=1)
    fig.update_yaxes(title_text=feature_y, range=data["y_range"], row=2, col=1)
    fig.update_xaxes(visible=False, row=2, col=2)
    fig.update_yaxes(visible=False, row=2, col=2)

    fig.update_layout(
        template=TEMPLATE,
        height=int(CHART_HEIGHT * 1.75),
        title=f"Decision boundaries: {feature_x} vs {feature_y}",
        showlegend=True,
        legend=dict(x=1.02, y=1),
        margin=dict(b=100),
        xaxis2=dict(matches="x"),
        yaxis2=dict(matches="y"),
        xaxis3=dict(matches="x"),
        yaxis3=dict(matches="y"),
        clickmode="event",
    )

    fig.layout.annotations[0].font.color = _LR_COLOR
    fig.layout.annotations[1].font.color = _EN_COLOR

    fig.add_annotation(
        text="Boundaries computed with all other features held at population median.",
        xref="paper", yref="paper",
        x=0.5, y=-0.07,
        showarrow=False,
        font=dict(color=TEXT_COLOR, size=11),
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
    """Convenience wrapper: precompute data then assemble the figure."""
    data = precompute_boundary_data(
        all_deltas, X, lr_pipeline, en_pipeline, feature_x, feature_y, grid_n
    )
    return assemble_boundary_fig(data)