"""
figures.py - Plotly figure-building functions for the CFE-Vis dash app.

Each function takes data in and returns a Plotly figure. No Dash logic,
no callbacks — just data in, figure out.

Usage:
    from components.figures import (
        build_heatmap, build_scatter,
        precompute_boundary_data, assemble_boundary_fig,
        make_predict_fns, _select_boundary_features, _MODEL_DISPLAY,
    )
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from components.theme import TEMPLATE, TEXT_COLOR, GRID_COLOR, CHART_HEIGHT

_MODEL_COLORS = {
    "logistic_regression": "#2a9d8f",
    "elastic_net": "#e76f51",
    "ac_svm": "#6a5acd",
}

_MODEL_DISPLAY = {
    "logistic_regression": "LR",
    "elastic_net": "EN",
    "ac_svm": "AC-SVM",
}

_DEFAULT_COLOR_A = "#2a9d8f"
_DEFAULT_COLOR_B = "#e76f51"


def make_predict_fns(spec: dict, pipeline, base_feature_cols: list):
    """
    Build (predict_proba_fn, predict_class_fn) from a model spec dict.

    Both functions accept a DataFrame with columns matching base_feature_cols in
    raw scale, and return numpy arrays (float proba / int class).
    """
    full_cols = spec["full_feature_cols"]

    if not spec["needs_scale_base"]:
        fill = spec["feature_fill_raw"]

        def _full(X_base_df):
            X = X_base_df[base_feature_cols].copy()
            for feat, val in fill.items():
                X[feat] = val
            return X[full_cols]

        def predict_proba(X_base_df):
            return pipeline.predict_proba(_full(X_base_df))

        def predict_class(X_base_df):
            return pipeline.predict(_full(X_base_df)).astype(int)
    else:
        data_min = np.array(spec["scaler_data_min"])
        data_range = np.array(spec["scaler_data_range"])
        extra_fill = spec["extra_feature_fill"]

        def _full(X_base_df):
            raw = X_base_df[base_feature_cols].values.astype(float)
            scaled = (raw - data_min) / data_range
            X = pd.DataFrame(scaled, columns=base_feature_cols)
            for feat, val in extra_fill.items():
                X[feat] = val
            return X[full_cols]

        def predict_proba(X_base_df):
            return pipeline.predict_proba(_full(X_base_df))

        def predict_class(X_base_df):
            return pipeline.predict(_full(X_base_df)).astype(int)

    return predict_proba, predict_class


def build_heatmap(
    all_deltas: pd.DataFrame,
    condensed: bool = False,
    model_a_name: str = "logistic_regression",
    model_b_name: str = "elastic_net",
) -> go.Figure:
    """
    Divergence heatmap: every patient × every feature, or a single-row summary.

    condensed=True  → one row showing mean disagreement per feature
    condensed=False → full patient × feature matrix (original behaviour)

    Clicking a column (feature) drives the drill-down to the scatter in both modes.
    """

    norm_cols = [col for col in all_deltas.columns if col.endswith("_norm")]

    a_avg = (all_deltas[all_deltas["model"] == model_a_name]
             .groupby("patient_idx")[norm_cols].mean())
    b_avg = (all_deltas[all_deltas["model"] == model_b_name]
             .groupby("patient_idx")[norm_cols].mean())

    shared = a_avg.index.intersection(b_avg.index)
    a_avg = a_avg.loc[shared]
    b_avg = b_avg.loc[shared]

    divergence = (b_avg - a_avg).abs()

    col_order = divergence.sum().sort_values(ascending=False).index
    divergence = divergence[col_order]

    feature_labels = [col.replace("_norm", "") for col in col_order]

    disp_a = _MODEL_DISPLAY.get(model_a_name, model_a_name)
    disp_b = _MODEL_DISPLAY.get(model_b_name, model_b_name)

    obs_max = float(divergence.values.max()) if divergence.size > 0 else 1.0

    if condensed:
        z = divergence.mean(axis=0).values.reshape(1, -1)
        y_labels = ["Mean disagreement"]
        title = f"{disp_a} vs {disp_b}: feature disagreement summary (mean across patients)"
        yaxis_cfg = dict(title="", showticklabels=False)
    else:
        divergence = divergence.loc[divergence.sum(axis=1).sort_values(ascending=True).index]
        z = divergence.values
        y_labels = [f"Patient {i+1}" for i in divergence.index]
        title = f"{disp_a} vs {disp_b}: Feature Disagreement Breakdown"
        yaxis_cfg = dict(
            title=dict(text="Patients (Least to most model disagreement)", standoff=10),
            showticklabels=False,
        )

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=feature_labels,
            y=y_labels,
            colorscale="OrRd",
            zmin=0,
            zmax=obs_max,
            colorbar=dict(title="Absolute<br>disagreement<br>(MinMaxScale)"),
            hovertemplate=(
                "Feature: %{x}<br>"
                "Patient: %{y}<br>"
                "Disagreement: %{z:.3f}"
                "<extra></extra>"
            ),
        ),
        layout=go.Layout(
            template=TEMPLATE,
            title=title,
            xaxis=dict(title="", tickangle=-45),
            yaxis=yaxis_cfg,
        ),
    )

    fig.update_layout(clickmode="event")

    return fig


def build_scatter(
    all_deltas: pd.DataFrame,
    feature: str | list[str],
    height: int = CHART_HEIGHT,
    axis_range: list | None = None,
    cmax: float | None = None,
    model_a_name: str = "logistic_regression",
    model_b_name: str = "elastic_net",
) -> go.Figure:
    """
    Scatter plot: model A delta vs model B delta across all patients.

    Parameters
    ----------
    feature    : clean feature name (e.g. "acpa") or list of names for
                 aggregated view — each patient's value is the mean delta
                 across those features.
    axis_range : fixed [min, max] for both axes — pass to keep scale consistent
                 across features. If None, auto-scales to the current feature.
    cmax       : fixed colorbar maximum — pass to keep the colour scale
                 consistent across features. If None, uses per-feature max.
    """

    disp_a = _MODEL_DISPLAY.get(model_a_name, model_a_name)
    disp_b = _MODEL_DISPLAY.get(model_b_name, model_b_name)

    if isinstance(feature, list):
        norm_cols = [f"{f}_norm" for f in feature]
        a_deltas = (all_deltas[all_deltas["model"] == model_a_name]
                    .groupby("patient_idx")[norm_cols].mean().mean(axis=1))
        b_deltas = (all_deltas[all_deltas["model"] == model_b_name]
                    .groupby("patient_idx")[norm_cols].mean().mean(axis=1))
        title = f"How do the models disagree across the top {len(feature)} features?"
        axis_label = f"Mean normalised delta (top {len(feature)} features)"
        x_label = f"{disp_a}: {axis_label}"
        y_label = f"{disp_b}: {axis_label}"
    else:
        norm_col = f"{feature}_norm"
        a_deltas = (all_deltas[all_deltas["model"] == model_a_name]
                    .groupby("patient_idx")[norm_col].mean())
        b_deltas = (all_deltas[all_deltas["model"] == model_b_name]
                    .groupby("patient_idx")[norm_col].mean())
        title = f"How do the models disagree about {feature}?"
        x_label = f"{disp_a}: Normalised delta"
        y_label = f"{disp_b}: Normalised delta"

    shared = a_deltas.index.intersection(b_deltas.index)
    a_vals = a_deltas.loc[shared].values
    b_vals = b_deltas.loc[shared].values
    patient_indices = shared.values

    distance_from_diagonal = np.abs(b_vals - a_vals) / np.sqrt(2)

    effective_cmax = cmax if cmax is not None else distance_from_diagonal.max()

    fig = go.Figure(
        data=go.Scatter(
            x=a_vals,
            y=b_vals,
            mode="markers",
            marker=dict(
                size=8,
                color=distance_from_diagonal,
                colorscale="OrRd",
                cmin=0,
                cmax=effective_cmax,
                showscale=False,
                line=dict(width=0.5, color="grey"),
                opacity=0.8,
            ),

            customdata=(patient_indices + 1).tolist(),

            hovertemplate=(
                f"Patient: %{{customdata}}<br>"
                f"{disp_a} delta: %{{x:.3f}}<br>"
                f"{disp_b} delta: %{{y:.3f}}"
                "<extra></extra>"
            ),
        ),
        layout=go.Layout(
            template=TEMPLATE,
            title=title,
            xaxis=dict(title=x_label, range=axis_range),
            yaxis=dict(title=y_label, range=axis_range),
        ),
    )

    diag_bound = axis_range[1] if axis_range is not None else np.abs(np.concatenate([a_vals, b_vals])).max()

    fig.add_shape(
        type="line",
        x0=-diag_bound, y0=-diag_bound,
        x1=diag_bound, y1=diag_bound,
        line=dict(color="grey", dash="dash", width=1),
    )

    fig.add_hline(y=0, line=dict(color=GRID_COLOR, width=0.5))
    fig.add_vline(x=0, line=dict(color=GRID_COLOR, width=0.5))

    fig.update_layout(clickmode="event")

    return fig



def _select_boundary_features(
    X: pd.DataFrame,
    predict_proba_a,
    predict_proba_b,
    coarse_n: int = 15,
) -> tuple[str, str]:
    """
    From all feature pairs, pick the pair where both models have the widest
    probability range in the 2D projection (other features fixed at median).

    Scores each pair by the minimum probability range across both models — this
    ensures neither model's boundary is invisible in the plot.
    """
    features = list(X.columns)
    medians = X.median()
    n_pts = coarse_n * coarse_n

    best_score = -1.0
    best_pair = (features[0], features[1])

    for i, fx in enumerate(features):
        x_vals = np.linspace(X[fx].min(), X[fx].max(), coarse_n)
        for j, fy in enumerate(features):
            if j <= i:
                continue
            y_vals = np.linspace(X[fy].min(), X[fy].max(), coarse_n)
            xx, yy = np.meshgrid(x_vals, y_vals)
            grid_df = pd.DataFrame(
                np.tile(medians.values, (n_pts, 1)), columns=X.columns
            )
            grid_df[fx] = xx.ravel()
            grid_df[fy] = yy.ravel()

            za = predict_proba_a(grid_df)[:, 1]
            zb = predict_proba_b(grid_df)[:, 1]
            score = min(float(za.max() - za.min()), float(zb.max() - zb.min()))
            if score > best_score:
                best_score = score
                best_pair = (fx, fy)

    return best_pair


def precompute_boundary_data(
    all_deltas: pd.DataFrame,
    X: pd.DataFrame,
    predict_proba_a,
    predict_class_a,
    predict_proba_b,
    predict_class_b,
    model_a_name: str = "logistic_regression",
    model_b_name: str = "elastic_net",
    feature_x: str | None = None,
    feature_y: str | None = None,
    grid_n: int = 60,
    indiv_cfe_batch: pd.DataFrame | None = None,
) -> dict:
    """
    Run all expensive computation (grid prediction, CFE aggregation) once.

    Returns a plain dict consumed by assemble_boundary_fig. Separating this
    from the figure builder means model calls happen only at page load / pair
    switch, not on every user interaction.

    predict_proba_a / predict_proba_b accept a DataFrame with the same columns
    as X (base features in raw scale) and return (n, 2) probability arrays.
    predict_class_a / predict_class_b return integer class arrays.

    indiv_cfe_batch must have columns: patient_idx, model, orig_{feature},
    cfe_{feature} for every feature in X.
    """

    # Auto-pick feature pair: find the pair where both models' boundaries are
    # visible in 2D projection. Pure disagreement-based selection fails when
    # one model (e.g. SVM) has its boundary along different features than the
    # other model.
    if feature_x is None or feature_y is None:
        feature_x, feature_y = _select_boundary_features(
            X, predict_proba_a, predict_proba_b
        )

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

    z_a = predict_proba_a(grid_df)[:, 1].reshape(grid_n, grid_n)
    z_b = predict_proba_b(grid_df)[:, 1].reshape(grid_n, grid_n)

    # patient_idx values are positional (from enumerate) — use iloc not loc
    patients = np.sort(all_deltas["patient_idx"].unique())
    X_pat = X.iloc[patients]
    orig_x = X_pat[feature_x].values
    orig_y = X_pat[feature_y].values

    a_cfe_x = np.full(len(patients), np.nan)
    a_cfe_y = np.full(len(patients), np.nan)
    b_cfe_x = np.full(len(patients), np.nan)
    b_cfe_y = np.full(len(patients), np.nan)

    if indiv_cfe_batch is not None:
        mean_dest = (
            indiv_cfe_batch
            .groupby(["patient_idx", "model"])[[f"cfe_{feature_x}", f"cfe_{feature_y}"]]
            .mean()
        )
        for pat_pos, pat_idx in enumerate(patients):
            pat_idx = int(pat_idx)
            try:
                a_cfe_x[pat_pos] = mean_dest.loc[(pat_idx, model_a_name), f"cfe_{feature_x}"]
                a_cfe_y[pat_pos] = mean_dest.loc[(pat_idx, model_a_name), f"cfe_{feature_y}"]
            except KeyError:
                pass
            try:
                b_cfe_x[pat_pos] = mean_dest.loc[(pat_idx, model_b_name), f"cfe_{feature_x}"]
                b_cfe_y[pat_pos] = mean_dest.loc[(pat_idx, model_b_name), f"cfe_{feature_y}"]
            except KeyError:
                pass

    all_x = np.concatenate([orig_x, a_cfe_x, b_cfe_x, [x_vals[0], x_vals[-1]]])
    all_y = np.concatenate([orig_y, a_cfe_y, b_cfe_y, [y_vals[0], y_vals[-1]]])
    finite_x = all_x[np.isfinite(all_x)]
    finite_y = all_y[np.isfinite(all_y)]
    x_range = [float(finite_x.min()) * 0.98, float(finite_x.max()) * 1.02]
    y_range = [float(finite_y.min()) * 0.98, float(finite_y.max()) * 1.02]

    a_preds = predict_class_a(X_pat)
    b_preds = predict_class_b(X_pat)

    plot_batch = None
    if indiv_cfe_batch is not None:
        plot_batch = (
            indiv_cfe_batch[[
                "patient_idx", "model",
                f"orig_{feature_x}", f"orig_{feature_y}",
                f"cfe_{feature_x}", f"cfe_{feature_y}",
            ]]
            .rename(columns={
                f"orig_{feature_x}": "orig_x",
                f"orig_{feature_y}": "orig_y",
                f"cfe_{feature_x}": "cfe_x",
                f"cfe_{feature_y}": "cfe_y",
            })
            .copy()
        )

    return dict(
        feature_x=feature_x,
        feature_y=feature_y,
        x_vals=x_vals,
        y_vals=y_vals,
        z_a=z_a,
        z_b=z_b,
        patients=patients,
        orig_x=orig_x,
        orig_y=orig_y,
        a_cfe_x=a_cfe_x,
        a_cfe_y=a_cfe_y,
        b_cfe_x=b_cfe_x,
        b_cfe_y=b_cfe_y,
        a_preds=a_preds,
        b_preds=b_preds,
        x_range=x_range,
        y_range=y_range,
        indiv_cfe_batch=plot_batch,
        model_a_name=model_a_name,
        model_b_name=model_b_name,
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
    """Build NaN-separated line arrays for patients where mask is True."""
    xs: list = []
    ys: list = []
    cdata: list = []
    for i in range(len(ox)):
        if mask[i] and np.isfinite(cx[i]) and np.isfinite(cy[i]):
            p = int(patients[i])
            xs += [float(ox[i]), float(cx[i]), None]
            ys += [float(oy[i]), float(cy[i]), None]
            cdata += [[p, p + 1], [p, p + 1], None]
    return xs, ys, cdata


def _point_opacities(
    patients: np.ndarray,
    selected_patient: int | None,
    full: float,
    dim: float,
) -> float | list[float]:
    """Per-point opacity for highlight — direction masking is handled by None coordinates."""
    if selected_patient is None:
        return full
    return [full if p == selected_patient else dim for p in patients]


def _masked_coords(arr: np.ndarray, mask: np.ndarray, finite_only: bool = False) -> list:
    """Replace masked or (optionally) non-finite positions with None to suppress click events."""
    result = []
    for i in range(len(mask)):
        if mask[i] and (not finite_only or np.isfinite(arr[i])):
            result.append(float(arr[i]))
        else:
            result.append(None)
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
        pca_results: pd.DataFrame | None = None,
        indiv_model: str = "Both",
) -> go.Figure:
    """
    Build the decision boundary figure from pre-computed data.

    indiv_model: "Both", "A" (model_a), or "B" (model_b)
    """
    feature_x = data["feature_x"]
    feature_y = data["feature_y"]
    patients = data["patients"]
    model_a_name = data.get("model_a_name", "logistic_regression")
    model_b_name = data.get("model_b_name", "elastic_net")

    disp_a = _MODEL_DISPLAY.get(model_a_name, model_a_name)
    disp_b = _MODEL_DISPLAY.get(model_b_name, model_b_name)
    color_a = _MODEL_COLORS.get(model_a_name, _DEFAULT_COLOR_A)
    color_b = _MODEL_COLORS.get(model_b_name, _DEFAULT_COLOR_B)

    # Unified mask: only patients where BOTH models agree on direction
    unified_mask = _direction_mask(data["a_preds"], direction) & \
                   _direction_mask(data["b_preds"], direction)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            disp_a, disp_b,
            "Individual counterfactuals (both models)",
            "Patient overview: directional vs outcome disagreement",
        ],
        horizontal_spacing=0.10,
        vertical_spacing=0.10,
        row_heights=[0.5, 0.5],
    )

    hover_cfe = "Patient: %{customdata[1]}<br>Value: %{x:.3f}, %{y:.3f}<extra>CFE</extra>"
    hover_arrow = "Patient: %{customdata[1]}<extra>Arrow</extra>"
    hover_orig = "Patient: %{customdata[1]}<br>Value: %{x:.3f}, %{y:.3f}<extra>Original</extra>"

    for col, (z, cfe_x, cfe_y, model_color) in enumerate(
            [
                (data["z_a"], data["a_cfe_x"], data["a_cfe_y"], color_a),
                (data["z_b"], data["b_cfe_x"], data["b_cfe_y"], color_b),
            ],
            start=1,
    ):
        show = col == 1
        orig_x = data["orig_x"]
        orig_y = data["orig_y"]

        sel_pos = None
        if selected_patient is not None:
            idxs = np.where(patients == selected_patient)[0]
            if idxs.size and unified_mask[idxs[0]]:
                sel_pos = idxs[0]

        bg_mask = unified_mask.copy()
        if sel_pos is not None:
            bg_mask[sel_pos] = False

        bg_xs, bg_ys, bg_cdata = _arrow_segments(orig_x, orig_y, cfe_x, cfe_y, bg_mask, patients)

        if sel_pos is not None:
            hl_xs = [float(orig_x[sel_pos]), float(cfe_x[sel_pos]), None]
            hl_ys = [float(orig_y[sel_pos]), float(cfe_y[sel_pos]), None]
            sp = int(selected_patient)
            hl_cdata = [[sp, sp + 1], [sp, sp + 1], None]
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
        patients_cdata = [[int(p), int(p) + 1] for p in patients]
        cfe_opacities = _point_opacities(patients, selected_patient, full=0.75, dim=0.1)
        cfe_sizes = _point_sizes(patients, selected_patient, normal=5, highlighted=11)
        fig.add_trace(
            go.Scatter(
                x=_masked_coords(cfe_x, unified_mask, finite_only=True),
                y=_masked_coords(cfe_y, unified_mask, finite_only=True),
                mode="markers",
                marker=dict(color=model_color, size=cfe_sizes, opacity=cfe_opacities, line=dict(width=0.5, color="grey")),
                name="CFE destination", showlegend=False,
                customdata=patients_cdata, hovertemplate=hover_cfe, visible=show_cfe,
            ),
            row=1, col=col,
        )

        # Trace 4/9 — Original scatter markers
        orig_opacities = _point_opacities(patients, selected_patient, full=0.85, dim=0.15)
        orig_sizes = _point_sizes(patients, selected_patient, normal=6, highlighted=11)
        orig_colors = ["#ffffff" if (selected_patient is not None and p == selected_patient) else "#d4d4d4" for p in patients]
        fig.add_trace(
            go.Scatter(
                x=_masked_coords(orig_x, unified_mask),
                y=_masked_coords(orig_y, unified_mask),
                mode="markers",
                marker=dict(color=orig_colors, size=orig_sizes, opacity=orig_opacities,
                            line=dict(width=0.5, color="grey")),
                name="Original", showlegend=show, customdata=patients_cdata, hovertemplate=hover_orig,
            ),
            row=1, col=col,
        )

    fig.update_xaxes(title_text=feature_x, range=data["x_range"], row=1, col=1)
    fig.update_xaxes(title_text=feature_x, range=data["x_range"], row=1, col=2)
    fig.update_yaxes(title_text=feature_y, range=data["y_range"], row=1, col=1)
    fig.update_yaxes(range=data["y_range"], row=1, col=2)

    # Traces 10, 11 — legend dummy entries
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(symbol="circle", size=6, color=color_a, line=dict(width=0.5, color="grey")),
            name=f"{disp_a} counterfactual",
            showlegend=True,
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(symbol="circle", size=6, color=color_b, line=dict(width=0.5, color="grey")),
            name=f"{disp_b} counterfactual",
            showlegend=True,
        ),
        row=1, col=1,
    )

    # Traces 12, 13 — lower-left boundary contours
    fig.add_trace(
        go.Contour(
            x=data["x_vals"], y=data["y_vals"], z=data["z_a"],
            contours=dict(coloring="none", showlines=True, start=0.5, end=0.5, size=1),
            line=dict(color=color_a, width=2, dash="dash"),
            showscale=False, opacity=0.7, name=f"{disp_a} boundary",
            showlegend=True, hoverinfo="skip",
        ),
        row=2, col=1,
    )
    fig.add_trace(
        go.Contour(
            x=data["x_vals"], y=data["y_vals"], z=data["z_b"],
            contours=dict(coloring="none", showlines=True, start=0.5, end=0.5, size=1),
            line=dict(color=color_b, width=2, dash="dash"),
            showscale=False, opacity=0.7, name=f"{disp_b} boundary",
            showlegend=True, hoverinfo="skip",
        ),
        row=2, col=1,
    )

    # --- Lower-right subplot: PCA structure scatter ---
    # Traces 14 (bg) and 15 (highlight) always added for stable indices
    if pca_results is not None:
        n_features = sum(1 for c in pca_results.columns if c.startswith("pc1_v"))
        a_pca = pca_results[pca_results["model"] == model_a_name].set_index("patient_idx")
        b_pca = pca_results[pca_results["model"] == model_b_name].set_index("patient_idx")
        shared_pca = a_pca.index.intersection(b_pca.index)

        visible_patients = set(patients[unified_mask].tolist())
        shared_f = shared_pca[shared_pca.isin(visible_patients)]
        a_f = a_pca.loc[shared_f]
        b_f = b_pca.loc[shared_f]

        pc1_cols = [f"pc1_v{i}" for i in range(n_features)]

        a_full = a_pca.loc[shared_pca]
        b_full = b_pca.loc[shared_pca]
        confidence_diff_full = b_full["mean_confidence"].values - a_full["mean_confidence"].values
        x_max_pca = (
            max(abs(float(confidence_diff_full.min())), abs(float(confidence_diff_full.max())))
            if len(confidence_diff_full) > 0 else 0.5
        )

        a_pc1 = a_f[pc1_cols].values
        b_pc1 = b_f[pc1_cols].values

        dots = np.clip(np.abs((a_pc1 * b_pc1).sum(axis=1)), -1.0, 1.0)
        pc1_angle = np.degrees(np.arccos(dots))
        confidence_diff = b_f["mean_confidence"].values - a_f["mean_confidence"].values
        a_evr = a_f["pc1_ratio"].values
        b_evr = b_f["pc1_ratio"].values
        reliability = np.minimum(a_evr, b_evr)

        evr_min = float(reliability.min()) if len(reliability) > 0 else 0.0
        evr_max = float(reliability.max()) if len(reliability) > 0 else 1.0
        evr_range = evr_max - evr_min if evr_max > evr_min else 1.0

        hover_pca = (
            f"Patient: %{{customdata[1]}}<br>"
            f"Confidence diff ({disp_b}−{disp_a}): %{{x:.3f}}<br>"
            "PC1 angle: %{y:.1f}°<br>"
            f"EVR PC1 {disp_a}: %{{customdata[2]:.3f}}<br>"
            f"EVR PC1 {disp_b}: %{{customdata[3]:.3f}}"
            "<extra></extra>"
        )

        bg_x, bg_y, bg_opacities, bg_cdata = [], [], [], []
        hl_x, hl_y, hl_cdata = [], [], []
        for i, p in enumerate(shared_f.tolist()):
            t = float(np.clip((reliability[i] - evr_min) / evr_range, 0.0, 1.0))
            opacity = 0.25 + 0.75 * t
            if selected_patient is not None and p == selected_patient:
                hl_x.append(float(confidence_diff[i]))
                hl_y.append(float(pc1_angle[i]))
                hl_cdata.append([p, p + 1, float(a_evr[i]), float(b_evr[i])])
            else:
                bg_x.append(float(confidence_diff[i]))
                bg_y.append(float(pc1_angle[i]))
                bg_opacities.append(opacity)
                bg_cdata.append([p, p + 1, float(a_evr[i]), float(b_evr[i])])

        fig.add_trace(
            go.Scatter(
                x=bg_x, y=bg_y,
                mode="markers",
                marker=dict(
                    size=8,
                    color="#d4d4d4",
                    opacity=bg_opacities or 0.5,
                    line=dict(width=0.5, color="grey"),
                ),
                customdata=bg_cdata,
                hovertemplate=hover_pca,
                showlegend=False,
            ),
            row=2, col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=hl_x, y=hl_y,
                mode="markers",
                marker=dict(
                    size=10,
                    color="rgba(0,0,0,0)",
                    opacity=1.0,
                    line=dict(width=2, color=TEXT_COLOR),
                ),
                customdata=hl_cdata,
                hovertemplate=hover_pca,
                showlegend=False,
            ),
            row=2, col=2,
        )

        fig.update_xaxes(
            title_text=f"Difference in mean CFE predicted probability ({disp_b} − {disp_a})",
            range=[-x_max_pca * 1.1, x_max_pca * 1.1],
            zeroline=False,
            row=2, col=2,
        )
        fig.update_yaxes(title_text="Angle between CFE PC1 vectors (°)", range=[0, 92], row=2, col=2)
        fig.add_annotation(
            text=f"← {disp_a}'s CFEs more confident   |   {disp_b}'s CFEs more confident →",
            xref="x4 domain", yref="paper",
            x=0.5, y=-0.065,
            showarrow=False,
            font=dict(color=GRID_COLOR, size=10),
            xanchor="center",
            yanchor="top",
        )

        # Reference lines rendered below data points
        fig.add_shape(
            type="line",
            x0=0, y0=0, x1=0, y1=92,
            xref="x4", yref="y4",
            line=dict(color="rgba(180,180,180,0.35)", width=0.5, dash="dash"),
            layer="below",
        )
        fig.add_shape(
            type="line",
            x0=-x_max_pca * 1.1, y0=45, x1=x_max_pca * 1.1, y1=45,
            xref="x4", yref="y4",
            line=dict(color="rgba(180,180,180,0.35)", width=0.5, dash="dash"),
            layer="below",
        )
        # Arrow + labels in inter-subplot gap (paper coords), parallel to y-axis.
        # axref/ayref don't support "paper" — draw shaft as a shape,
        # arrowheads as pixel-offset annotations.
        # h_spacing=0.10 → gap x=[0.45, 0.55]; row2 paper y≈[0.00, 0.45]
        fig.add_shape(
            type="line",
            x0=0.505, y0=0.10, x1=0.505, y1=0.38,
            xref="paper", yref="paper",
            line=dict(color=GRID_COLOR, width=1),
        )
        fig.add_annotation(
            x=0.505, y=0.38,
            ax=0, ay=14,
            xref="paper", yref="paper",
            axref="pixel", ayref="pixel",
            arrowhead=2, arrowside="end", arrowsize=1.5, arrowcolor=GRID_COLOR,
            showarrow=True, text="",
        )
        fig.add_annotation(
            x=0.505, y=0.10,
            ax=0, ay=-14,
            xref="paper", yref="paper",
            axref="pixel", ayref="pixel",
            arrowhead=2, arrowside="end", arrowsize=1.5, arrowcolor=GRID_COLOR,
            showarrow=True, text="",
        )
        fig.add_annotation(
            x=0.493, y=0.29,
            xref="paper", yref="paper",
            text="Divergent",
            textangle=-90,
            showarrow=False,
            font=dict(color=TEXT_COLOR, size=9),
            xanchor="center",
        )
        fig.add_annotation(
            x=0.493, y=0.16,
            xref="paper", yref="paper",
            text="Aligned",
            textangle=-90,
            showarrow=False,
            font=dict(color=TEXT_COLOR, size=9),
            xanchor="center",
        )
    else:
        for _ in range(2):
            fig.add_trace(go.Scatter(x=[], y=[], mode="markers", showlegend=False), row=2, col=2)
        fig.update_xaxes(visible=False, row=2, col=2)
        fig.update_yaxes(visible=False, row=2, col=2)

    indiv_batch = data.get("indiv_cfe_batch")
    if selected_patient is not None and indiv_batch is not None:
        pat_df = indiv_batch[indiv_batch["patient_idx"] == selected_patient]
        show_a = indiv_model in ("Both", "A")
        show_b = indiv_model in ("Both", "B")

        a_df = pat_df[pat_df["model"] == model_a_name].dropna(subset=["cfe_x", "cfe_y"]) if show_a else pd.DataFrame()
        b_df = pat_df[pat_df["model"] == model_b_name].dropna(subset=["cfe_x", "cfe_y"]) if show_b else pd.DataFrame()

        if not pat_df.empty:
            orig_x_pt = float(pat_df["orig_x"].iloc[0])
            orig_y_pt = float(pat_df["orig_y"].iloc[0])

            if not a_df.empty:
                a_xs, a_ys = [], []
                for _, r in a_df.iterrows():
                    a_xs += [orig_x_pt, float(r["cfe_x"]), None]
                    a_ys += [orig_y_pt, float(r["cfe_y"]), None]
                fig.add_trace(
                    go.Scatter(x=a_xs, y=a_ys, mode="lines",
                               line=dict(color=color_a, width=1), opacity=0.4,
                               showlegend=False, hoverinfo="skip"),
                    row=2, col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=a_df["cfe_x"].tolist(), y=a_df["cfe_y"].tolist(), mode="markers",
                        marker=dict(color=color_a, size=6, opacity=0.7, line=dict(width=0.5, color="grey")),
                        name=f"{disp_a} CFE", showlegend=False,
                        hovertemplate=f"{disp_a} CFE: (%{{x:.3f}}, %{{y:.3f}})<extra></extra>",
                    ),
                    row=2, col=1,
                )

            if not b_df.empty:
                b_xs, b_ys = [], []
                for _, r in b_df.iterrows():
                    b_xs += [orig_x_pt, float(r["cfe_x"]), None]
                    b_ys += [orig_y_pt, float(r["cfe_y"]), None]
                fig.add_trace(
                    go.Scatter(x=b_xs, y=b_ys, mode="lines",
                               line=dict(color=color_b, width=1), opacity=0.4,
                               showlegend=False, hoverinfo="skip"),
                    row=2, col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=b_df["cfe_x"].tolist(), y=b_df["cfe_y"].tolist(), mode="markers",
                        marker=dict(color=color_b, size=6, opacity=0.7, line=dict(width=0.5, color="grey")),
                        name=f"{disp_b} CFE", showlegend=False,
                        hovertemplate=f"{disp_b} CFE: (%{{x:.3f}}, %{{y:.3f}})<extra></extra>",
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

    fig.update_layout(
        template=TEMPLATE,
        autosize=True,
        title=f"Mean CFE paths on decision boundary slice: {feature_x} vs {feature_y}",
        showlegend=True,
        legend=dict(x=1.02, y=1),
        margin=dict(b=130),
        xaxis2=dict(matches="x"),
        yaxis2=dict(matches="y"),
        xaxis3=dict(matches="x"),
        yaxis3=dict(matches="y"),
        clickmode="event",
    )

    fig.layout.annotations[0].font.color = color_a
    fig.layout.annotations[1].font.color = color_b
    fig.layout.annotations[3].font.size = 13
    fig.layout.annotations[3].font.weight = 700

    # Italic subtitle below patient overview panel title
    title_ann = fig.layout.annotations[3]
    fig.add_annotation(
        x=title_ann.x + 0.118,
        y=title_ann.y - 0.032,
        xref="paper", yref="paper",
        text="<i>Click a point to inspect that patient</i>",
        showarrow=False,
        font=dict(color=GRID_COLOR, size=10),
        xanchor="center",
        yanchor="top",
    )

    # Subtle border-left priority cue for the patient overview panel
    # h_spacing=0.10 → col2 domain x=[0.55, 1.0]; row2 y=[0.0, 0.45]
    fig.add_shape(
        type="line",
        x0=0.55, y0=0.0, x1=0.55, y1=0.45,
        xref="paper", yref="paper",
        line=dict(color="rgba(128,128,128,0.40)", width=1),
        layer="below",
    )

    fig.add_annotation(
        text="Boundaries computed with all other features held at population median.",
        xref="paper", yref="paper",
        x=0.5, y=-0.10,
        showarrow=False,
        font=dict(color=TEXT_COLOR, size=11),
    )

    return fig


