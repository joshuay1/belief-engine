"""
Generate combined figures for Belief Engine parameter recommendations.

The parameter recommendation runs already create per-run figures. This script
combines the group-held-out and topic-held-out quality runs into a compact
paper-facing visualization of:

1. where each recommended preset sits in update_s / anchor_s space, and
2. how stable the selected parameters are across folds and split types, and
3. why those presets were chosen by plotting per-class RMSE surfaces.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/review_artifact_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/review_artifact_cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

from validation.data.load_debate import load_debate_dataset
from validation.data.preprocess import build_participant_timelines
from validation.evaluation.common import build_records, filter_records, make_folds
from validation.scripts.run_parameter_recommendations import (
    DEFAULT_GRID,
    _candidate_rows,
    _load_excluded_ids,
    _load_json_if_exists,
    _prepare_update_args,
    _records_for_key,
    _score_rows,
)


DEFAULT_GROUP_DIR = "validation/results/parameter_recommendations_group_20260426_002858"
DEFAULT_TOPIC_DIR = "validation/results/parameter_recommendations_topic_20260426_002944"
DEFAULT_MOVERS_GROUP_DIR = "validation/results/parameter_recommendations_group_20260426_002742"
DEFAULT_MOVERS_TOPIC_DIR = "validation/results/parameter_recommendations_topic_20260426_002813"
DEFAULT_OUTPUT_DIR = "validation/results/parameter_recommendations_showcase"

MAIN_PRESET_ORDER = [
    "Open / conflicted",
    "Leaning",
    "Committed",
    "General default",
    "Broad social/political",
    "Technical/factual",
]

RESPONSE_TYPE_ORDER = [
    "Stable",
    "Softened",
    "Reinforced",
    "Boundary switchers",
    "Strong switchers",
]

EVIDENCE_ALIGNMENT_ORDER = [
    "All participants",
    "Evidence-aligned movers",
    "Evidence-opposed movers",
    "Weak-signal movers",
    "Stable despite pressure",
    "Stable low-pressure",
]

EVIDENCE_ALIGNMENT_FOCUS_ORDER = [
    "All participants",
    "Evidence-aligned movers",
    "Evidence-opposed movers",
    "Stable",
]

PRESET_ORDER = MAIN_PRESET_ORDER

COLORS = {
    "General default": "#4c566a",
    "All participants": "#4c566a",
    "Evidence-aligned movers": "#19747e",
    "Evidence-opposed movers": "#b34a4a",
    "Weak-signal movers": "#7b5ea7",
    "Stable despite pressure": "#c96f2d",
    "Stable low-pressure": "#6b7280",
    "Stable": "#4c566a",
    "Softened": "#19747e",
    "Reinforced": "#4f8f45",
    "Boundary switchers": "#c96f2d",
    "Strong switchers": "#b34a4a",
    "Open / conflicted": "#2f6f9f",
    "Leaning": "#4f8f45",
    "Committed": "#b34a4a",
    "Broad social/political": "#c96f2d",
    "Technical/factual": "#7b5ea7",
}

RMSE_CMAP = LinearSegmentedColormap.from_list(
    "rmse_flat",
    ["#f7f1d5", "#d9c77e", "#97b36b", "#4f9a7c", "#19747e", "#22435f", "#171c2e"],
)

IMPORTANCE_COLORS = {
    "update_share": "#19747e",
    "anchor_share": "#d9c77e",
    "interaction_share": "#22435f",
}

PRESET_SPECS = [
    ("global", "all", "General default"),
    ("evidence_alignment", "evidence_all", "All participants"),
    ("evidence_alignment", "evidence_aligned_mover", "Evidence-aligned movers"),
    ("evidence_alignment", "evidence_opposed_mover", "Evidence-opposed movers"),
    ("evidence_alignment", "weak_signal_mover", "Weak-signal movers"),
    ("evidence_alignment", "stable_despite_pressure", "Stable despite pressure"),
    ("evidence_alignment", "stable_low_pressure", "Stable low-pressure"),
    ("response_type", "stable", "Stable"),
    ("response_type", "softened", "Softened"),
    ("response_type", "reinforced", "Reinforced"),
    ("response_type", "boundary_switcher", "Boundary switchers"),
    ("response_type", "strong_switcher", "Strong switchers"),
    ("participant_type", "open", "Open / conflicted"),
    ("participant_type", "leaning", "Leaning"),
    ("participant_type", "committed", "Committed"),
    ("topic_category", "breadth", "Broad social/political"),
    ("topic_category", "depth", "Technical/factual"),
]


def _load(group_dir: str, topic_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    folds = []
    for split, directory in [("group-held-out", group_dir), ("topic-held-out", topic_dir)]:
        recs = pd.read_csv(os.path.join(directory, "parameter_recommendations.csv"))
        recs["split"] = split
        frames.append(recs)
        fold = pd.read_csv(os.path.join(directory, "fold_selected_params.csv"))
        fold["split"] = split
        folds.append(fold)
    return pd.concat(frames, ignore_index=True), pd.concat(folds, ignore_index=True)


def _plot_recommended_space(recs: pd.DataFrame, folds: pd.DataFrame, output_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 6.2))

    for label in PRESET_ORDER:
        fold_group = folds[folds["preset_label"] == label]
        if fold_group.empty:
            continue
        color = COLORS[label]
        for split, marker in [("group-held-out", "o"), ("topic-held-out", "^")]:
            split_df = fold_group[fold_group["split"] == split]
            ax.scatter(
                split_df["update_s"],
                split_df["anchor_s"],
                s=42,
                color=color,
                marker=marker,
                alpha=0.28,
                edgecolor="none",
            )

        rec_group = recs[recs["preset_label"] == label]
        if not rec_group.empty:
            x = float(rec_group["recommended_update_s"].median())
            y = float(rec_group["recommended_anchor_s"].median())
            ax.scatter(
                x,
                y,
                s=190,
                color=color,
                marker="*",
                edgecolor="white",
                linewidth=1.2,
                zorder=5,
            )
            ax.annotate(
                label,
                (x, y),
                xytext=(7, 5),
                textcoords="offset points",
                fontsize=9,
                weight="bold" if label in {"Open / conflicted", "Leaning", "Committed"} else "normal",
            )

    ax.scatter([], [], marker="o", color="black", alpha=0.35, label="group fold selections")
    ax.scatter([], [], marker="^", color="black", alpha=0.35, label="topic fold selections")
    ax.scatter([], [], marker="*", color="black", label="recommended preset")
    ax.legend(frameon=False, loc="lower right", fontsize=9)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.015, 0.16)
    ax.set_ylim(0.32, 1.05)
    ax.set_xlabel("update_s: evidence responsiveness")
    ax.set_ylabel("anchor_s: prior anchoring")
    ax.set_title("Held-Out Parameter Presets Are Stable Across Splits")
    ax.grid(True, which="both", alpha=0.18)
    fig.tight_layout()

    for ext in ["pdf", "png", "svg"]:
        fig.savefig(os.path.join(output_dir, f"figure_parameter_preset_space.{ext}"), dpi=220)
    plt.close(fig)


def _plot_fold_stability(folds: pd.DataFrame, output_dir: str) -> None:
    selected = folds[folds["preset_label"].isin(PRESET_ORDER)].copy()
    selected["pair"] = (
        selected["update_s"].map(lambda v: f"{v:g}")
        + " / "
        + selected["anchor_s"].map(lambda v: f"{v:g}")
    )
    selected["preset_label"] = pd.Categorical(
        selected["preset_label"], categories=PRESET_ORDER, ordered=True
    )
    selected = selected.sort_values(["preset_label", "split", "fold"])

    table = selected.pivot_table(
        index="preset_label",
        columns=["split", "fold"],
        values="pair",
        aggfunc="first",
        observed=False,
    )

    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    ax.axis("off")
    cell_text = table.fillna("").values.tolist()
    col_labels = [f"{split.replace('-held-out', '')}\nF{fold}" for split, fold in table.columns]
    tab = ax.table(
        cellText=cell_text,
        rowLabels=table.index.tolist(),
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    tab.auto_set_font_size(False)
    tab.set_fontsize(8)
    tab.scale(1.0, 1.35)
    for (row, col), cell in tab.get_celld().items():
        cell.set_edgecolor("#d8dee9")
        if row == 0:
            cell.set_facecolor("#eceff4")
            cell.set_text_props(weight="bold")
        elif col == -1:
            label = table.index[row - 1]
            cell.set_facecolor("#f7f7f7")
            cell.set_text_props(color=COLORS.get(label, "#2e3440"), weight="bold")
    ax.set_title("Selected update_s / anchor_s by Fold", pad=16, weight="bold")
    fig.tight_layout()

    for ext in ["pdf", "png", "svg"]:
        fig.savefig(os.path.join(output_dir, f"figure_parameter_fold_stability.{ext}"), dpi=220)
    plt.close(fig)


def _load_records(
    extraction_cache_path: str,
    reclass_cache_path: str,
    exclusion_registry: str,
    clean_mode: str,
    uptake_mode: str = "raw",
    strength_source: str = "active",
    argument_similarity_threshold: float = 0.60,
    self_similarity_threshold: float = 0.45,
):
    extraction_cache = _load_json_if_exists(extraction_cache_path)
    reclass_cache = _load_json_if_exists(reclass_cache_path)
    timelines = build_participant_timelines(load_debate_dataset())
    excluded = _load_excluded_ids(exclusion_registry)
    if excluded:
        timelines = [t for t in timelines if str(t.worker_id) not in excluded]
    records_all = build_records(timelines, extraction_cache, reclass_cache=None)
    records = filter_records(records_all, clean_mode)
    update_args_by_worker = _prepare_update_args(
        records,
        extraction_cache,
        reclass_cache,
        "received",
        uptake_mode,
        strength_source,
        argument_similarity_threshold,
        self_similarity_threshold,
    )
    return records, update_args_by_worker


def _load_run_summary(run_dir: str) -> dict:
    path = os.path.join(run_dir, "parameter_recommendation_summary.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _rmse_surfaces(
    records,
    update_args_by_worker,
    update_grid: list[float],
    anchor_grid: list[float],
    n_folds: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for split_by in ["group", "topic"]:
        folds = make_folds(records, n_folds, seed, split_by=split_by)
        for fold_idx, _test_records in enumerate(folds, start=1):
            train_records = [
                record
                for idx, fold in enumerate(folds, start=1)
                if idx != fold_idx
                for record in fold
            ]
            for scheme, key, label in PRESET_SPECS:
                subset = _records_for_key(
                    train_records,
                    scheme,
                    key,
                    update_args_by_worker,
                )
                if not subset:
                    continue
                for update_s in update_grid:
                    for anchor_s in anchor_grid:
                        candidate = _candidate_rows(
                            subset,
                            update_args_by_worker,
                            update_s=update_s,
                            anchor_s=anchor_s,
                            fold=fold_idx,
                        )
                        rows.append(
                            {
                                "split": f"{split_by}-held-out",
                                "fold": fold_idx,
                                "scheme": scheme,
                                "preset_key": key,
                                "preset_label": label,
                                "update_s": update_s,
                                "anchor_s": anchor_s,
                                "rmse": _score_rows(candidate, "rmse"),
                                "n_train": len(subset),
                            }
                        )
    return pd.DataFrame(rows)


def _parameter_importance(
    summary: pd.DataFrame,
    labels: list[str] | None = None,
) -> pd.DataFrame:
    """Quantify how much each parameter axis shapes a balanced RMSE surface."""
    rows = []
    for label in labels or PRESET_ORDER:
        subset = summary[summary["preset_label"] == label]
        if subset.empty:
            continue
        pivot = subset.pivot(index="anchor_s", columns="update_s", values="mean_rmse")
        pivot = pivot.reindex(index=sorted(pivot.index), columns=sorted(pivot.columns))
        values = pivot.values.astype(float)
        grand = float(values.mean())
        total_ss = float(((values - grand) ** 2).sum())
        update_means = values.mean(axis=0)
        anchor_means = values.mean(axis=1)
        update_ss = float(values.shape[0] * ((update_means - grand) ** 2).sum())
        anchor_ss = float(values.shape[1] * ((anchor_means - grand) ** 2).sum())
        interaction_ss = max(0.0, total_ss - update_ss - anchor_ss)

        best = subset.loc[subset["mean_rmse"].idxmin()]
        best_update = float(best["update_s"])
        best_anchor = float(best["anchor_s"])
        best_rmse = float(best["mean_rmse"])
        update_values = list(pivot.columns)
        anchor_values = list(pivot.index)
        best_x = update_values.index(best_update)
        best_y = anchor_values.index(best_anchor)
        update_neighbors = []
        for x_idx in [best_x - 1, best_x + 1]:
            if 0 <= x_idx < len(update_values):
                update_neighbors.append(float(values[best_y, x_idx] - best_rmse))
        anchor_neighbors = []
        for y_idx in [best_y - 1, best_y + 1]:
            if 0 <= y_idx < len(anchor_values):
                anchor_neighbors.append(float(values[y_idx, best_x] - best_rmse))

        if total_ss > 0:
            update_share = update_ss / total_ss
            anchor_share = anchor_ss / total_ss
            interaction_share = interaction_ss / total_ss
        else:
            update_share = anchor_share = interaction_share = 0.0

        update_range = float(update_means.max() - update_means.min())
        anchor_range = float(anchor_means.max() - anchor_means.min())
        if anchor_range > update_range * 1.15:
            dominant = "anchor_s"
        elif update_range > anchor_range * 1.15:
            dominant = "update_s"
        else:
            dominant = "mixed"

        rows.append(
            {
                "preset_label": label,
                "best_update_s": best_update,
                "best_anchor_s": best_anchor,
                "best_rmse": best_rmse,
                "update_marginal_range": update_range,
                "anchor_marginal_range": anchor_range,
                "range_ratio_anchor_over_update": (
                    anchor_range / update_range if update_range > 0 else np.inf
                ),
                "update_share": update_share,
                "anchor_share": anchor_share,
                "interaction_share": interaction_share,
                "local_update_cost": float(np.mean(update_neighbors)) if update_neighbors else np.nan,
                "local_anchor_cost": float(np.mean(anchor_neighbors)) if anchor_neighbors else np.nan,
                "dominant_parameter": dominant,
            }
        )
    return pd.DataFrame(rows)


def _plot_parameter_importance(
    importance: pd.DataFrame,
    output_dir: str,
    suffix: str = "",
    title: str = "Parameter Importance Across RMSE Surfaces",
) -> None:
    if importance.empty:
        return
    suffix_part = f"_{suffix}" if suffix else ""
    importance.to_csv(
        os.path.join(output_dir, f"parameter_importance{suffix_part}.csv"),
        index=False,
    )

    plot_df = importance.copy()
    if suffix == "response_types":
        label_order = RESPONSE_TYPE_ORDER
    elif suffix == "evidence_alignment":
        label_order = EVIDENCE_ALIGNMENT_ORDER
    elif suffix == "evidence_alignment_focus":
        label_order = EVIDENCE_ALIGNMENT_FOCUS_ORDER
    else:
        label_order = PRESET_ORDER
    label_order = [label for label in label_order if label in set(plot_df["preset_label"])]
    plot_df["preset_label"] = pd.Categorical(
        plot_df["preset_label"], categories=label_order, ordered=True
    )
    plot_df = plot_df.sort_values("preset_label").dropna(subset=["preset_label"])

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Inter",
                "Avenir Next",
                "Helvetica Neue",
                "Arial",
                "DejaVu Sans",
            ],
            "font.size": 10.5,
            "axes.titlesize": 13,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 10,
        }
    )
    fig_height = max(2.45, 0.46 * len(plot_df) + (0.82 if title else 0.48))
    fig, ax = plt.subplots(figsize=(7.25, fig_height))
    background = "#ffffff"
    fig.patch.set_facecolor(background)
    ax.set_facecolor(background)
    y = np.arange(len(plot_df))
    left = np.zeros(len(plot_df))
    parts = [
        ("update_share", r"Update sensitivity $K$"),
        ("anchor_share", r"Anchor sensitivity $K_a$"),
        ("interaction_share", "interaction"),
    ]
    for column, label in parts:
        width = plot_df[column].to_numpy(dtype=float)
        ax.barh(
            y,
            width,
            left=left,
            color=IMPORTANCE_COLORS[column],
            edgecolor=background,
            linewidth=1.0,
            label=label,
            height=0.62,
        )
        for idx, value in enumerate(width):
            label_text = f"{value:.0%}"
            if value >= 0.055:
                x = left[idx] + value / 2
                ha = "center"
                text_color = background if column != "anchor_share" else "#2f2410"
                bbox = None
            else:
                x = min(left[idx] + value + 0.012, 1.065)
                ha = "left"
                text_color = "#111827"
                bbox = {
                    "boxstyle": "round,pad=0.12",
                    "facecolor": background,
                    "edgecolor": "none",
                    "alpha": 0.92,
                }
            ax.text(
                x,
                y[idx],
                label_text,
                ha=ha,
                va="center",
                fontsize=8.0,
                fontweight=700,
                color=text_color,
                bbox=bbox,
                clip_on=False,
                zorder=5,
            )
        left += width

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["preset_label"].tolist())
    ax.set_ylim(len(plot_df) - 0.48, -0.48)
    for tick in ax.get_yticklabels():
        tick.set_fontweight(700)
        tick.set_color(COLORS.get(tick.get_text(), "#293241"))
    ax.set_xlim(0, 1.08)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xticklabels([f"{int(v * 100)}%" for v in np.linspace(0, 1, 6)])
    ax.set_xlabel(
        "normalized sum-of-squares share",
        fontweight=650,
        color="#293241",
        labelpad=4,
    )
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.7, alpha=0.72)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#d1d5db")
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="x", colors="#293241", width=0)
    ax.tick_params(axis="y", length=0)
    handles, labels = ax.get_legend_handles_labels()
    if title:
        ax.set_title(title, weight=750, fontsize=12.5, color="#111827", pad=22)
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.60, 0.97 if not title else 0.845),
        ncol=3,
        handlelength=1.0,
        handleheight=0.55,
        columnspacing=1.0,
        prop={"weight": 650, "size": 9},
    )
    fig.subplots_adjust(
        left=0.31 if suffix == "response_types" else 0.28,
        right=0.965,
        bottom=0.18,
        top=0.90 if not title else 0.76,
    )
    for ext in ["pdf", "png", "svg"]:
        fig.savefig(
            os.path.join(output_dir, f"figure_parameter_importance{suffix_part}.{ext}"),
            dpi=240,
            bbox_inches="tight",
            pad_inches=0.08,
        )
    plt.close(fig)


def _plot_rmse_surfaces(
    surfaces: pd.DataFrame,
    recs: pd.DataFrame,
    output_dir: str,
    suffix: str = "",
    title: str = "Belief Engine Parameter Selection by Preset Class",
    movers_surfaces: pd.DataFrame | None = None,
    label_order_override: list[str] | None = None,
    show_movers_optimum: bool = True,
    show_legend: bool = True,
    n_heldout_overrides: dict[str, int] | None = None,
) -> None:
    if surfaces.empty:
        return
    large_focus_fonts = suffix == "evidence_alignment_focus"
    fonts = {
        "base": 17 if large_focus_fonts else 12,
        "title": 21 if large_focus_fonts else 15,
        "axis_label": 18 if large_focus_fonts else 13,
        "tick": 15 if large_focus_fonts else 11,
        "annotation": 16 if large_focus_fonts else 12,
        "legend": 15 if large_focus_fonts else 13,
        "suptitle": 22 if large_focus_fonts else 19,
        "guide": 17 if large_focus_fonts else 13,
    }

    summary = (
        surfaces.groupby(["preset_label", "update_s", "anchor_s"], as_index=False)
        .agg(mean_rmse=("rmse", "mean"), sd_rmse=("rmse", "std"), n=("rmse", "size"))
    )
    movers_summary = pd.DataFrame()
    if movers_surfaces is not None and not movers_surfaces.empty:
        movers_summary = (
            movers_surfaces.groupby(["preset_label", "update_s", "anchor_s"], as_index=False)
            .agg(mean_rmse=("rmse", "mean"), sd_rmse=("rmse", "std"), n=("rmse", "size"))
        )
    suffix_part = f"_{suffix}" if suffix else ""
    if label_order_override is not None:
        label_order = label_order_override
    elif suffix == "response_types":
        label_order = RESPONSE_TYPE_ORDER
    elif suffix == "evidence_alignment":
        label_order = EVIDENCE_ALIGNMENT_ORDER
    else:
        label_order = PRESET_ORDER
    available_labels = [label for label in label_order if label in set(summary["preset_label"])]
    summary_for_suffix = summary[summary["preset_label"].isin(available_labels)].copy()
    surfaces_for_suffix = surfaces[surfaces["preset_label"].isin(available_labels)].copy()

    summary_for_suffix.to_csv(
        os.path.join(output_dir, f"parameter_rmse_surface{suffix_part}.csv"),
        index=False,
    )
    surfaces_for_suffix.to_csv(
        os.path.join(output_dir, f"parameter_rmse_surface_by_fold{suffix_part}.csv"),
        index=False,
    )
    _plot_parameter_importance(
        _parameter_importance(summary_for_suffix, labels=available_labels),
        output_dir,
        suffix=suffix,
        title=(
            "Parameter Importance"
            if not suffix
            else (
                "Parameter Importance (Response Types)"
                if suffix == "response_types"
                else (
                    "Parameter Importance (Evidence Alignment)"
                    if suffix == "evidence_alignment"
                    else (
                        "Parameter Importance (Focused Evidence Alignment)"
                        if suffix == "evidence_alignment_focus"
                        else "Parameter Importance (Movers Only)"
                    )
                )
            )
        ),
    )

    plt.rcParams.update(
        {
            "font.size": fonts["base"],
            "axes.titlesize": fonts["title"],
            "axes.labelsize": fonts["axis_label"],
            "xtick.labelsize": fonts["tick"],
            "ytick.labelsize": fonts["tick"],
        }
    )
    regret_values: list[float] = []
    for label in available_labels:
        subset = summary_for_suffix[summary_for_suffix["preset_label"] == label]
        if subset.empty:
            continue
        regret_values.extend((subset["mean_rmse"] - subset["mean_rmse"].min()).tolist())
    regret_array = np.array(regret_values, dtype=float)
    if regret_array.size:
        vmax = float(np.quantile(regret_array, 0.92))
        if vmax <= 0:
            vmax = float(regret_array.max() or 1.0)
    else:
        vmax = 1.0
    vmin = 0.0
    n_panels = max(1, len(available_labels))
    if suffix == "evidence_alignment_focus" and n_panels == 4:
        ncols = 2
    else:
        ncols = min(3, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    fig = plt.figure(figsize=(4.9 * ncols + 0.7, 4.9 * nrows + 0.8))
    gs = fig.add_gridspec(
        nrows,
        ncols + 1,
        width_ratios=[1] * ncols + [0.06],
        wspace=0.18 if large_focus_fonts else 0.24,
        hspace=0.26 if large_focus_fonts else 0.34,
    )
    axes = [fig.add_subplot(gs[row, col]) for row in range(nrows) for col in range(ncols)]
    cax = fig.add_subplot(gs[:, ncols])
    n_heldout_overrides = n_heldout_overrides or {}

    def _fmt_param(value: float) -> str:
        return f"{value:g}"

    def _add_param_label(
        ax,
        x: int,
        y: int,
        update_s: float,
        anchor_s: float,
        rmse: float,
        color: str,
    ) -> None:
        x_min, x_max = ax.get_xlim()
        place_left = x > (x_min + x_max) / 2
        ax.annotate(
            f"u={_fmt_param(update_s)}\na={_fmt_param(anchor_s)}\nRMSE={rmse:.3f}",
            (x, y),
            xytext=((-13 if place_left else 13), 1),
            textcoords="offset points",
            ha=("right" if place_left else "left"),
            va="center",
            fontsize=fonts["annotation"],
            fontweight=900,
            color=color,
            linespacing=0.9,
            zorder=12,
        )

    im = None
    for panel_idx, (ax, label) in enumerate(zip(axes, available_labels)):
        subset = summary_for_suffix[summary_for_suffix["preset_label"] == label]
        if subset.empty:
            ax.axis("off")
            continue
        rec_group = recs[recs["preset_label"] == label]
        if label in n_heldout_overrides:
            n_heldout = int(n_heldout_overrides[label])
        else:
            n_heldout = (
                int(round(float(rec_group["n_heldout"].median())))
                if not rec_group.empty and "n_heldout" in rec_group
                else 0
            )
        pivot = subset.pivot(index="anchor_s", columns="update_s", values="mean_rmse")
        pivot = pivot.reindex(index=sorted(pivot.index, reverse=True), columns=sorted(pivot.columns))
        display_values = pivot.values - float(np.nanmin(pivot.values))
        im = ax.imshow(
            display_values,
            cmap="magma_r",
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
            interpolation="nearest",
        )
        panel_max = float(np.nanmax(display_values))
        if panel_max > 0:
            contour_top = min(panel_max, vmax)
            if contour_top > 0:
                levels = np.linspace(0, contour_top, 5)[1:]
                ax.contour(
                    display_values,
                    levels=levels,
                    colors="#111827",
                    linewidths=0.65,
                    alpha=0.34,
                )
        ax.set_title(
            f"{label}\nN={n_heldout:,}",
            color=COLORS.get(label, "#2e3440"),
            weight="bold",
            pad=10,
        )
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels([f"{v:g}" for v in pivot.columns], rotation=45, ha="right")
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels([f"{v:g}" for v in pivot.index])
        if large_focus_fonts:
            row_idx = panel_idx // ncols
            col_idx = panel_idx % ncols
            if row_idx < nrows - 1:
                ax.tick_params(axis="x", bottom=False, labelbottom=False)
            if col_idx > 0:
                ax.tick_params(axis="y", left=False, labelleft=False)
        ax.set_xticks(np.arange(-0.5, len(pivot.columns), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(pivot.index), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.15, alpha=0.92)
        ax.tick_params(which="minor", bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_visible(False)

        best = subset.loc[subset["mean_rmse"].idxmin()]
        best_x = list(pivot.columns).index(best["update_s"])
        best_y = list(pivot.index).index(best["anchor_s"])
        ax.scatter(
            best_x,
            best_y,
            s=560,
            marker="*",
            color="white",
            edgecolors="white",
            linewidth=2.0,
            zorder=5,
        )
        ax.scatter(
            best_x,
            best_y,
            s=395,
            marker="*",
            color="#f2c94c",
            edgecolors="#17202a",
            linewidth=1.2,
            zorder=7,
        )
        _add_param_label(
            ax,
            best_x,
            best_y,
            float(best["update_s"]),
            float(best["anchor_s"]),
            float(best["mean_rmse"]),
            "#111827",
        )

        if show_movers_optimum and not movers_summary.empty:
            mover_subset = movers_summary[movers_summary["preset_label"] == label]
            if not mover_subset.empty:
                mover_best = mover_subset.loc[mover_subset["mean_rmse"].idxmin()]
                if mover_best["update_s"] in pivot.columns and mover_best["anchor_s"] in pivot.index:
                    mover_x = list(pivot.columns).index(mover_best["update_s"])
                    mover_y = list(pivot.index).index(mover_best["anchor_s"])
                    ax.scatter(
                        mover_x,
                        mover_y,
                        s=260,
                        marker="D",
                        color="white",
                        edgecolors="white",
                        linewidth=2.4,
                        zorder=8,
                    )
                    ax.scatter(
                        mover_x,
                        mover_y,
                        s=190,
                        marker="D",
                        color="#d8873a",
                        edgecolors="#3a2412",
                        linewidth=1.1,
                        zorder=9,
                    )
                    _add_param_label(
                        ax,
                        mover_x,
                        mover_y,
                        float(mover_best["update_s"]),
                        float(mover_best["anchor_s"]),
                        float(mover_best["mean_rmse"]),
                        "#6b330d",
                    )

    for ax in axes[len(available_labels):]:
        ax.axis("off")
    if not large_focus_fonts:
        for ax in axes[-ncols:]:
            ax.set_xlabel("update_s, evidence responsiveness")
        for ax in axes[::ncols]:
            ax.set_ylabel("anchor_s, prior anchoring")

    if im is None:
        plt.close(fig)
        return
    cbar = fig.colorbar(im, cax=cax, extend="max")
    cbar.set_label(
        "RMSE above each panel optimum, lower is better",
        fontsize=fonts["axis_label"],
    )
    cbar.ax.tick_params(labelsize=fonts["tick"])
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="*",
            color="none",
            markerfacecolor="#f2c94c",
            markeredgecolor="#16202a",
            markeredgewidth=1.2,
            markersize=14,
            label="panel optimum" if not show_movers_optimum else "full population optimum",
        ),
    ]
    if show_movers_optimum:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="D",
                color="none",
                markerfacecolor="#d8873a",
                markeredgecolor="#3a2412",
                markeredgewidth=1.1,
                markersize=9,
                label="movers-only optimum",
            )
        )
    if show_legend:
        fig.legend(
            handles=legend_handles,
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.962),
            ncol=2,
            fontsize=fonts["legend"],
            handletextpad=0.65,
            columnspacing=1.8,
        )
    if title:
        fig.suptitle(
            title,
            y=0.985,
            weight="bold",
            fontsize=fonts["suptitle"],
        )
    fig.add_artist(
        FancyArrowPatch(
            (0.075 if large_focus_fonts else 0.035, 0.17),
            (0.075 if large_focus_fonts else 0.035, 0.79),
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=2.0,
            color="#293241",
        )
    )
    fig.text(
        0.055 if large_focus_fonts else 0.018,
        0.48,
        "higher anchor_s -> stronger prior anchoring",
        rotation=90,
        ha="center",
        va="center",
        fontsize=fonts["guide"],
        fontweight=800,
        color="#293241",
        linespacing=1.05,
    )
    fig.add_artist(
        FancyArrowPatch(
            (0.13, 0.062 if large_focus_fonts else 0.035),
            (0.86, 0.062 if large_focus_fonts else 0.035),
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=2.0,
            color="#293241",
        )
    )
    fig.text(
        0.495,
        0.034 if large_focus_fonts else 0.017,
        "higher update_s  ->  stronger evidence responsiveness",
        ha="center",
        va="center",
        fontsize=fonts["guide"],
        fontweight=800,
        color="#293241",
    )
    fig.subplots_adjust(
        left=0.125 if large_focus_fonts else 0.105,
        right=0.95 if large_focus_fonts else 0.93,
        bottom=0.125 if large_focus_fonts else 0.13,
        top=0.835 if (title or show_legend) else 0.92,
    )

    for ext in ["pdf", "png", "svg"]:
        save_kwargs = (
            {"bbox_inches": "tight", "pad_inches": 0.03}
            if large_focus_fonts
            else {}
        )
        fig.savefig(
            os.path.join(output_dir, f"figure_parameter_rmse_surfaces{suffix_part}.{ext}"),
            dpi=240,
            **save_kwargs,
        )
    plt.close(fig)


def _write_readme(output_dir: str, group_dir: str, topic_dir: str) -> None:
    lines = [
        "# Parameter Recommendation Showcase Figures",
        "",
        "Generated from quality-filtered full-population parameter recommendation runs.",
        "",
        f"- Group-held-out source: `{group_dir}`",
        f"- Topic-held-out source: `{topic_dir}`",
        "",
        "## Figures",
        "",
        "- `figure_parameter_preset_space.pdf`: log-log scatter of fold-selected and recommended presets.",
        "- `figure_parameter_fold_stability.pdf`: selected `update_s / anchor_s` pair by fold.",
        "- `figure_parameter_rmse_surfaces.pdf`: one RMSE heatmap per preset class.",
        "- `figure_parameter_rmse_surfaces_evidence_alignment_focus.pdf`: four-panel evidence-alignment summary.",
        "- `figure_parameter_rmse_surfaces_movers.pdf`: same heatmaps after removing non-movers.",
        "- `figure_parameter_importance.pdf`: variance decomposition of each full-population RMSE surface.",
        "- `figure_parameter_importance_evidence_alignment_focus.pdf`: importance decomposition for the focused evidence-alignment panels.",
        "- `figure_parameter_importance_movers.pdf`: same decomposition after removing non-movers.",
        "",
        "Large stars are the recommended median/mode presets; faint points are fold selections.",
        "In the RMSE heatmaps, gold stars mark the full-population optimum and",
        "diamonds mark the movers-only optimum.",
        "`parameter_importance.csv` reports marginal RMSE ranges, local costs,",
        "and variance shares for `update_s`, `anchor_s`, and their interaction.",
    ]
    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate combined parameter recommendation figures."
    )
    parser.add_argument("--group-dir", default=DEFAULT_GROUP_DIR)
    parser.add_argument("--topic-dir", default=DEFAULT_TOPIC_DIR)
    parser.add_argument("--movers-group-dir", default=DEFAULT_MOVERS_GROUP_DIR)
    parser.add_argument("--movers-topic-dir", default=DEFAULT_MOVERS_TOPIC_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--extraction-cache",
        default="validation/data/cache/extractions/extraction_cache.json",
    )
    parser.add_argument(
        "--reclass-cache",
        default="validation/data/cache/extractions/polarity_reclassification.json",
    )
    parser.add_argument(
        "--exclusion-registry",
        default="validation/docs/exclusion_registry.csv",
    )
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-surfaces", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    recs, folds = _load(args.group_dir, args.topic_dir)
    movers_recs, _movers_folds = _load(args.movers_group_dir, args.movers_topic_dir)
    surface_summary = _load_run_summary(args.group_dir)
    mover_surface_summary = _load_run_summary(args.movers_group_dir)
    surface_update_grid = surface_summary.get("update_grid", list(DEFAULT_GRID))
    surface_anchor_grid = surface_summary.get("anchor_grid", list(DEFAULT_GRID))
    mover_update_grid = mover_surface_summary.get("update_grid", list(DEFAULT_GRID))
    mover_anchor_grid = mover_surface_summary.get("anchor_grid", list(DEFAULT_GRID))
    _plot_recommended_space(recs, folds, args.output_dir)
    _plot_fold_stability(folds, args.output_dir)
    if not args.skip_surfaces:
        print("Computing RMSE surfaces for quality-filtered preset classes...")
        records, update_args_by_worker = _load_records(
            args.extraction_cache,
            args.reclass_cache,
            args.exclusion_registry,
            "quality",
            uptake_mode=surface_summary.get("uptake_mode", "raw"),
            strength_source=surface_summary.get("strength_source", "active"),
            argument_similarity_threshold=float(surface_summary.get("argument_similarity_threshold", 0.60)),
            self_similarity_threshold=float(surface_summary.get("self_similarity_threshold", 0.45)),
        )
        surfaces = _rmse_surfaces(
            records,
            update_args_by_worker,
            update_grid=list(surface_update_grid),
            anchor_grid=list(surface_anchor_grid),
            n_folds=args.n_folds,
            seed=args.seed,
        )
        print("Computing RMSE surfaces for movers-only preset classes...")
        mover_records, mover_update_args_by_worker = _load_records(
            args.extraction_cache,
            args.reclass_cache,
            args.exclusion_registry,
            "movers",
            uptake_mode=mover_surface_summary.get("uptake_mode", "raw"),
            strength_source=mover_surface_summary.get("strength_source", "active"),
            argument_similarity_threshold=float(mover_surface_summary.get("argument_similarity_threshold", 0.60)),
            self_similarity_threshold=float(mover_surface_summary.get("self_similarity_threshold", 0.45)),
        )
        mover_surfaces = _rmse_surfaces(
            mover_records,
            mover_update_args_by_worker,
            update_grid=list(mover_update_grid),
            anchor_grid=list(mover_anchor_grid),
            n_folds=args.n_folds,
            seed=args.seed,
        )
        _plot_rmse_surfaces(
            surfaces,
            recs,
            args.output_dir,
            movers_surfaces=mover_surfaces,
        )
        _plot_rmse_surfaces(
            surfaces,
            recs,
            args.output_dir,
            suffix="response_types",
            title="Belief Engine Parameter Selection by Observed Response Type",
            movers_surfaces=mover_surfaces,
        )
        _plot_rmse_surfaces(
            surfaces,
            recs,
            args.output_dir,
            suffix="evidence_alignment",
            title="Belief Engine Parameter Selection by Evidence Alignment",
            movers_surfaces=mover_surfaces,
        )
        _plot_rmse_surfaces(
            surfaces,
            recs,
            args.output_dir,
            suffix="evidence_alignment_focus",
            title="",
            movers_surfaces=None,
            label_order_override=EVIDENCE_ALIGNMENT_FOCUS_ORDER,
            show_movers_optimum=False,
            show_legend=False,
            n_heldout_overrides={
                "All participants": int(surface_summary.get("n_participants", 0))
            },
        )
        _plot_rmse_surfaces(
            mover_surfaces,
            movers_recs,
            args.output_dir,
            suffix="movers",
            title="Belief Engine Parameter Selection by Preset Class (Movers Only)",
        )
    _write_readme(args.output_dir, args.group_dir, args.topic_dir)
    print(f"Saved parameter recommendation showcase figures to: {args.output_dir}")


if __name__ == "__main__":
    main()
