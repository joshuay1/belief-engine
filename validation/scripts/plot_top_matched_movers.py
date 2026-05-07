"""
Visualize the best-matched individual movers from parameter recommendation runs.

The parameter-recommendation showcase directory is built from separate group-
and topic-held-out source runs. This script reads one of those source
`individual_results.csv` files, selects the top matched movers for a chosen
model, reconstructs the accepted-evidence belief path, and writes compact
figures plus audit CSVs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/review_artifact_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/review_artifact_cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

from validation.data.load_debate import load_debate_dataset
from validation.data.preprocess import build_participant_timelines
from validation.data.scale_mapping import log_odds_to_stance, stance_to_log_odds
from validation.evaluation.common import build_records, filter_records
from src.memory.judgement_strategies import VectorJudgementStrategy
from validation.scripts.run_parameter_recommendations import (
    _load_excluded_ids,
    _load_json_if_exists,
    _update_args,
)


DEFAULT_OUTPUT_DIR = (
    "validation/results/individual_opinion_tracking_response5_evidence13x13"
)
DEFAULT_SOURCE_DIR = (
    "validation/results/parameter_recommendations_group_response5_evidence13x13_quality"
)
DEFAULT_PRESET_KEY = "evidence_aligned_mover"

PALETTE_GOLD = "#edae49"
PALETTE_CORAL = "#d1495b"
PALETTE_TEAL = "#00798c"
PALETTE_BLUE = "#30638e"
PALETTE_NAVY = "#003d5b"

COLOR_MODEL = PALETTE_TEAL
COLOR_ACTUAL = PALETTE_CORAL
COLOR_INITIAL = PALETTE_NAVY
COLOR_TEXT = PALETTE_NAVY
COLOR_MUTED = PALETTE_BLUE
COLOR_PRESSURE_UP = PALETTE_GOLD
COLOR_PRESSURE_DOWN = PALETTE_BLUE
COLOR_GRID = "#e9eef2"
COLOR_SPINE = "#d7e0e7"
COLOR_REFERENCE = PALETTE_BLUE
COLOR_NEUTRAL_BAND = "#f7f9fb"

TYPE_SHOWCASE_SPECS = [
    ("All participants", "preset_global", "all"),
    ("Evidence-aligned movers", "preset_evidence_alignment", "evidence_aligned_mover"),
    ("Evidence-opposed movers", "preset_evidence_alignment", "evidence_opposed_mover"),
    ("Stable", "preset_response_type", "stable"),
]
TYPE_SHOWCASE_PREFERRED_WORKERS = {
    "Evidence-aligned movers": "67d3cJDKBT",
}


def _read_summary(source_dir: Path) -> dict:
    path = source_dir / "parameter_recommendation_summary.json"
    if path.exists():
        with path.open() as f:
            return json.load(f)
    return {}


def _record_key(worker_id: str, group_file: str, topic: str) -> tuple[str, str, str]:
    return (str(worker_id), str(group_file), str(topic))


def _short_topic(topic: str, width: int = 54) -> str:
    topic = " ".join(str(topic).split())
    if len(topic) <= width:
        return topic
    return topic[: width - 1].rstrip() + "..."


def _load_record_lookup(
    *,
    clean_mode: str,
    extraction_cache: dict,
    exclusion_registry: str | None,
) -> dict[tuple[str, str, str], object]:
    timelines = build_participant_timelines(load_debate_dataset())
    excluded = _load_excluded_ids(exclusion_registry)
    if excluded:
        timelines = [t for t in timelines if str(t.worker_id) not in excluded]

    # The recommendation run uses reclass_cache=None for participant typing and
    # quality filtering; polarity reclassification is applied only to updates.
    records = filter_records(
        build_records(timelines, extraction_cache, reclass_cache=None),
        clean_mode,
    )
    lookup = {}
    for record in records:
        timeline = record.timeline
        lookup[
            _record_key(timeline.worker_id, timeline.group_file, timeline.topic)
        ] = record
    return lookup


def _select_top_movers(
    df: pd.DataFrame,
    model: str,
    top_n: int,
    preset_key: str | None,
) -> pd.DataFrame:
    required = {
        "model",
        "worker_id",
        "group_file",
        "topic",
        "initial_stance",
        "final_stance_actual",
        "final_stance_predicted",
        "opinion_change_actual",
        "opinion_change_predicted",
        "update_s",
        "anchor_s",
    }
    if preset_key:
        required.add("preset_key")
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in individual_results.csv: {missing}")

    selected = df[df["model"] == model].copy()
    if preset_key:
        selected = selected[selected["preset_key"].astype(str) == preset_key].copy()
    selected["opinion_change_actual"] = pd.to_numeric(
        selected["opinion_change_actual"], errors="coerce"
    )
    selected["opinion_change_predicted"] = pd.to_numeric(
        selected["opinion_change_predicted"], errors="coerce"
    )
    selected["abs_error"] = (
        selected["opinion_change_predicted"] - selected["opinion_change_actual"]
    ).abs()
    selected["direction_match"] = (
        np.sign(selected["opinion_change_actual"])
        == np.sign(selected["opinion_change_predicted"])
    )
    selected = selected[
        selected["opinion_change_actual"].abs().gt(0.01)
        & selected["direction_match"]
        & selected["abs_error"].notna()
    ].copy()
    selected = selected.sort_values(
        ["abs_error", "worker_id", "topic"], kind="mergesort"
    )
    if top_n:
        selected = selected.head(top_n)
    selected.insert(0, "rank", range(1, len(selected) + 1))
    return selected


def _trajectory_for_row(
    row: pd.Series,
    record,
    *,
    extraction_cache: dict,
    reclass_cache: dict,
    exposure_mode: str,
    uptake_mode: str,
    strength_source: str,
    argument_similarity_threshold: float,
    self_similarity_threshold: float,
    judgement: VectorJudgementStrategy | None = None,
) -> pd.DataFrame:
    update_s = float(row["update_s"])
    anchor_s = float(row["anchor_s"])
    initial = float(row["initial_stance"])

    log_odds = stance_to_log_odds(initial) * anchor_s
    rows = [
        {
            "step": 0.0,
            "stance_predicted": initial,
            "stage": "human_initial",
            "strength": np.nan,
            "sign": 0,
        },
        {
            "step": 0.35,
            "stance_predicted": log_odds_to_stance(log_odds),
            "stage": "anchored_prior",
            "strength": np.nan,
            "sign": 0,
        },
    ]

    updates = _update_args(
        record.timeline,
        extraction_cache,
        reclass_cache,
        exposure_mode,
        uptake_mode,
        strength_source,
        argument_similarity_threshold,
        self_similarity_threshold,
        judgement,
    )
    for step, (strength, sign) in enumerate(updates, start=1):
        log_odds += sign * math.log(1.0 + float(strength) * update_s)
        rows.append(
            {
                "step": float(step),
                "stance_predicted": log_odds_to_stance(log_odds),
                "stage": "accepted_evidence",
                "strength": float(strength),
                "sign": int(sign),
            }
        )

    if len(rows) == 2:
        # Keep empty-update cases visible on a nonzero x-axis.
        rows.append(
            {
                "step": 1.0,
                "stance_predicted": rows[-1]["stance_predicted"],
                "stage": "no_accepted_evidence",
                "strength": np.nan,
                "sign": 0,
            }
        )
    out = pd.DataFrame(rows)
    out["rank"] = int(row["rank"])
    out["worker_id"] = str(row["worker_id"])
    out["topic"] = str(row["topic"])
    out["update_s"] = update_s
    out["anchor_s"] = anchor_s
    out["final_stance_actual"] = float(row["final_stance_actual"])
    out["final_stance_predicted_row"] = float(row["final_stance_predicted"])
    out["trajectory_final_diff"] = (
        out["stance_predicted"].iloc[-1] - float(row["final_stance_predicted"])
    )
    return out


def _build_trajectories(
    selected: pd.DataFrame,
    record_lookup: dict[tuple[str, str, str], object],
    *,
    extraction_cache: dict,
    reclass_cache: dict,
    summary: dict,
    args: argparse.Namespace,
) -> pd.DataFrame:
    frames = []
    exposure_mode = str(summary.get("exposure_mode", args.exposure_mode))
    uptake_mode = str(summary.get("uptake_mode", args.uptake_mode))
    strength_source = str(summary.get("strength_source", args.strength_source))
    argument_similarity_threshold = float(
        summary.get(
            "argument_similarity_threshold",
            args.argument_similarity_threshold,
        )
    )
    self_similarity_threshold = float(
        summary.get(
            "self_similarity_threshold",
            args.self_similarity_threshold,
        )
    )
    judgement = None
    if uptake_mode == "human_like":
        judgement = VectorJudgementStrategy(
            similarity_threshold=argument_similarity_threshold,
            self_similarity_threshold=self_similarity_threshold,
            cross_polarity_dedup=False,
        )
    for _, row in selected.iterrows():
        key = _record_key(row["worker_id"], row["group_file"], row["topic"])
        record = record_lookup.get(key)
        if record is None:
            raise KeyError(f"Could not find source timeline for selected row: {key}")
        frames.append(
            _trajectory_for_row(
                row,
                record,
                extraction_cache=extraction_cache,
                reclass_cache=reclass_cache,
                exposure_mode=exposure_mode,
                uptake_mode=uptake_mode,
                strength_source=strength_source,
                argument_similarity_threshold=argument_similarity_threshold,
                self_similarity_threshold=self_similarity_threshold,
                judgement=judgement,
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _score_back_and_forth(
    selected: pd.DataFrame,
    trajectories: pd.DataFrame,
    *,
    epsilon: float,
) -> pd.DataFrame:
    rows = []
    for _, row in selected.iterrows():
        traj = trajectories[trajectories["rank"] == int(row["rank"])].sort_values("step")
        debate_path = traj[traj["stage"].isin(["anchored_prior", "accepted_evidence"])]
        evidence_path = traj[traj["stage"] == "accepted_evidence"]
        if len(debate_path) < 2:
            reversals = 0
            total_movement = 0.0
            net_movement = 0.0
            excess_movement = 0.0
        else:
            y = debate_path["stance_predicted"].to_numpy(dtype=float)
            deltas = np.diff(y)
            total_movement = float(np.abs(deltas).sum())
            net_movement = float(abs(y[-1] - y[0]))
            excess_movement = max(0.0, total_movement - net_movement)
            signs = np.sign(deltas[np.abs(deltas) >= epsilon])
            reversals = int(np.sum(signs[1:] != signs[:-1])) if len(signs) > 1 else 0

        if evidence_path.empty:
            opposing_steps = 0
            opposing_movement = 0.0
        else:
            evidence_y = evidence_path["stance_predicted"].to_numpy(dtype=float)
            prior_y = debate_path["stance_predicted"].iloc[0]
            evidence_deltas = np.diff(np.r_[prior_y, evidence_y])
            actual_direction = np.sign(float(row["opinion_change_actual"]))
            opposing = evidence_deltas * actual_direction < -epsilon
            opposing_steps = int(opposing.sum())
            opposing_movement = float(np.abs(evidence_deltas[opposing]).sum())

        rows.append(
            {
                "rank": int(row["rank"]),
                "reversal_count": reversals,
                "opposing_step_count": opposing_steps,
                "opposing_movement": opposing_movement,
                "total_movement": total_movement,
                "net_movement": net_movement,
                "excess_movement": excess_movement,
                "back_and_forth_score": (
                    reversals * 2.0
                    + opposing_steps * 0.75
                    + excess_movement
                    + opposing_movement
                ),
            }
        )
    return pd.DataFrame(rows)


def _rerank_by_back_and_forth(
    candidates: pd.DataFrame,
    trajectories: pd.DataFrame,
    *,
    top_n: int,
    max_abs_error: float,
    min_reversals: int,
    min_opposing_steps: int,
    epsilon: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = _score_back_and_forth(candidates, trajectories, epsilon=epsilon)
    scored = candidates.merge(metrics, on="rank", how="left")
    filtered = scored[scored["abs_error"].le(max_abs_error)].copy()
    if min_reversals > 0 or min_opposing_steps > 0:
        filtered = filtered[
            filtered["reversal_count"].ge(min_reversals)
            | filtered["opposing_step_count"].ge(min_opposing_steps)
        ].copy()
    if len(filtered) < top_n:
        filtered = scored.copy()

    selected = filtered.sort_values(
        [
            "back_and_forth_score",
            "reversal_count",
            "opposing_step_count",
            "excess_movement",
            "abs_error",
        ],
        ascending=[False, False, False, False, True],
        kind="mergesort",
    ).head(top_n)
    selected = selected.copy()
    selected["candidate_rank"] = selected["rank"].astype(int)
    selected["rank"] = range(1, len(selected) + 1)

    rank_map = dict(zip(selected["candidate_rank"], selected["rank"]))
    selected_trajectories = trajectories[
        trajectories["rank"].astype(int).isin(rank_map)
    ].copy()
    selected_trajectories["candidate_rank"] = selected_trajectories["rank"].astype(int)
    selected_trajectories["rank"] = selected_trajectories["candidate_rank"].map(rank_map)
    selected_trajectories.sort_values(["rank", "step"], inplace=True)
    return selected, selected_trajectories


def _set_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Inter",
                "Avenir Next",
                "Helvetica Neue",
                "Arial",
                "DejaVu Sans",
            ],
            "font.size": 11.3,
            "axes.titlesize": 12.2,
            "axes.labelsize": 12.0,
            "xtick.labelsize": 10.3,
            "ytick.labelsize": 10.3,
            "legend.fontsize": 11.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def _selection_slug(preset_key: str | None, model: str) -> str:
    if preset_key == "evidence_aligned_mover":
        return "evidence_aligned_movers"
    if preset_key:
        return preset_key.replace("_mover", "_movers")
    return model.replace("preset_", "").replace("_", "-")


def _save(fig: plt.Figure, output_dir: Path, stem: str, formats: list[str]) -> None:
    for fmt in formats:
        fig.savefig(output_dir / f"{stem}.{fmt}", dpi=260, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def _format_stance(value: float) -> str:
    return f"{float(value):+.2f}"


def _annotate_endpoint(
    ax: plt.Axes,
    *,
    x: float,
    y: float,
    label: str,
    align: str,
    color: str = COLOR_TEXT,
) -> None:
    y_offset = -13 if y > 0.72 else 13
    x_offset = 9 if align == "left" else -9
    ax.annotate(
        f"{label} {_format_stance(y)}",
        xy=(x, y),
        xytext=(x_offset, y_offset),
        textcoords="offset points",
        ha=align,
        va="center",
        fontsize=10.2,
        fontweight=750,
        color=color,
        bbox={
            "boxstyle": "round,pad=0.14",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.84,
        },
        zorder=12,
    )


def _annotate_point_value(
    ax: plt.Axes,
    *,
    x: float,
    y: float,
    color: str,
    xytext: tuple[float, float],
    ha: str,
) -> None:
    ax.annotate(
        _format_stance(y),
        xy=(x, y),
        xytext=xytext,
        textcoords="offset points",
        ha=ha,
        va="center",
        fontsize=13.2,
        fontweight=850,
        color=color,
        bbox={
            "boxstyle": "round,pad=0.09",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.55,
        },
        zorder=13,
    )


def _value_label_offset(
    y: float, *, role: str, close_endpoints: bool = False
) -> tuple[int, int, str]:
    if role == "start":
        return (6, 16 if y >= 0 else -16, "left")
    if close_endpoints:
        return (-6, 17 if role == "be" else -17, "right")
    return (-6, 16 if y >= 0 else -16, "right")


def _human_bridge_overlaps_model(
    traj: pd.DataFrame,
    human_x: list[float],
    human_y: list[float],
) -> bool:
    steps = traj["step"].to_numpy(dtype=float)
    model_y = traj["stance_predicted"].to_numpy(dtype=float)
    human_line_y = np.interp(steps, human_x, human_y)
    return bool(np.nanmax(np.abs(model_y - human_line_y)) < 0.055)


def _plot_human_bridge(
    ax: plt.Axes,
    human_x: list[float],
    human_y: list[float],
    *,
    curved: bool,
    label: str | None,
) -> None:
    if not curved:
        ax.plot(
            human_x,
            human_y,
            color=COLOR_ACTUAL,
            linestyle="--",
            linewidth=4.65,
            alpha=0.96,
            dash_capstyle="round",
            solid_capstyle="round",
            label=label,
            zorder=6,
        )
        return

    x0, x1 = human_x
    y0, y1 = human_y
    mid_x = (x0 + x1) / 2.0
    mid_y = (y0 + y1) / 2.0
    control_y = mid_y - 0.12 if mid_y <= 0 else mid_y + 0.12
    path = MplPath(
        [(x0, y0), (mid_x, control_y), (x1, y1)],
        [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3],
    )
    patch = PathPatch(
        path,
        facecolor="none",
        edgecolor=COLOR_ACTUAL,
        linewidth=4.65,
        linestyle="--",
        alpha=0.96,
        capstyle="round",
        joinstyle="round",
        label=label,
        zorder=6,
    )
    ax.add_patch(patch)


def _add_type_panel_example(ax: plt.Axes, row: pd.Series) -> None:
    ax.text(
        0.965,
        0.965,
        f"Example: {row['worker_id']}",
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=7.2,
        fontweight=500,
        color=COLOR_MUTED,
        bbox={
            "boxstyle": "round,pad=0.10",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.50,
        },
        zorder=12,
    )


def plot_change_match(
    selected: pd.DataFrame,
    output_dir: Path,
    *,
    stem: str,
    title: str,
    formats: list[str],
) -> None:
    plot_df = selected.sort_values("rank", ascending=False).copy()
    labels = [
        f"{int(row.rank):02d}  {row.worker_id}"
        for row in plot_df.itertuples(index=False)
    ]
    y = np.arange(len(plot_df))

    fig, ax = plt.subplots(figsize=(7.2, max(5.8, 0.29 * len(plot_df) + 1.4)))
    ax.hlines(
        y,
        plot_df["opinion_change_actual"],
        plot_df["opinion_change_predicted"],
        color=COLOR_REFERENCE,
        linewidth=1.2,
        alpha=0.52,
    )
    ax.scatter(
        plot_df["opinion_change_actual"],
        y,
        color=COLOR_ACTUAL,
        s=36,
        label="Human observed change",
        zorder=3,
    )
    ax.scatter(
        plot_df["opinion_change_predicted"],
        y,
        color=COLOR_MODEL,
        s=32,
        label="Model predicted change",
        zorder=3,
    )
    ax.axvline(0, color=COLOR_INITIAL, linewidth=0.9)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Opinion change (final - initial stance)")
    ax.set_title(title, loc="left", pad=10)
    ax.grid(axis="x", color=COLOR_GRID, linewidth=0.8, alpha=0.28)
    ax.legend(frameon=False, loc="lower right")
    _save(fig, output_dir, stem, formats)


def plot_trajectory_grid(
    selected: pd.DataFrame,
    trajectories: pd.DataFrame,
    output_dir: Path,
    *,
    stem: str,
    title: str,
    formats: list[str],
) -> None:
    n = len(selected)
    cols = 4
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(13.2, max(3.0, 2.58 * rows + 1.45)),
        sharey=True,
    )
    axes = np.array(axes).reshape(rows, cols)

    for ax in axes.flat[n:]:
        ax.axis("off")

    for ax, (_, row) in zip(axes.flat, selected.iterrows()):
        traj = trajectories[trajectories["rank"] == int(row["rank"])].copy()
        max_step = max(float(traj["step"].max()), 1.0)
        human_x = [0.0, max_step]
        human_y = [float(row["initial_stance"]), float(row["final_stance_actual"])]

        ax.axhspan(-0.1, 0.1, color=COLOR_NEUTRAL_BAND, alpha=0.12, linewidth=0, zorder=0)
        ax.axhline(
            0,
            color=COLOR_REFERENCE,
            linestyle="--",
            linewidth=0.95,
            alpha=0.82,
            zorder=1,
        )
        ax.plot(
            human_x,
            human_y,
            color=COLOR_ACTUAL,
            linestyle="--",
            linewidth=2.15,
            alpha=0.86,
            solid_capstyle="round",
            zorder=2,
        )
        ax.plot(
            traj["step"],
            traj["stance_predicted"],
            color=COLOR_MODEL,
            linewidth=2.7,
            marker="o",
            markersize=3.8,
            alpha=0.95,
            solid_capstyle="round",
            zorder=3,
        )
        ax.scatter(
            [0.0],
            [float(row["initial_stance"])],
            color=COLOR_INITIAL,
            s=40,
            zorder=5,
        )
        ax.scatter(
            [max_step],
            [float(row["final_stance_actual"])],
            color=COLOR_ACTUAL,
            s=48,
            marker="D",
            edgecolor="white",
            linewidth=0.6,
            zorder=5,
        )
        ax.scatter(
            [max_step],
            [float(row["final_stance_predicted"])],
            color=COLOR_MODEL,
            s=52,
            marker="o",
            edgecolor="white",
            linewidth=0.8,
            zorder=6,
        )
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlim(-0.1, max_step + max(0.6, 0.08 * max_step))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
        ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        ax.grid(axis="y", color=COLOR_GRID, linewidth=0.82, alpha=0.30)
        ax.grid(axis="x", color=COLOR_GRID, linewidth=0.58, alpha=0.18)
        ax.set_axisbelow(True)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color(COLOR_SPINE)
            ax.spines[spine].set_linewidth(0.85)
        ax.tick_params(axis="both", colors=COLOR_TEXT, width=0.85)

        ax.set_title(
            f"{int(row['rank']):02d} {row['worker_id']}  "
            f"err={float(row['abs_error']):.4f}",
            loc="left",
            pad=4,
            fontsize=12.0,
            fontweight=750,
            color=COLOR_TEXT,
        )
        ax.text(
            0.035,
            0.945,
            str(row.get("participant_type", "")).title(),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8.8,
            fontweight=650,
            color=COLOR_MUTED,
        )
        ax.text(
            0.035,
            0.065,
            "\n".join(textwrap.wrap(_short_topic(row["topic"], 62), width=34)[:2]),
            transform=ax.transAxes,
            va="bottom",
            ha="left",
            fontsize=8.0,
            color=COLOR_MUTED,
            bbox={
                "boxstyle": "round,pad=0.16",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.72,
            },
            zorder=5,
        )
        ax.text(
            0.965,
            0.945,
            (
                f"u={float(row['update_s']):g}  a={float(row['anchor_s']):g}\n"
                f"n={int(traj[traj['stage'] == 'accepted_evidence'].shape[0])}"
            ),
            transform=ax.transAxes,
            va="top",
            ha="right",
            fontsize=8.8,
            fontweight=650,
            color=COLOR_MUTED,
            bbox={
                "boxstyle": "round,pad=0.16",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.78,
            },
            zorder=6,
        )

    handles = [
        plt.Line2D([0], [0], color=COLOR_MODEL, marker="o", linewidth=2.7),
        plt.Line2D([0], [0], color=COLOR_ACTUAL, linestyle="--", linewidth=2.15),
        plt.Line2D([0], [0], marker="D", color="none", markerfacecolor=COLOR_ACTUAL),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=COLOR_INITIAL),
    ]
    labels = [
        "Model belief path",
        "Human observed change",
        "Human final",
        "Human initial",
    ]
    fig.legend(
        handles,
        labels,
        ncol=4,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.006),
        handlelength=1.8,
        prop={"weight": 650, "size": 11.0},
    )
    fig.suptitle(
        title,
        y=0.992,
        fontsize=15.0,
        fontweight=700,
        color=COLOR_MUTED,
        linespacing=0.92,
    )
    fig.supxlabel(
        "Accepted evidence update step",
        y=0.052,
        fontsize=13.5,
        fontweight=750,
        color=COLOR_TEXT,
    )
    fig.supylabel("Stance", x=0.018, fontsize=14.0, fontweight=750, color=COLOR_TEXT)
    fig.subplots_adjust(left=0.072, right=0.992, bottom=0.115, top=0.89, wspace=0.15, hspace=0.47)
    _save(fig, output_dir, stem, formats)


def _largest_change_subset(
    selected: pd.DataFrame,
    trajectories: pd.DataFrame,
    *,
    top_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    subset = selected.copy()
    subset["abs_actual_change"] = subset["opinion_change_actual"].abs()
    subset = subset.sort_values(
        ["abs_actual_change", "back_and_forth_score", "abs_error"],
        ascending=[False, False, True],
        kind="mergesort",
    ).head(top_n)
    subset = subset.copy()
    subset["source_rank"] = subset["rank"].astype(int)
    subset["rank"] = range(1, len(subset) + 1)

    rank_map = dict(zip(subset["source_rank"], subset["rank"]))
    sub_trajectories = trajectories[
        trajectories["rank"].astype(int).isin(rank_map)
    ].copy()
    sub_trajectories["source_rank"] = sub_trajectories["rank"].astype(int)
    sub_trajectories["rank"] = sub_trajectories["source_rank"].map(rank_map)
    sub_trajectories.sort_values(["rank", "step"], inplace=True)
    return subset, sub_trajectories


def _type_showcase_candidate_pool(
    df: pd.DataFrame,
    *,
    max_per_type: int,
) -> pd.DataFrame:
    frames = []
    for surface_label, model, preset_key in TYPE_SHOWCASE_SPECS:
        subset = df[(df["model"] == model) & (df["preset_key"] == preset_key)].copy()
        if subset.empty:
            raise ValueError(
                f"No individual rows found for {surface_label}: {model}/{preset_key}"
            )
        subset["abs_error"] = (
            subset["opinion_change_predicted"] - subset["opinion_change_actual"]
        ).abs()
        subset["abs_actual_change"] = subset["opinion_change_actual"].abs()
        subset["abs_predicted_change"] = subset["opinion_change_predicted"].abs()
        subset["direction_match"] = (
            np.sign(subset["opinion_change_actual"])
            == np.sign(subset["opinion_change_predicted"])
        )
        subset["surface_label"] = surface_label
        subset["preferred_type_example"] = (
            subset["worker_id"].astype(str)
            == TYPE_SHOWCASE_PREFERRED_WORKERS.get(surface_label, "")
        )

        if surface_label == "All participants":
            pool = subset[
                subset["abs_actual_change"].ge(0.25) & subset["direction_match"]
            ].copy()
            reason = "global all-participant row with a visible matched change"
            sort_cols = [
                "preferred_type_example",
                "n_arguments",
                "abs_actual_change",
                "abs_predicted_change",
                "abs_error",
            ]
            ascending = [False, False, False, False, True]
        elif surface_label == "Evidence-aligned movers":
            pool = subset[
                subset["abs_actual_change"].ge(0.25) & subset["direction_match"]
            ].copy()
            reason = "accepted evidence pressure and human movement align"
            sort_cols = [
                "preferred_type_example",
                "n_arguments",
                "abs_actual_change",
                "abs_predicted_change",
                "abs_error",
            ]
            ascending = [False, False, False, False, True]
        elif surface_label == "Evidence-opposed movers":
            pool = subset[
                subset["abs_actual_change"].ge(0.25)
                & (
                    np.sign(subset["opinion_change_actual"])
                    != np.sign(subset["opinion_change_predicted"])
                )
            ].copy()
            if pool.empty:
                pool = subset[subset["abs_actual_change"].ge(0.25)].copy()
            pool["opposition_showcase_score"] = (
                pool["abs_actual_change"] + pool["abs_predicted_change"]
            )
            reason = "human movement goes against the accepted-evidence model pressure"
            sort_cols = [
                "preferred_type_example",
                "n_arguments",
                "opposition_showcase_score",
                "abs_actual_change",
                "abs_predicted_change",
            ]
            ascending = [False, False, False, False, False]
        else:
            pool = subset.copy()
            reason = "human initial and final stance are unchanged"
            sort_cols = [
                "preferred_type_example",
                "n_arguments",
                "abs_predicted_change",
                "abs_error",
            ]
            ascending = [False, False, False, True]

        if pool.empty:
            pool = subset.copy()
        preferred = subset[subset["preferred_type_example"]].copy()
        pool = pd.concat([preferred, pool], ignore_index=True)
        pool = pool.drop_duplicates(
            ["model", "preset_key", "worker_id", "group_file", "topic"],
            keep="first",
        )
        pool["showcase_reason"] = reason
        pool = pool.sort_values(
            sort_cols,
            ascending=ascending,
            kind="mergesort",
        ).head(max_per_type)
        frames.append(pool)

    candidates = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    candidates.insert(0, "rank", range(1, len(candidates) + 1))
    return candidates


def _select_zigzag_type_showcase_examples(
    candidates: pd.DataFrame,
    trajectories: pd.DataFrame,
    *,
    epsilon: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = _score_back_and_forth(candidates, trajectories, epsilon=epsilon)
    scored = candidates.merge(metrics, on="rank", how="left")
    scored["zigzag_score"] = (
        scored["reversal_count"].fillna(0) * 2.5
        + scored["opposing_step_count"].fillna(0) * 0.6
        + scored["excess_movement"].fillna(0) * 4.0
        + scored["total_movement"].fillna(0) * 1.5
    )

    selected_rows = []
    for surface_label, _, _ in TYPE_SHOWCASE_SPECS:
        group = scored[scored["surface_label"] == surface_label].copy()
        if group.empty:
            continue
        preferred = group[group["preferred_type_example"]]
        if not preferred.empty:
            row = preferred.sort_values(
                ["zigzag_score", "reversal_count", "total_movement"],
                ascending=[False, False, False],
                kind="mergesort",
            ).iloc[0]
        else:
            row = group.sort_values(
                [
                    "zigzag_score",
                    "reversal_count",
                    "opposing_step_count",
                    "excess_movement",
                    "total_movement",
                    "abs_actual_change",
                ],
                ascending=[False, False, False, False, False, False],
                kind="mergesort",
            ).iloc[0]
        selected_rows.append(row)

    selected = pd.DataFrame(selected_rows).reset_index(drop=True)
    selected["source_rank"] = selected["rank"].astype(int)
    selected["rank"] = range(1, len(selected) + 1)

    rank_map = dict(zip(selected["source_rank"], selected["rank"]))
    selected_trajectories = trajectories[
        trajectories["rank"].astype(int).isin(rank_map)
    ].copy()
    selected_trajectories["source_rank"] = selected_trajectories["rank"].astype(int)
    selected_trajectories["rank"] = selected_trajectories["source_rank"].map(rank_map)
    selected_trajectories.sort_values(["rank", "step"], inplace=True)
    return selected, selected_trajectories


def plot_largest_change_row(
    selected: pd.DataFrame,
    trajectories: pd.DataFrame,
    output_dir: Path,
    *,
    stem: str,
    title: str,
    formats: list[str],
) -> None:
    n = len(selected)
    fig, axes = plt.subplots(1, n, figsize=(14.2, 4.95), sharey=True)
    axes = np.array(axes).reshape(1, n).ravel()

    for idx, (ax, (_, row)) in enumerate(zip(axes, selected.iterrows())):
        traj = trajectories[trajectories["rank"] == int(row["rank"])].copy()
        max_step = max(float(traj["step"].max()), 1.0)
        human_x = [0.0, max_step]
        human_y = [float(row["initial_stance"]), float(row["final_stance_actual"])]
        topic = "\n".join(textwrap.wrap(_short_topic(row["topic"], 66), width=24)[:3])

        ax.axhspan(-0.1, 0.1, color=COLOR_NEUTRAL_BAND, alpha=0.12, linewidth=0, zorder=0)
        ax.axhline(
            0,
            color=COLOR_REFERENCE,
            linestyle="--",
            linewidth=1.05,
            alpha=0.84,
            zorder=1,
        )
        ax.plot(
            traj["step"],
            traj["stance_predicted"],
            color=COLOR_MODEL,
            linewidth=3.30,
            marker="o",
            markersize=4.9,
            alpha=0.92,
            solid_capstyle="round",
            label="Model belief path" if idx == 0 else None,
            zorder=3,
        )
        ax.plot(
            human_x,
            human_y,
            color=COLOR_ACTUAL,
            linestyle="--",
            linewidth=3.90,
            alpha=0.96,
            dash_capstyle="round",
            solid_capstyle="round",
            label="Human observed change" if idx == 0 else None,
            zorder=7,
        )
        ax.scatter(
            [0.0],
            [float(row["initial_stance"])],
            color=COLOR_INITIAL,
            s=62,
            label="Human initial" if idx == 0 else None,
            zorder=9,
        )
        ax.scatter(
            [max_step],
            [float(row["final_stance_actual"])],
            color=COLOR_ACTUAL,
            s=76,
            marker="D",
            edgecolor="white",
            linewidth=0.9,
            label="Human final" if idx == 0 else None,
            zorder=10,
        )
        ax.scatter(
            [max_step],
            [float(row["final_stance_predicted"])],
            color=COLOR_MODEL,
            s=60,
            marker="o",
            edgecolor="white",
            linewidth=0.9,
            zorder=5,
        )

        ax.set_ylim(-1.06, 1.06)
        ax.set_xlim(-0.1, max_step + max(0.7, 0.08 * max_step))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
        ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        ax.grid(axis="y", color=COLOR_GRID, linewidth=0.95, alpha=0.34)
        ax.grid(axis="x", color=COLOR_GRID, linewidth=0.62, alpha=0.20)
        ax.set_axisbelow(True)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color(COLOR_SPINE)
            ax.spines[spine].set_linewidth(0.95)
        ax.tick_params(axis="both", colors=COLOR_TEXT, width=0.95, labelsize=11.8)

        ax.text(
            0.5,
            1.205,
            str(row["worker_id"]),
            transform=ax.transAxes,
            va="bottom",
            ha="center",
            fontsize=15.4,
            fontweight=800,
            color=COLOR_TEXT,
            clip_on=False,
        )
        topic = "\n".join(
            textwrap.wrap(f"Topic: {_short_topic(row['topic'], 72)}", width=30)[:3]
        )
        ax.text(
            0.5,
            1.045,
            topic,
            transform=ax.transAxes,
            va="bottom",
            ha="center",
            fontsize=9.8,
            linespacing=0.94,
            color=COLOR_MUTED,
            clip_on=False,
        )
        ax.text(
            0.965,
            0.925,
            (
                f"u={float(row['update_s']):g}  a={float(row['anchor_s']):g}\n"
                f"|Δhuman|={float(row['abs_actual_change']):.2f}"
            ),
            transform=ax.transAxes,
            va="top",
            ha="right",
            fontsize=10.0,
            fontweight=700,
            color=COLOR_MUTED,
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.80,
            },
            zorder=10,
        )

    handles, labels = axes[0].get_legend_handles_labels()
    ordered = [0, 1, 3, 2]
    handles = [handles[i] for i in ordered]
    labels = [labels[i] for i in ordered]
    fig.legend(
        handles,
        labels,
        ncol=4,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        handlelength=1.9,
        prop={"weight": 700, "size": 12.0},
    )
    fig.suptitle(
        title,
        y=0.985,
        fontsize=17.0,
        fontweight=800,
        color=COLOR_MUTED,
    )
    fig.supxlabel(
        "Accepted evidence update step",
        y=0.088,
        fontsize=14.8,
        fontweight=800,
        color=COLOR_TEXT,
    )
    fig.supylabel("Stance", x=0.018, fontsize=15.4, fontweight=800, color=COLOR_TEXT)
    fig.subplots_adjust(left=0.07, right=0.992, bottom=0.22, top=0.68, wspace=0.16)
    _save(fig, output_dir, stem, formats)


def plot_type_showcase_row(
    selected: pd.DataFrame,
    trajectories: pd.DataFrame,
    output_dir: Path,
    *,
    stem: str,
    title: str,
    formats: list[str],
) -> None:
    n = len(selected)
    fig, axes = plt.subplots(1, n, figsize=(13.4, 4.15), sharey=True)
    axes = np.array(axes).reshape(1, n).ravel()

    for idx, (ax, (_, row)) in enumerate(zip(axes, selected.iterrows())):
        traj = trajectories[trajectories["rank"] == int(row["rank"])].copy()
        max_step = max(float(traj["step"].max()), 1.0)
        human_x = [0.0, max_step]
        human_y = [float(row["initial_stance"]), float(row["final_stance_actual"])]
        topic = "\n".join(textwrap.wrap(_short_topic(row["topic"], 72), width=30)[:3])

        ax.axhline(
            0,
            color=COLOR_REFERENCE,
            linestyle="--",
            linewidth=1.0,
            alpha=0.72,
            zorder=1,
        )
        ax.plot(
            traj["step"],
            traj["stance_predicted"],
            color=COLOR_MODEL,
            linewidth=4.05,
            alpha=0.94,
            solid_capstyle="round",
            label="BE agent belief path" if idx == 0 else None,
            zorder=8,
        )
        prior_points = traj[traj["sign"].eq(0)]
        if not prior_points.empty:
            ax.scatter(
                prior_points["step"],
                prior_points["stance_predicted"],
                color=COLOR_MODEL,
                s=52,
                marker="o",
                edgecolor="white",
                linewidth=0.55,
                zorder=9,
            )
        for sign, marker, marker_color in [
            (1, "^", COLOR_PRESSURE_UP),
            (-1, "v", COLOR_PRESSURE_DOWN),
        ]:
            pressure_points = traj[traj["sign"].eq(sign)]
            if pressure_points.empty:
                continue
            ax.scatter(
                pressure_points["step"],
                pressure_points["stance_predicted"],
                color=marker_color,
                s=72,
                marker=marker,
                edgecolor="white",
                linewidth=0.55,
                zorder=10,
            )
        _plot_human_bridge(
            ax,
            human_x,
            human_y,
            curved=_human_bridge_overlaps_model(traj, human_x, human_y),
            label="Human observed change" if idx == 0 else None,
        )
        ax.scatter(
            [0.0],
            [float(row["initial_stance"])],
            color=COLOR_INITIAL,
            s=78,
            label="Human initial" if idx == 0 else None,
            zorder=11,
        )
        ax.scatter(
            [max_step],
            [float(row["final_stance_actual"])],
            color=COLOR_ACTUAL,
            s=96,
            marker="D",
            edgecolor="white",
            linewidth=0.9,
            label="Human final" if idx == 0 else None,
            zorder=14,
        )
        final_pressure = int(traj["sign"].iloc[-1])
        final_marker = "^" if final_pressure > 0 else "v" if final_pressure < 0 else "o"
        final_color = (
            COLOR_PRESSURE_UP
            if final_pressure > 0
            else COLOR_PRESSURE_DOWN
            if final_pressure < 0
            else COLOR_MODEL
        )
        ax.scatter(
            [max_step],
            [float(row["final_stance_predicted"])],
            color=final_color,
            s=96,
            marker=final_marker,
            edgecolor="white",
            linewidth=0.9,
            zorder=11,
        )
        human_start = float(row["initial_stance"])
        human_end = float(row["final_stance_actual"])
        be_end = float(row["final_stance_predicted"])
        close_endpoints = abs(human_end - be_end) < 0.12
        start_dx, start_dy, start_ha = _value_label_offset(human_start, role="start")
        be_dx, be_dy, be_ha = _value_label_offset(
            be_end, role="be", close_endpoints=close_endpoints
        )
        human_dx, human_dy, human_ha = _value_label_offset(
            human_end, role="human", close_endpoints=close_endpoints
        )
        _annotate_point_value(
            ax,
            x=0.0,
            y=human_start,
            color=COLOR_INITIAL,
            xytext=(start_dx, start_dy),
            ha=start_ha,
        )
        _annotate_point_value(
            ax,
            x=max_step,
            y=be_end,
            color=COLOR_MODEL,
            xytext=(be_dx, be_dy),
            ha=be_ha,
        )
        _annotate_point_value(
            ax,
            x=max_step,
            y=human_end,
            color=COLOR_ACTUAL,
            xytext=(human_dx, human_dy),
            ha=human_ha,
        )

        ax.set_ylim(-1.06, 1.06)
        ax.set_xlim(-0.1, max_step + max(0.7, 0.08 * max_step))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
        ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        ax.grid(False)
        ax.set_axisbelow(True)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color(COLOR_SPINE)
            ax.spines[spine].set_linewidth(0.95)
        ax.tick_params(axis="both", colors=COLOR_TEXT, width=0.95, labelsize=13.5)
        _add_type_panel_example(ax, row)

        ax.text(
            0.5,
            1.13,
            str(row["surface_label"]),
            transform=ax.transAxes,
            va="bottom",
            ha="center",
            fontsize=14.0,
            fontweight=800,
            color=COLOR_TEXT,
            clip_on=False,
        )
        ax.text(
            0.5,
            1.02,
            f"u={float(row['update_s']):g}  a={float(row['anchor_s']):g}",
            transform=ax.transAxes,
            va="bottom",
            ha="center",
            fontsize=12.8,
            fontweight=750,
            color=COLOR_MUTED,
            clip_on=False,
        )
        ax.text(
            0.045,
            0.08,
            topic,
            transform=ax.transAxes,
            va="bottom",
            ha="left",
            fontsize=11.7,
            linespacing=0.94,
            color=COLOR_MUTED,
            bbox={
                "boxstyle": "round,pad=0.16",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.74,
            },
            zorder=11,
        )
        if idx == 0:
            handles, labels = ax.get_legend_handles_labels()
            pressure_handles = [
                plt.Line2D(
                    [0],
                    [0],
                    color=COLOR_PRESSURE_UP,
                    marker="^",
                    linestyle="None",
                    markersize=7.2,
                    markeredgecolor="white",
                    markeredgewidth=0.7,
                ),
                plt.Line2D(
                    [0],
                    [0],
                    color=COLOR_PRESSURE_DOWN,
                    marker="v",
                    linestyle="None",
                    markersize=7.2,
                    markeredgecolor="white",
                    markeredgewidth=0.7,
                ),
            ]
            ax.legend(
                handles[:2] + pressure_handles,
                ["BE agent", "Human", "Up pressure", "Down pressure"],
                loc="upper left",
                bbox_to_anchor=(0.03, 0.98),
                frameon=True,
                facecolor="white",
                edgecolor="none",
                framealpha=0.86,
                fontsize=10.8,
                handlelength=1.7,
                borderpad=0.28,
                labelspacing=0.18,
            )

    fig.supylabel("Stance", x=0.018, fontsize=15.4, fontweight=800, color=COLOR_TEXT)
    fig.supxlabel(
        "Accepted evidence update turn",
        y=0.028,
        fontsize=14.2,
        fontweight=800,
        color=COLOR_TEXT,
    )
    fig.subplots_adjust(left=0.07, right=0.992, bottom=0.18, top=0.80, wspace=0.055)
    _save(fig, output_dir, stem, formats)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot top matched movers from parameter recommendation individual rows."
    )
    parser.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default="preset_evidence_alignment")
    parser.add_argument(
        "--preset-key",
        default=DEFAULT_PRESET_KEY,
        help="Optional preset_key filter before ranking; default keeps evidence-aligned movers.",
    )
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument(
        "--selection-mode",
        choices=["back_and_forth", "best_fit"],
        default="back_and_forth",
    )
    parser.add_argument("--candidate-pool", type=int, default=0)
    parser.add_argument("--max-abs-error", type=float, default=0.15)
    parser.add_argument("--min-reversals", type=int, default=1)
    parser.add_argument("--min-opposing-steps", type=int, default=1)
    parser.add_argument("--back-and-forth-epsilon", type=float, default=0.015)
    parser.add_argument("--top-change-row-n", type=int, default=4)
    parser.add_argument("--type-showcase-candidate-pool", type=int, default=80)
    parser.add_argument("--skip-type-showcase", action="store_true")
    parser.add_argument("--split-label", default=None)
    parser.add_argument("--formats", default="pdf,png")
    parser.add_argument("--include-change-match", action="store_true")
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
    parser.add_argument("--exposure-mode", default="received")
    parser.add_argument("--uptake-mode", default="human_like")
    parser.add_argument("--strength-source", default="classifier")
    parser.add_argument("--argument-similarity-threshold", type=float, default=0.60)
    parser.add_argument("--self-similarity-threshold", type=float, default=0.45)
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    formats = [fmt.strip().lstrip(".") for fmt in args.formats.split(",") if fmt.strip()]
    summary = _read_summary(source_dir)
    split_label = args.split_label or str(summary.get("split_by", source_dir.name))
    clean_mode = str(summary.get("clean_mode", "quality"))

    individual_path = source_dir / "individual_results.csv"
    if not individual_path.exists():
        raise FileNotFoundError(f"No individual_results.csv found at {individual_path}")

    df = pd.read_csv(individual_path)
    candidates = _select_top_movers(df, args.model, 0, args.preset_key)
    if args.selection_mode == "best_fit":
        selected = candidates.head(args.top_n).copy()
        selected["rank"] = range(1, len(selected) + 1)
    else:
        selected = candidates.copy()
        if args.candidate_pool and args.candidate_pool > 0:
            selected = selected.head(args.candidate_pool).copy()
        selected["rank"] = range(1, len(selected) + 1)
    extraction_cache = _load_json_if_exists(args.extraction_cache)
    reclass_cache = _load_json_if_exists(args.reclass_cache)
    record_lookup = _load_record_lookup(
        clean_mode=clean_mode,
        extraction_cache=extraction_cache,
        exclusion_registry=args.exclusion_registry,
    )
    trajectories = _build_trajectories(
        selected,
        record_lookup,
        extraction_cache=extraction_cache,
        reclass_cache=reclass_cache,
        summary=summary,
        args=args,
    )
    if args.selection_mode == "back_and_forth":
        selected, trajectories = _rerank_by_back_and_forth(
            selected,
            trajectories,
            top_n=args.top_n,
            max_abs_error=args.max_abs_error,
            min_reversals=args.min_reversals,
            min_opposing_steps=args.min_opposing_steps,
            epsilon=args.back_and_forth_epsilon,
        )

    selection_slug = _selection_slug(args.preset_key, args.model)
    selected_out = output_dir / f"top{args.top_n}_{selection_slug}_{split_label}.csv"
    trajectory_out = (
        output_dir / f"top{args.top_n}_{selection_slug[:-1] if selection_slug.endswith('s') else selection_slug}_trajectories_{split_label}.csv"
    )
    selected.to_csv(selected_out, index=False)
    trajectories.to_csv(trajectory_out, index=False)

    top_change_selected = pd.DataFrame()
    top_change_trajectories = pd.DataFrame()
    if args.top_change_row_n and args.top_change_row_n > 0:
        top_change_selected, top_change_trajectories = _largest_change_subset(
            selected,
            trajectories,
            top_n=args.top_change_row_n,
        )
        top_change_selected.to_csv(
            output_dir / f"top{args.top_change_row_n}_largest_change_{selection_slug}_{split_label}.csv",
            index=False,
        )
        top_change_trajectories.to_csv(
            output_dir
            / f"top{args.top_change_row_n}_largest_change_{selection_slug[:-1] if selection_slug.endswith('s') else selection_slug}_trajectories_{split_label}.csv",
            index=False,
        )

    type_showcase_selected = pd.DataFrame()
    type_showcase_trajectories = pd.DataFrame()
    if not args.skip_type_showcase:
        type_showcase_candidates = _type_showcase_candidate_pool(
            df,
            max_per_type=args.type_showcase_candidate_pool,
        )
        type_showcase_candidate_trajectories = _build_trajectories(
            type_showcase_candidates,
            record_lookup,
            extraction_cache=extraction_cache,
            reclass_cache=reclass_cache,
            summary=summary,
            args=args,
        )
        type_showcase_selected, type_showcase_trajectories = (
            _select_zigzag_type_showcase_examples(
                type_showcase_candidates,
                type_showcase_candidate_trajectories,
                epsilon=args.back_and_forth_epsilon,
            )
        )
        type_showcase_selected.to_csv(
            output_dir / f"type_showcase_examples_{split_label}.csv",
            index=False,
        )
        type_showcase_trajectories.to_csv(
            output_dir / f"type_showcase_example_trajectories_{split_label}.csv",
            index=False,
        )

    _set_plot_style()
    title = (
        f"Top {len(selected)} Evidence-Aligned Movers with Back-and-Forth Trajectories\n"
        f"{split_label.title()}-held-out | evidence uptake $u$ and prior anchoring $a$ shown per panel"
    )
    if args.include_change_match:
        plot_change_match(
            selected,
            output_dir,
            stem=f"figure_top{args.top_n}_{selection_slug}_change_match_{split_label}",
            title=title,
            formats=formats,
        )
    plot_trajectory_grid(
        selected,
        trajectories,
        output_dir,
        stem=f"figure_top{args.top_n}_{selection_slug}_tracking_{split_label}",
        title=title,
        formats=formats,
    )
    if not top_change_selected.empty:
        plot_largest_change_row(
            top_change_selected,
            top_change_trajectories,
            output_dir,
            stem=f"figure_top{args.top_change_row_n}_largest_change_{selection_slug}_row_{split_label}",
            title="Four largest evidence-aligned human opinion changes",
            formats=formats,
        )
    if not type_showcase_selected.empty:
        plot_type_showcase_row(
            type_showcase_selected,
            type_showcase_trajectories,
            output_dir,
            stem=f"figure_individual_type_showcase_{split_label}",
            title="Individual examples of the four evidence-alignment surface groups",
            formats=formats,
        )

    max_diff = (
        trajectories.groupby("rank")["trajectory_final_diff"].last().abs().max()
        if not trajectories.empty
        else math.nan
    )
    print(f"Selected rows: {selected_out}")
    print(f"Trajectory rows: {trajectory_out}")
    print(f"Max reconstructed final-vs-row difference: {max_diff:.6g}")


if __name__ == "__main__":
    main()
