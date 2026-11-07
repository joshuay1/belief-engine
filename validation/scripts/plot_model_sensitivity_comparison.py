#!/usr/bin/env python3
"""Plot matched canonical sensitivity sweeps across language models.

The raw log folders contain resolved configs with API settings, so this script
extracts only paper-facing metrics from results.json files and writes clean CSVs
plus compact comparison figures.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOGS_DIR = REPO_ROOT / "logs"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "validation" / "results" / "model_sensitivity_comparison"

RUN_NAME_RE = re.compile(
    r"^(CoreBot|OpenClaw)_(Update|Anchor)_([0-9.]+)_.*_([0-9]{3})_"
)

MODEL_LABELS = {
    "GPT-4o-mini": "GPT-4o-mini",
    "QWEN": "Qwen3.5 9B",
    "GEMMA4": "Gemma4 E4B",
}

AGENT_LABELS = {
    "CoreBot": "BE Agent",
    "OpenClaw": "OpenClaw",
}

SWEEP_LABELS = {
    "Update": r"Update sensitivity $K$",
    "Anchor": r"Anchor sensitivity $K_a$",
}

MODEL_COLORS = {
    "GPT-4o-mini": "#19747e",
    "QWEN": "#b34a4a",
    "GEMMA4": "#c77724",
}

PARAMETER_COLORS = {
    0.2: "#19747e",
    0.4: "#5a8f7b",
    0.6: "#b49b3f",
    0.8: "#d07a24",
    1.0: "#b34a4a",
}


def import_plotting():
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    return plt, np, pd


def model_from_folder(path: Path) -> str:
    return path.name.split("_canonical_sensitivity_sweep", 1)[0]


def iter_model_folders(logs_dir: Path) -> Iterable[Path]:
    yield from sorted(logs_dir.glob("*canonical_sensitivity_sweep*"))


def first_round_at_or_below(values: List[float], threshold: float) -> Optional[int]:
    for idx, value in enumerate(values):
        if value <= threshold:
            return idx
    return None


def collect_rows(logs_dir: Path) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    run_rows: List[Dict[str, Any]] = []
    round_rows: List[Dict[str, Any]] = []

    for model_dir in iter_model_folders(logs_dir):
        model = model_from_folder(model_dir)
        for results_path in sorted(model_dir.glob("*/results.json")):
            run_dir = results_path.parent
            match = RUN_NAME_RE.match(run_dir.name)
            if not match:
                continue
            agent, sweep, parameter_value, trial = match.groups()
            with results_path.open() as f:
                results = json.load(f)

            history = results.get("stance_history") or []
            subject_stances = [
                float(entry["Subject"])
                for entry in history
                if "Subject" in entry and entry["Subject"] is not None
            ]
            if not subject_stances:
                continue

            initial = float(results["initial_stance"])
            final = float(results["final_stance"])
            counter_stance = float(history[0].get("Counter", -1.0)) if history else -1.0
            midpoint = (initial + counter_stance) / 2.0
            movement_toward_counter = abs(initial - counter_stance) - abs(final - counter_stance)
            switch_round = first_round_at_or_below(subject_stances, 0.0)
            halfway_round = first_round_at_or_below(subject_stances, midpoint)

            replacements = results.get("replacements") or []
            counter_arguments_added = sum(
                1
                for row in replacements
                if row.get("receiver") == "Subject"
                and row.get("sender") == "Counter"
                and row.get("status") == "added"
            )

            base = {
                "model": model,
                "model_label": MODEL_LABELS.get(model, model),
                "agent": agent,
                "agent_label": AGENT_LABELS.get(agent, agent),
                "sweep": sweep,
                "sweep_label": SWEEP_LABELS.get(sweep, sweep),
                "parameter_value": float(parameter_value),
                "trial": int(trial),
                "run_folder": run_dir.name,
            }
            run_rows.append(
                {
                    **base,
                    "initial_stance": initial,
                    "final_stance": final,
                    "stance_shift": float(results["stance_shift"]),
                    "movement_toward_counter": movement_toward_counter,
                    "switched_sides": switch_round is not None,
                    "switch_round": switch_round,
                    "halfway_round": halfway_round,
                    "new_arguments_count": results.get("new_arguments_count"),
                    "counter_arguments_added": counter_arguments_added,
                    "final_memory_count": results.get("final_memory_count"),
                }
            )
            for entry in history:
                if "Subject" not in entry:
                    continue
                round_rows.append(
                    {
                        **base,
                        "round": int(entry.get("round", len(round_rows))),
                        "subject_stance": float(entry["Subject"]),
                        "subject_evaluated": entry.get("Subject_evaluated"),
                        "counter_stance": entry.get("Counter"),
                    }
                )

    return run_rows, round_rows


def summarize_metrics(runs_df):
    grouped = (
        runs_df.groupby(["model", "model_label", "agent", "agent_label", "sweep", "sweep_label"])
        .agg(
            n=("final_stance", "size"),
            mean_final_stance=("final_stance", "mean"),
            mean_movement_toward_counter=("movement_toward_counter", "mean"),
            switch_rate=("switched_sides", "mean"),
            mean_switch_round=("switch_round", "mean"),
            mean_halfway_round=("halfway_round", "mean"),
            mean_new_arguments=("new_arguments_count", "mean"),
            mean_counter_arguments_added=("counter_arguments_added", "mean"),
        )
        .reset_index()
    )

    effects = []
    for keys, sub in runs_df.groupby(["model", "agent", "sweep"]):
        model, agent, sweep = keys
        lo = sub[sub["parameter_value"] == sub["parameter_value"].min()]
        hi = sub[sub["parameter_value"] == sub["parameter_value"].max()]
        effects.append(
            {
                "model": model,
                "agent": agent,
                "sweep": sweep,
                "endpoint_effect_movement_high_minus_low": (
                    hi["movement_toward_counter"].mean()
                    - lo["movement_toward_counter"].mean()
                ),
                "endpoint_effect_final_stance_high_minus_low": (
                    hi["final_stance"].mean() - lo["final_stance"].mean()
                ),
            }
        )
    return grouped.merge(
        runs_df[["model", "model_label", "agent", "agent_label", "sweep", "sweep_label"]]
        .drop_duplicates()
        .merge(runs_df.__class__(effects), on=["model", "agent", "sweep"], how="left"),
        on=["model", "model_label", "agent", "agent_label", "sweep", "sweep_label"],
        how="left",
    )


def style_axes(ax, text_color: str, grid_color: str, spine_color: str) -> None:
    ax.axhspan(-0.1, 0.1, color="#f3f4f6", alpha=1.0, linewidth=0, zorder=0)
    ax.axhline(0, color="#9CA3AF", linestyle="--", linewidth=0.9, alpha=0.85, zorder=1)
    ax.grid(axis="y", color=grid_color, linewidth=0.75, alpha=0.78)
    ax.grid(axis="x", color=grid_color, linewidth=0.55, alpha=0.45)
    ax.set_axisbelow(True)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(spine_color)
        ax.spines[spine].set_linewidth(0.8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="both", colors=text_color, width=0.8, labelsize=11.0)
    ax.set_ylim(-1.08, 1.08)
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])


def plot_agent_figure(runs_df, output_dir: Path, agent: str) -> None:
    plt, np, _pd = import_plotting()
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
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 13.0,
            "axes.titlesize": 15.0,
            "xtick.labelsize": 12.0,
            "ytick.labelsize": 12.0,
            "legend.fontsize": 11.5,
            "font.size": 12.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    text_color = "#293241"
    grid_color = "#e5e7eb"
    spine_color = "#d1d5db"
    agent_df = runs_df[runs_df["agent"] == agent].copy()
    agent_label = AGENT_LABELS.get(agent, agent)

    fig, axes = plt.subplots(1, 2, figsize=(9.7, 3.95), sharey=True, constrained_layout=False)
    for ax, sweep in zip(axes, ["Update", "Anchor"]):
        style_axes(ax, text_color, grid_color, spine_color)
        sub = agent_df[agent_df["sweep"] == sweep]
        endpoint_labels = []
        for model in ["GPT-4o-mini", "QWEN", "GEMMA4"]:
            model_df = sub[sub["model"] == model]
            if model_df.empty:
                continue
            summary = (
                model_df.groupby("parameter_value")["final_stance"]
                .agg(["mean", "std", "count"])
                .reset_index()
                .sort_values("parameter_value")
            )
            summary["std"] = summary["std"].fillna(0.0)
            summary["se"] = summary["std"] / np.sqrt(summary["count"].clip(lower=1))
            x = summary["parameter_value"].to_numpy(dtype=float)
            y = summary["mean"].to_numpy(dtype=float)
            se = summary["se"].to_numpy(dtype=float)
            color = MODEL_COLORS.get(model, "#4b5563")
            for param, param_df in model_df.groupby("parameter_value"):
                jitter = np.linspace(-0.012, 0.012, len(param_df))
                ax.scatter(
                    np.full(len(param_df), float(param)) + jitter,
                    param_df.sort_values("trial")["final_stance"],
                    s=18,
                    color=color,
                    alpha=0.22,
                    linewidth=0,
                    zorder=2,
                )
            ax.fill_between(x, y - se, y + se, color=color, alpha=0.12, linewidth=0, zorder=1)
            ax.plot(
                x,
                y,
                color=color,
                linewidth=2.55,
                marker="o",
                markersize=5.4,
                solid_capstyle="round",
                label=MODEL_LABELS.get(model, model),
                zorder=3,
            )
            endpoint_labels.append((y[-1], x[-1], model, color))

        endpoint_labels.sort(key=lambda item: item[0])
        label_offsets = [0.0 for _ in endpoint_labels]
        for idx in range(1, len(endpoint_labels)):
            previous_y = endpoint_labels[idx - 1][0] + label_offsets[idx - 1]
            current_y = endpoint_labels[idx][0] + label_offsets[idx]
            if current_y - previous_y < 0.105:
                label_offsets[idx] += 0.105 - (current_y - previous_y)
        for idx in range(len(endpoint_labels) - 2, -1, -1):
            next_y = endpoint_labels[idx + 1][0] + label_offsets[idx + 1]
            current_y = endpoint_labels[idx][0] + label_offsets[idx]
            if next_y - current_y < 0.105:
                label_offsets[idx] -= 0.105 - (next_y - current_y)
        for (y_last, x_last, model, color), offset in zip(endpoint_labels, label_offsets):
            label_y = max(-1.04, min(1.04, y_last + offset))
            ax.annotate(
                f"{y_last:+.2f}",
                xy=(x_last, y_last),
                xycoords="data",
                xytext=(1.018, label_y),
                textcoords=("data", "data"),
                arrowprops=(
                    {
                        "arrowstyle": "-",
                        "color": color,
                        "linewidth": 0.9,
                        "alpha": 0.6,
                        "shrinkA": 0,
                        "shrinkB": 4,
                    }
                    if abs(label_y - y_last) > 0.035
                    else None
                ),
                annotation_clip=False,
                clip_on=False,
                va="center",
                ha="left",
                color=color,
                fontsize=10.3,
                fontweight=750,
                bbox={
                    "boxstyle": "round,pad=0.12",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.86,
                },
                zorder=4,
            )

        ax.set_xlim(0.15, 1.08)
        ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xlabel(SWEEP_LABELS[sweep], fontsize=13.0, fontweight=700, color=text_color)
        ax.set_title(
            "Evidence responsiveness sweep" if sweep == "Update" else "Prior anchoring sweep",
            fontsize=15.0,
            fontweight=750,
            color=text_color,
            pad=8,
        )
    axes[0].set_ylabel("Final stance", fontsize=13.0, fontweight=700, color=text_color)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
        handlelength=1.8,
        prop={"weight": 650, "size": 11.5},
    )
    fig.subplots_adjust(left=0.08, right=0.975, bottom=0.265, top=0.86, wspace=0.12)
    stem = f"figure_model_sensitivity_{agent_label.lower().replace(' ', '_')}"
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.06)
    fig.savefig(output_dir / f"{stem}.png", dpi=240, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def plot_agent_trajectory_grid(rounds_df, output_dir: Path, agent: str) -> None:
    plt, np, _pd = import_plotting()
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
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 15.4,
            "axes.titlesize": 18.6,
            "xtick.labelsize": 13.6,
            "ytick.labelsize": 13.6,
            "legend.fontsize": 13.4,
            "font.size": 13.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    text_color = "#293241"
    grid_color = "#e5e7eb"
    spine_color = "#d1d5db"
    agent_df = rounds_df[rounds_df["agent"] == agent].copy()
    agent_label = AGENT_LABELS.get(agent, agent)
    models = ["GPT-4o-mini", "QWEN", "GEMMA4"]
    sweeps = ["Update", "Anchor"]
    parameter_values = [0.2, 0.4, 0.6, 0.8, 1.0]
    parameter_colors = PARAMETER_COLORS
    label_min_y = -0.98
    label_max_y = 1.02
    endpoint_label_gap = 0.20

    def _spread_label_positions(values: list[float]) -> list[float]:
        if not values:
            return []
        gap = min(endpoint_label_gap, (label_max_y - label_min_y) / max(1, len(values) - 1))
        positions = [max(label_min_y, min(label_max_y, value)) for value in values]
        for idx in range(1, len(positions)):
            positions[idx] = max(positions[idx], positions[idx - 1] + gap)
        if positions[-1] > label_max_y:
            overflow = positions[-1] - label_max_y
            positions = [position - overflow for position in positions]
        for idx in range(len(positions) - 2, -1, -1):
            positions[idx] = min(positions[idx], positions[idx + 1] - gap)
        if positions[0] < label_min_y:
            underflow = label_min_y - positions[0]
            positions = [position + underflow for position in positions]
        return positions

    fig, axes = plt.subplots(
        len(models),
        len(sweeps),
        figsize=(7.65, 8.55),
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )

    for row_idx, model in enumerate(models):
        for col_idx, sweep in enumerate(sweeps):
            ax = axes[row_idx, col_idx]
            style_axes(ax, text_color, grid_color, spine_color)
            ax.tick_params(axis="both", labelsize=13.6)
            ax.set_xlim(-0.3, 16.35)
            ax.set_xticks([0, 5, 10, 15])
            sub = agent_df[(agent_df["sweep"] == sweep) & (agent_df["model"] == model)]
            endpoint_labels = []
            for parameter_value in parameter_values:
                param_df = sub[sub["parameter_value"] == parameter_value]
                if param_df.empty:
                    continue
                for _trial, trial_df in param_df.groupby("trial"):
                    trial_df = trial_df.sort_values("round")
                    ax.plot(
                        trial_df["round"],
                        trial_df["subject_stance"],
                        color=parameter_colors[parameter_value],
                        linewidth=1.05,
                        alpha=0.24,
                        zorder=2,
                    )
                summary = (
                    param_df.groupby("round")["subject_stance"]
                    .agg(["mean", "std", "count"])
                    .reset_index()
                    .sort_values("round")
                )
                summary["std"] = summary["std"].fillna(0.0)
                summary["se"] = summary["std"] / np.sqrt(summary["count"].clip(lower=1))
                rounds = summary["round"].to_numpy(dtype=float)
                mean = summary["mean"].to_numpy(dtype=float)
                se = summary["se"].to_numpy(dtype=float)
                color = parameter_colors[parameter_value]
                ax.fill_between(
                    rounds,
                    mean - se,
                    mean + se,
                    color=color,
                    alpha=0.08,
                    linewidth=0,
                    zorder=1,
                )
                ax.plot(
                    rounds,
                    mean,
                    color=color,
                    linewidth=2.85,
                    solid_capstyle="round",
                    label=f"{parameter_value:.1f}",
                    zorder=3,
                )
                endpoint_labels.append((mean[-1], rounds[-1], parameter_value, color))

            endpoint_labels.sort(key=lambda item: item[0])
            label_positions = _spread_label_positions(
                [stance_end for stance_end, _round_end, _parameter_value, _color in endpoint_labels]
            )
            for (stance_end, round_end, _parameter_value, color), label_y in zip(
                endpoint_labels, label_positions
            ):
                ax.annotate(
                    f"{stance_end:+.2f}",
                    xy=(round_end, stance_end),
                    xycoords="data",
                    xytext=(15.35, label_y),
                    textcoords=("data", "data"),
                    arrowprops=(
                        {
                            "arrowstyle": "-",
                            "color": color,
                            "linewidth": 0.8,
                            "alpha": 0.55,
                            "shrinkA": 0,
                            "shrinkB": 4,
                        }
                        if abs(label_y - stance_end) > 0.035
                        else None
                    ),
                    annotation_clip=False,
                    clip_on=False,
                    va="center",
                    ha="left",
                    color=color,
                    fontsize=11.3,
                    fontweight=750,
                    bbox={
                        "boxstyle": "round,pad=0.08",
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.86,
                    },
                    zorder=4,
                )

            if row_idx == 0:
                ax.set_title(
                    r"Evidence uptake $u$" if sweep == "Update" else r"Prior anchoring $a$",
                    fontsize=18.6,
                    fontweight=780,
                    color=text_color,
                    pad=8,
                )
            if col_idx == 0:
                ax.text(
                    -0.21,
                    0.5,
                    MODEL_LABELS.get(model, model),
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=16.4,
                    fontweight=780,
                    color=text_color,
                    clip_on=False,
                )
            else:
                ax.tick_params(axis="y", left=False, labelleft=False)
            if row_idx == len(models) - 1:
                ax.set_xlabel("Round", fontsize=15.4, fontweight=720, color=text_color)
            else:
                ax.tick_params(axis="x", bottom=False, labelbottom=False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            title="parameter value",
            loc="lower center",
            ncol=5,
            frameon=False,
            bbox_to_anchor=(0.56, 0.01),
            handlelength=1.8,
            columnspacing=1.0,
            prop={"weight": 650, "size": 13.5},
            title_fontproperties={"weight": 720, "size": 13.5},
    )
    fig.text(
        0.02,
        0.52,
        "Subject stance",
        rotation=90,
        ha="center",
        va="center",
        fontsize=16.0,
        fontweight=730,
        color=text_color,
    )
    fig.subplots_adjust(left=0.155, right=0.975, bottom=0.13, top=0.935, wspace=0.12, hspace=0.08)
    stem = f"figure_model_trajectory_sweep_{agent_label.lower().replace(' ', '_')}"
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.045)
    fig.savefig(output_dir / f"{stem}.png", dpi=240, bbox_inches="tight", pad_inches=0.045)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-dir", type=Path, default=DEFAULT_LOGS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    os.environ.setdefault("MPLCONFIGDIR", str(args.output_dir / ".mplconfig"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    _plt, _np, pd = import_plotting()
    run_rows, round_rows = collect_rows(args.logs_dir)
    if not run_rows:
        raise SystemExit(f"No model sensitivity results found under {args.logs_dir}")

    runs_df = pd.DataFrame(run_rows).sort_values(
        ["agent", "sweep", "parameter_value", "model", "trial"]
    )
    rounds_df = pd.DataFrame(round_rows).sort_values(
        ["agent", "sweep", "parameter_value", "model", "trial", "round"]
    )
    metrics_df = summarize_metrics(runs_df)

    runs_df.to_csv(args.output_dir / "model_sensitivity_runs.csv", index=False)
    rounds_df.to_csv(args.output_dir / "model_sensitivity_rounds.csv", index=False)
    metrics_df.to_csv(args.output_dir / "model_sensitivity_metrics.csv", index=False)

    for agent in ["CoreBot", "OpenClaw"]:
        plot_agent_figure(runs_df, args.output_dir, agent)
        plot_agent_trajectory_grid(rounds_df, args.output_dir, agent)

    print(f"Wrote {len(runs_df)} runs and figures to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
