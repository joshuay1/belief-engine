#!/usr/bin/env python
"""Build 6-panel topic grids and an all-topic metrics table.

The 6-panel figure is intended as a main-paper candidate:
four Belief Engine model panels plus the two strongest fair baselines.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONDITION_ORDER = [
    "be_gpt4omini",
    "be_gpt54mini",
    "be_qwen",
    "be_gemma",
    "prompt_self_update_gpt54mini",
    "rag_self_update_gpt54mini",
]

PANEL_TITLES = {
    "be_gpt4omini": "Belief Engine\nGPT-4o-mini",
    "be_gpt54mini": "Belief Engine\nGPT-5.4-mini",
    "be_qwen": "Belief Engine\nQwen 3.5 9B",
    "be_gemma": "Belief Engine\nGemma4 E4b",
    "prompt_self_update_gpt54mini": "Prompt\nself-update",
    "rag_self_update_gpt54mini": "RAG +\nself-update",
}

TABLE_LABELS = {
    "be_gpt4omini": "BE GPT-4o-mini",
    "be_gpt54mini": "BE GPT-5.4-mini",
    "be_qwen": "BE Qwen 3.5 9B",
    "be_gemma": "BE Gemma4 E4b",
    "prompt_self_update_gpt54mini": "Prompt self-update",
    "rag_self_update_gpt54mini": "RAG + self-update",
}

MODEL_LABELS = {
    "be_gpt4omini": "gpt-4o-mini",
    "be_gpt54mini": "gpt-5.4-mini",
    "be_qwen": "qwen3.5:9b",
    "be_gemma": "gemma4:latest",
    "prompt_self_update_gpt54mini": "gpt-5.4-mini",
    "rag_self_update_gpt54mini": "gpt-5.4-mini",
}

TOPIC_ORDER = [
    "social_media",
    "journalism",
    "compulsory_voting",
    "atheism",
    "libertarianism",
]

INITIAL_PRO = 0.75
INITIAL_CON = -0.75
INITIAL_GAP = INITIAL_PRO - INITIAL_CON


def normalize_trajectories(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["side"] = df["side"].astype(str).str.capitalize()
    df["agent"] = df["agent"].astype(str).str.capitalize()
    df["condition"] = df["condition"].astype(str)
    df = df[df["condition"].isin(CONDITION_ORDER)].copy()
    df["condition"] = pd.Categorical(df["condition"], categories=CONDITION_ORDER, ordered=True)
    df.sort_values(["condition", "trial", "side", "round"], inplace=True)
    df["condition"] = df["condition"].astype(str)
    return df


def apply_plot_style() -> None:
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
            "axes.labelsize": 11.6,
            "axes.titlesize": 15.0,
            "xtick.labelsize": 11.3,
            "ytick.labelsize": 11.3,
            "legend.fontsize": 11.0,
            "font.size": 11.3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_six_panel(df: pd.DataFrame, topic: str, output_stem: Path) -> None:
    apply_plot_style()
    fig, axes = plt.subplots(3, 2, figsize=(6.25, 8.15), sharex=True, sharey=True)
    axes = axes.ravel()

    pro_color = "#19747e"
    con_color = "#b34a4a"
    text_color = "#293241"
    grid_color = "#e5e7eb"
    spine_color = "#d1d5db"
    max_round = int(df["round"].max())

    for ax, condition in zip(axes, CONDITION_ORDER):
        sub = df[df["condition"] == condition]
        ax.set_title(
            PANEL_TITLES[condition],
            fontsize=13.7,
            fontweight=750,
            color=text_color,
            pad=3,
            linespacing=0.86,
        )
        ax.axhspan(-0.1, 0.1, color="#f3f4f6", alpha=1.0, linewidth=0, zorder=0)
        ax.axhline(0, color="#9CA3AF", linestyle="--", linewidth=0.9, alpha=0.8, zorder=1)
        ax.set_ylim(-1.08, 1.08)
        ax.set_xlim(-0.4, max_round + 0.6)
        ax.set_xticks([0, 5, 10, 15])
        ax.grid(axis="y", color=grid_color, linewidth=0.75, alpha=0.78)
        ax.grid(axis="x", color=grid_color, linewidth=0.55, alpha=0.45)
        ax.set_axisbelow(True)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color(spine_color)
            ax.spines[spine].set_linewidth(0.8)

        for side, color in [("Pro", pro_color), ("Con", con_color)]:
            side_df = sub[sub["side"] == side].dropna(subset=["stance"])
            for _, trial_df in side_df.groupby("run_id"):
                trial_df = trial_df.sort_values("round")
                ax.plot(
                    trial_df["round"],
                    trial_df["stance"],
                    color=color,
                    alpha=0.34,
                    linewidth=1.0,
                    solid_capstyle="round",
                    zorder=2,
                )
            stats = (
                side_df.groupby("round")["stance"]
                .agg(["mean", "std"])
                .reset_index()
                .sort_values("round")
            )
            x = stats["round"].to_numpy(dtype=float)
            mean = stats["mean"].to_numpy(dtype=float)
            std = stats["std"].fillna(0.0).to_numpy(dtype=float)
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.10, linewidth=0, zorder=1)
            ax.plot(
                x,
                mean,
                color=color,
                linewidth=2.45,
                solid_capstyle="round",
                label=f"{side.upper()} mean",
                zorder=3,
            )
            if len(x) > 0:
                ax.annotate(
                    f"{mean[0]:+.2f}",
                    xy=(x[0], mean[0]),
                    xytext=(4, 5 if side == "Pro" else -10),
                    textcoords="offset points",
                    ha="left",
                    color=color,
                    fontsize=10.2,
                    fontweight=750,
                    bbox={
                        "boxstyle": "round,pad=0.12",
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.86,
                    },
                )
                ax.annotate(
                    f"{mean[-1]:+.2f}",
                    xy=(x[-1], mean[-1]),
                    xytext=(-4, 4 if side == "Pro" else -11),
                    textcoords="offset points",
                    ha="right",
                    color=color,
                    fontsize=10.2,
                    fontweight=750,
                    bbox={
                        "boxstyle": "round,pad=0.12",
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.86,
                    },
                )

        if ax.get_subplotspec().is_last_row():
            ax.set_xlabel("Round", fontsize=11.6, fontweight=700, color=text_color)
        ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        ax.tick_params(axis="both", colors=text_color, width=0.8)
        trial_count = sub["trial"].nunique()
        ax.text(
            0.965,
            0.055,
            f"n={trial_count}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9.8,
            fontweight=650,
            color="#6b7280",
            bbox={
                "boxstyle": "round,pad=0.16",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.78,
            },
            zorder=5,
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.004),
        ncol=2,
        frameon=False,
        handlelength=1.8,
        prop={"weight": 650, "size": 11.0},
    )
    fig.suptitle(
        f"Topic: {topic}\nBE UA: $(u,a)=(0.2,0.4)$ | Baseline models: GPT-5.4-mini",
        fontsize=12.7,
        fontweight=700,
        color="#6b7280",
        y=0.988,
        linespacing=0.92,
    )
    fig.supylabel(
        "Stance (BE internal; baselines judged)",
        fontsize=14.0,
        fontweight=750,
        color=text_color,
        x=0.045,
    )
    fig.subplots_adjust(left=0.135, right=0.985, bottom=0.105, top=0.855, wspace=0.13, hspace=0.33)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def topic_short_label(topic: str) -> str:
    replacements = {
        "Social media brings more harm than good": "Social media",
        "We should subsidize journalism": "Journalism",
        "We should introduce compulsory voting": "Compulsory voting",
        "We should adopt atheism": "Atheism",
        "We should adopt libertarianism": "Libertarianism",
    }
    return replacements.get(topic, topic)


def compute_metrics(df: pd.DataFrame, topic_slug: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    topic = str(df["topic"].dropna().iloc[0])
    for condition in CONDITION_ORDER:
        sub = df[df["condition"] == condition]
        trial_rows = []
        for trial, trial_df in sub.groupby("trial"):
            final_round = int(trial_df["round"].max())
            final = trial_df[trial_df["round"] == final_round]
            initial = trial_df[trial_df["round"] == int(trial_df["round"].min())]
            try:
                pro_initial = float(initial[initial["side"] == "Pro"]["stance"].iloc[0])
                con_initial = float(initial[initial["side"] == "Con"]["stance"].iloc[0])
                pro_final = float(final[final["side"] == "Pro"]["stance"].iloc[0])
                con_final = float(final[final["side"] == "Con"]["stance"].iloc[0])
            except IndexError:
                continue
            final_gap = pro_final - con_final
            abs_final_gap = abs(final_gap)
            abs_initial_gap = abs(pro_initial - con_initial)
            pro_shift = pro_final - pro_initial
            con_shift = con_final - con_initial
            trial_rows.append(
                {
                    "trial": int(trial),
                    "final_pro": pro_final,
                    "final_con": con_final,
                    "final_gap": final_gap,
                    "abs_final_gap": abs_final_gap,
                    "gap_reduction": abs_initial_gap - abs_final_gap,
                    "pro_shift": pro_shift,
                    "con_shift": con_shift,
                    "mean_abs_shift": (abs(pro_shift) + abs(con_shift)) / 2.0,
                    "center_shift": ((pro_initial - pro_final) + (con_final - con_initial)) / 2.0,
                    "final_midpoint": (pro_final + con_final) / 2.0,
                    "crossed": pro_final < con_final,
                }
            )
        metric_df = pd.DataFrame(trial_rows)
        if metric_df.empty:
            continue
        rows.append(
            {
                "topic_slug": topic_slug,
                "topic": topic,
                "topic_short": topic_short_label(topic),
                "condition": condition,
                "setup": TABLE_LABELS[condition],
                "model": MODEL_LABELS[condition],
                "stance_measure": (
                    "BE internal" if condition.startswith("be_") else "External judge"
                ),
                "n": int(metric_df["trial"].nunique()),
                "final_pro_mean": metric_df["final_pro"].mean(),
                "final_pro_sd": metric_df["final_pro"].std(ddof=1),
                "final_con_mean": metric_df["final_con"].mean(),
                "final_con_sd": metric_df["final_con"].std(ddof=1),
                "final_gap_mean": metric_df["final_gap"].mean(),
                "abs_final_gap_mean": metric_df["abs_final_gap"].mean(),
                "gap_reduction_mean": metric_df["gap_reduction"].mean(),
                "mean_abs_shift": metric_df["mean_abs_shift"].mean(),
                "center_shift_mean": metric_df["center_shift"].mean(),
                "final_midpoint_mean": metric_df["final_midpoint"].mean(),
                "crossing_rate": metric_df["crossed"].mean(),
            }
        )
    return rows


def format_metric_table(metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "topic_short",
        "setup",
        "stance_measure",
        "n",
        "final_pro_mean",
        "final_con_mean",
        "abs_final_gap_mean",
        "gap_reduction_mean",
        "mean_abs_shift",
        "center_shift_mean",
        "crossing_rate",
    ]
    formatted = metrics[columns].copy()
    rename = {
        "topic_short": "Topic",
        "setup": "Setup",
        "stance_measure": "Measure",
        "n": "n",
        "final_pro_mean": "Final Pro",
        "final_con_mean": "Final Con",
        "abs_final_gap_mean": "|Final gap|",
        "gap_reduction_mean": "Gap reduction",
        "mean_abs_shift": "Mean abs. shift",
        "center_shift_mean": "Center shift",
        "crossing_rate": "Crossing rate",
    }
    formatted.rename(columns=rename, inplace=True)
    for column in [
        "Final Pro",
        "Final Con",
        "|Final gap|",
        "Gap reduction",
        "Mean abs. shift",
        "Center shift",
        "Crossing rate",
    ]:
        formatted[column] = formatted[column].map(lambda value: f"{value:.2f}")
    return formatted


def write_metrics(metrics: pd.DataFrame, metrics_dir: Path) -> None:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = metrics_dir / "all_topics_6panel_metrics.csv"
    metrics.to_csv(metrics_csv, index=False)

    formatted = format_metric_table(metrics)
    (metrics_dir / "all_topics_6panel_metrics.md").write_text(
        "# All Topics 6-Panel Metrics\n\n"
        "Rows summarize the six main-paper candidate setups for each topic. "
        "`Gap reduction` is the reduction in absolute Pro-Con separation from "
        "the fixed initial gap of 1.50; positive values indicate convergence. "
        "`Center shift` is positive when both sides move toward the neutral center "
        "on average. BE rows use internal BE stance; baseline rows use the external judge.\n\n"
        + formatted.to_markdown(index=False)
        + "\n"
    )
    latex = formatted.to_latex(index=False, escape=True)
    (metrics_dir / "all_topics_6panel_metrics.tex").write_text(latex)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--all-topics-dir",
        type=Path,
        default=Path("validation/results/all_topics_u0.2_a0.4_8models"),
    )
    args = parser.parse_args()

    all_metrics: list[dict[str, object]] = []
    for topic_slug in TOPIC_ORDER:
        topic_dir = args.all_topics_dir / topic_slug
        csv_path = topic_dir / "combined_8model_topic_grid_stance_trajectories_gpt54mini_baselines.csv"
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        df = normalize_trajectories(pd.read_csv(csv_path))
        topic = str(df["topic"].dropna().iloc[0])
        output_stem = (
            topic_dir
            / "figures"
            / "topic_grids_6panel"
            / f"{topic_slug}_6panel_gpt54mini_baselines_grid_3x2"
        )
        plot_six_panel(df, topic, output_stem)
        all_metrics.extend(compute_metrics(df, topic_slug))
        print(f"Wrote {output_stem.with_suffix('.pdf')}")

    metrics = pd.DataFrame(all_metrics)
    metrics["topic_slug"] = pd.Categorical(metrics["topic_slug"], categories=TOPIC_ORDER, ordered=True)
    metrics["condition"] = pd.Categorical(metrics["condition"], categories=CONDITION_ORDER, ordered=True)
    metrics.sort_values(["topic_slug", "condition"], inplace=True)
    metrics["topic_slug"] = metrics["topic_slug"].astype(str)
    metrics["condition"] = metrics["condition"].astype(str)
    write_metrics(metrics, args.all_topics_dir / "metrics")
    print(f"Wrote {args.all_topics_dir / 'metrics'}")


if __name__ == "__main__":
    main()
