"""
Held-out parameter recommendation run for Belief Engine presets.

This script turns the exploratory sensitivity sweeps into a reproducible
recommendation protocol:

1. Classify participants using pre-debate information only.
2. Select update_s and anchor_s on train folds.
3. Evaluate global, participant-type, and topic-context presets on held-out
   folds.

The deployment-friendly model used here treats anchor_s as a multiplier on the
initial log-odds prior and update_s as the sensitivity applied to directed
debate evidence. This produces presets that can be used directly when an agent
has an initial stance and then receives debate messages.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/review_artifact_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/review_artifact_cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.constants import POLARITY_AFFIRMATIVE, POLARITY_NEGATIVE
from src.memory.argument_record import ArgumentRecord
from src.memory.judgement_strategies import VectorJudgementStrategy
from validation.data.load_debate import load_debate_dataset
from validation.data.preprocess import ParticipantTimeline, build_participant_timelines
from validation.data.scale_mapping import log_odds_to_stance, stance_to_log_odds
from validation.evaluation.common import (
    TimelineRecord,
    build_records,
    filter_records,
    get_cached_args,
    group_dynamics,
    make_folds,
    metric_dict,
    prediction_rows_for_model,
    summarize_prediction_rows,
)


DEFAULT_GRID = [0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4]
EVIDENCE_PRESSURE_THRESHOLD = 1.0

PRESET_LABELS = {
    "all": "General default",
    "evidence_all": "All participants",
    "evidence_aligned_mover": "Evidence-aligned movers",
    "evidence_opposed_mover": "Evidence-opposed movers",
    "weak_signal_mover": "Weak-signal movers",
    "stable_despite_pressure": "Stable despite pressure",
    "stable_low_pressure": "Stable low-pressure",
    "open": "Open / conflicted",
    "leaning": "Leaning",
    "committed": "Committed",
    "stable": "Stable",
    "softened": "Softened",
    "reinforced": "Reinforced",
    "boundary_switcher": "Boundary switchers",
    "strong_switcher": "Strong switchers",
    "breadth": "Broad social/political",
    "depth": "Technical/factual",
}

RECOMMENDED_USE = {
    "all": "Default when no pre-debate type is available.",
    "evidence_all": "Diagnostic evidence-alignment reference panel for all participants.",
    "evidence_aligned_mover": "Diagnostic group for movers whose accepted evidence pressure points in their movement direction.",
    "evidence_opposed_mover": "Diagnostic group for movers whose accepted evidence pressure points opposite their movement.",
    "weak_signal_mover": "Diagnostic group for movers with weak or mixed accepted evidence pressure.",
    "stable_despite_pressure": "Diagnostic group for stable participants despite non-trivial accepted evidence pressure.",
    "stable_low_pressure": "Diagnostic group for stable participants with weak or mixed accepted evidence pressure.",
    "open": "Use for agents with weak, neutral, or internally conflicted priors.",
    "leaning": "Use for agents with moderate but consistent priors.",
    "committed": "Use for agents with strong consistent priors and high inertia.",
    "stable": "Diagnostic group for participants with no Likert movement.",
    "softened": "Diagnostic group for participants who moved toward the agree/disagree boundary without crossing it.",
    "reinforced": "Diagnostic group for participants who moved further toward their initial side.",
    "boundary_switcher": "Diagnostic group for participants who crossed the agree/disagree boundary by one Likert step.",
    "strong_switcher": "Diagnostic group for participants who crossed the agree/disagree boundary by more than one Likert step.",
    "breadth": "Use for broad social or political propositions.",
    "depth": "Use for more technical or factual propositions.",
}


def _parse_grid(value: str | None) -> list[float]:
    if not value:
        return list(DEFAULT_GRID)
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def _load_json_if_exists(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _load_excluded_ids(path: str | None) -> set[str]:
    if not path or not os.path.exists(path):
        return set()
    return set(pd.read_csv(path)["worker_id"].astype(str).tolist())


def _iter_update_texts(timeline: ParticipantTimeline, exposure_mode: str) -> Iterable[tuple[str, str]]:
    for rd in timeline.rounds:
        for text in rd.tweets_received:
            yield text, "opponent"
        for text in rd.messages_received:
            yield text, "opponent"
        if exposure_mode == "all":
            if rd.tweet_sent:
                yield rd.tweet_sent, "self"
            for text in rd.messages_sent:
                yield text, "self"


def _strength_from_arg(arg: dict, strength_source: str) -> float:
    if strength_source == "classifier":
        return float(arg.get("support_strength_classifier", arg.get("support_strength", 0.0)))
    if strength_source == "llm":
        return float(arg.get("support_strength_llm", arg.get("support_strength", 0.0)))
    return float(arg.get("support_strength", 0.0))


def _update_args(
    timeline: ParticipantTimeline,
    extraction_cache: dict,
    reclass_cache: dict | None,
    exposure_mode: str,
    uptake_mode: str,
    strength_source: str,
    argument_similarity_threshold: float,
    self_similarity_threshold: float,
    judgement: VectorJudgementStrategy | None = None,
) -> list[tuple[float, int]]:
    args: list[tuple[float, int]] = []
    active_arguments: dict[str, ArgumentRecord] = {}
    if uptake_mode == "human_like":
        judgement = judgement or VectorJudgementStrategy(
            similarity_threshold=argument_similarity_threshold,
            self_similarity_threshold=self_similarity_threshold,
            cross_polarity_dedup=False,
        )

    for text, source_role in _iter_update_texts(timeline, exposure_mode):
        for arg in get_cached_args(text, timeline.topic, extraction_cache, reclass_cache):
            polarity = arg.get("polarity")
            if polarity == POLARITY_AFFIRMATIVE:
                sign = 1
            elif polarity == POLARITY_NEGATIVE:
                sign = -1
            else:
                continue
            strength = _strength_from_arg(arg, strength_source)
            if judgement is not None:
                record = ArgumentRecord(
                    topic_id=arg.get("topic_id", timeline.topic),
                    claim=arg.get("claim", ""),
                    polarity=polarity,
                    source_role=source_role,
                    support_strength=strength,
                    support_strength_llm=float(arg.get("support_strength_llm", arg.get("support_strength", 0.0))),
                    support_strength_classifier=float(arg.get("support_strength_classifier", arg.get("support_strength", 0.0))),
                    evidence_snippets=arg.get("evidence_snippets", []),
                    category_tags=arg.get("category_tags", []),
                )
                status, _target_id, _reason, processed = judgement.process(record, active_arguments)
                active_arguments[processed.id] = processed
                if status != "added" or not getattr(processed, "credence_relevant", False):
                    continue
            args.append((strength, sign))
    return args


def _prepare_update_args(
    records: list[TimelineRecord],
    extraction_cache: dict,
    reclass_cache: dict | None,
    exposure_mode: str,
    uptake_mode: str,
    strength_source: str,
    argument_similarity_threshold: float,
    self_similarity_threshold: float,
) -> dict[str, list[tuple[float, int]]]:
    judgement = None
    if uptake_mode == "human_like":
        judgement = VectorJudgementStrategy(
            similarity_threshold=argument_similarity_threshold,
            self_similarity_threshold=self_similarity_threshold,
            cross_polarity_dedup=False,
        )
    return {
        str(record.timeline.worker_id): _update_args(
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
        for record in records
    }


def _predict_change(
    record: TimelineRecord,
    update_args_by_worker: dict[str, list[tuple[float, int]]],
    update_s: float,
    anchor_s: float,
) -> float:
    timeline = record.timeline
    log_odds = stance_to_log_odds(timeline.initial_stance) * anchor_s
    for strength, sign in update_args_by_worker.get(str(timeline.worker_id), []):
        log_odds += sign * math.log(1.0 + strength * update_s)
    pred_final = log_odds_to_stance(log_odds)
    return pred_final - timeline.initial_stance


def _candidate_rows(
    records: list[TimelineRecord],
    update_args_by_worker: dict[str, list[tuple[float, int]]],
    update_s: float,
    anchor_s: float,
    fold: int,
    model: str = "candidate",
    params: dict | None = None,
) -> list[dict]:
    params = params or {}
    return prediction_rows_for_model(
        records,
        fold,
        model,
        lambda record: _predict_change(
            record, update_args_by_worker, update_s=update_s, anchor_s=anchor_s
        ),
        params={"update_s": update_s, "anchor_s": anchor_s, **params},
    )


def _score_rows(rows: list[dict], objective: str) -> float:
    if not rows:
        return math.inf
    actual = [float(row["opinion_change_actual"] or 0.0) for row in rows]
    predicted = [float(row["opinion_change_predicted"]) for row in rows]
    metrics = metric_dict(actual, predicted)
    if objective == "mae":
        return metrics["mae"]
    if objective == "balanced":
        groups = group_dynamics(rows)
        if groups.empty:
            convergence_error = 0.0
            drift_error = 0.0
        else:
            convergence_error = abs(
                float(groups["predicted_convergence"].mean())
                - float(groups["actual_convergence"].mean())
            )
            drift_error = abs(float(groups["drift_error"].mean()))
        return metrics["rmse"] + convergence_error + drift_error
    return metrics["rmse"]


def _fit_params(
    records: list[TimelineRecord],
    update_args_by_worker: dict[str, list[tuple[float, int]]],
    update_grid: list[float],
    anchor_grid: list[float],
    objective: str,
) -> dict:
    if not records:
        return {
            "update_s": update_grid[0],
            "anchor_s": anchor_grid[0],
            "train_score": math.nan,
            "n_train": 0,
        }

    best = None
    for update_s in update_grid:
        for anchor_s in anchor_grid:
            rows = _candidate_rows(
                records,
                update_args_by_worker,
                update_s=update_s,
                anchor_s=anchor_s,
                fold=0,
            )
            score = _score_rows(rows, objective)
            candidate = {
                "update_s": update_s,
                "anchor_s": anchor_s,
                "train_score": score,
                "n_train": len(records),
            }
            if best is None or score < best["train_score"]:
                best = candidate
    return best or {
        "update_s": update_grid[0],
        "anchor_s": anchor_grid[0],
        "train_score": math.nan,
        "n_train": len(records),
    }


def _records_for_key(
    records: list[TimelineRecord],
    scheme: str,
    key: str,
    update_args_by_worker: dict[str, list[tuple[float, int]]] | None = None,
) -> list[TimelineRecord]:
    if scheme == "global":
        return records
    if scheme == "participant_type":
        return [record for record in records if record.participant_type == key]
    if scheme == "response_type":
        return [record for record in records if _response_type(record) == key]
    if scheme == "evidence_alignment":
        if key == "evidence_all":
            return records
        return [
            record
            for record in records
            if _evidence_alignment_type(record, update_args_by_worker or {}) == key
        ]
    if scheme == "topic_category":
        return [record for record in records if record.timeline.topic_category == key]
    raise ValueError(f"Unknown scheme: {scheme}")


def _key_for_record(
    record: TimelineRecord,
    scheme: str,
    update_args_by_worker: dict[str, list[tuple[float, int]]] | None = None,
) -> str:
    if scheme == "global":
        return "all"
    if scheme == "participant_type":
        return record.participant_type
    if scheme == "response_type":
        return _response_type(record)
    if scheme == "evidence_alignment":
        return _evidence_alignment_type(record, update_args_by_worker or {})
    if scheme == "topic_category":
        return str(record.timeline.topic_category)
    raise ValueError(f"Unknown scheme: {scheme}")


def _scheme_keys(train_records: list[TimelineRecord], scheme: str) -> list[str]:
    if scheme == "global":
        return ["all"]
    if scheme == "participant_type":
        return sorted({record.participant_type for record in train_records})
    if scheme == "response_type":
        return [
            "stable",
            "softened",
            "reinforced",
            "boundary_switcher",
            "strong_switcher",
        ]
    if scheme == "evidence_alignment":
        return [
            "evidence_all",
            "evidence_aligned_mover",
            "evidence_opposed_mover",
            "weak_signal_mover",
            "stable_despite_pressure",
            "stable_low_pressure",
        ]
    if scheme == "topic_category":
        return sorted({str(record.timeline.topic_category) for record in train_records})
    raise ValueError(f"Unknown scheme: {scheme}")


def _response_type(record: TimelineRecord) -> str:
    """Observed DEBATE response pattern on the original six-point scale."""
    initial = float(record.timeline.initial_opinion)
    final = record.timeline.final_opinion
    if final is None:
        return "stable"
    final = float(final)
    delta = final - initial
    if abs(delta) < 1.0:
        return "stable"
    initial_side = -1 if initial <= 3.0 else 1
    final_side = -1 if final <= 3.0 else 1
    if initial_side != final_side:
        return "boundary_switcher" if abs(delta) == 1.0 else "strong_switcher"

    initial_distance = abs(initial - 3.5)
    final_distance = abs(final - 3.5)
    if final_distance < initial_distance:
        return "softened"
    return "reinforced"


def _evidence_alignment_type(
    record: TimelineRecord,
    update_args_by_worker: dict[str, list[tuple[float, int]]],
) -> str:
    """Observed change grouped by alignment with accepted received evidence."""
    final = record.timeline.final_stance
    if final is None:
        return "stable_low_pressure"
    actual_change = float(final - record.timeline.initial_stance)
    pressure = sum(
        strength * sign
        for strength, sign in update_args_by_worker.get(str(record.timeline.worker_id), [])
    )
    is_stable = abs(float(record.timeline.final_opinion) - float(record.timeline.initial_opinion)) < 1.0
    weak_signal = abs(pressure) < EVIDENCE_PRESSURE_THRESHOLD
    if is_stable:
        return "stable_low_pressure" if weak_signal else "stable_despite_pressure"
    if weak_signal:
        return "weak_signal_mover"
    if actual_change * pressure > 0:
        return "evidence_aligned_mover"
    return "evidence_opposed_mover"


def _evaluate_scheme(
    scheme: str,
    fold_idx: int,
    train_records: list[TimelineRecord],
    test_records: list[TimelineRecord],
    update_args_by_worker: dict[str, list[tuple[float, int]]],
    update_grid: list[float],
    anchor_grid: list[float],
    objective: str,
    min_train: int,
) -> tuple[list[dict], list[dict]]:
    global_params = _fit_params(
        train_records,
        update_args_by_worker,
        update_grid,
        anchor_grid,
        objective,
    )

    selected: dict[str, dict] = {}
    calibration_rows: list[dict] = []
    for key in _scheme_keys(train_records, scheme):
        subset = _records_for_key(train_records, scheme, key, update_args_by_worker)
        if scheme != "global" and len(subset) < min_train:
            params = {**global_params, "fallback_to_global": True}
        else:
            params = {
                **_fit_params(
                    subset,
                    update_args_by_worker,
                    update_grid,
                    anchor_grid,
                    objective,
                ),
                "fallback_to_global": False,
            }
        selected[key] = params
        calibration_rows.append(
            {
                "fold": fold_idx,
                "scheme": scheme,
                "preset_key": key,
                "preset_label": PRESET_LABELS.get(key, key),
                "update_s": params["update_s"],
                "anchor_s": params["anchor_s"],
                "train_score": params["train_score"],
                "n_train": params["n_train"],
                "fallback_to_global": params["fallback_to_global"],
            }
        )

    prediction_rows: list[dict] = []
    for record in test_records:
        key = _key_for_record(record, scheme, update_args_by_worker)
        params = selected.get(key, global_params)
        prediction_rows.extend(
            _candidate_rows(
                [record],
                update_args_by_worker,
                update_s=params["update_s"],
                anchor_s=params["anchor_s"],
                fold=fold_idx,
                model=f"preset_{scheme}",
                params={
                    "scheme": scheme,
                    "preset_key": key,
                    "preset_label": PRESET_LABELS.get(key, key),
                    "fallback_to_global": params.get("fallback_to_global", False),
                },
            )
        )
    return prediction_rows, calibration_rows


def _prior_only_rows(records: list[TimelineRecord], fold_idx: int) -> list[dict]:
    return prediction_rows_for_model(
        records,
        fold_idx,
        "prior_only_no_change",
        lambda record: 0.0,
        params={
            "scheme": "baseline",
            "preset_key": "prior_only",
            "preset_label": "Prior only",
        },
    )


def _metric_summary(rows: list[dict]) -> pd.DataFrame:
    aggregate = summarize_prediction_rows(rows, ["model"])
    groups = group_dynamics(rows)
    if aggregate.empty:
        return aggregate
    conv = []
    if not groups.empty:
        for model, group in groups.groupby("model", dropna=False):
            conv.append(
                {
                    "model": model,
                    "actual_convergence": float(group["actual_convergence"].mean()),
                    "predicted_convergence": float(group["predicted_convergence"].mean()),
                    "convergence_error": abs(
                        float(group["predicted_convergence"].mean())
                        - float(group["actual_convergence"].mean())
                    ),
                    "drift_error": abs(float(group["drift_error"].mean())),
                }
            )
    conv_df = pd.DataFrame(conv)
    if not conv_df.empty:
        aggregate = aggregate.merge(conv_df, on="model", how="left")
    return aggregate


def _recommendations(
    individual_df: pd.DataFrame,
    calibration_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for (scheme, key), group in calibration_df.groupby(["scheme", "preset_key"], dropna=False):
        applied = individual_df[
            (individual_df["scheme"] == scheme)
            & (individual_df["preset_key"] == key)
        ].copy()
        metrics = {}
        if not applied.empty:
            actual = applied["opinion_change_actual"].astype(float).tolist()
            predicted = applied["opinion_change_predicted"].astype(float).tolist()
            metrics = metric_dict(actual, predicted)
        updates = group["update_s"].astype(float)
        anchors = group["anchor_s"].astype(float)
        update_mode = Counter(updates.tolist()).most_common(1)[0][0]
        anchor_mode = Counter(anchors.tolist()).most_common(1)[0][0]
        rows.append(
            {
                "scheme": scheme,
                "preset_key": key,
                "preset_label": PRESET_LABELS.get(str(key), str(key)),
                "recommended_update_s": update_mode,
                "recommended_anchor_s": anchor_mode,
                "median_update_s": float(updates.median()),
                "median_anchor_s": float(anchors.median()),
                "min_update_s": float(updates.min()),
                "max_update_s": float(updates.max()),
                "min_anchor_s": float(anchors.min()),
                "max_anchor_s": float(anchors.max()),
                "n_fold_selections": int(len(group)),
                "n_heldout": int(metrics.get("n", 0)),
                "heldout_rmse": metrics.get("rmse", math.nan),
                "heldout_mae": metrics.get("mae", math.nan),
                "heldout_pearson_r": metrics.get("pearson_r", math.nan),
                "heldout_direction_accuracy": metrics.get(
                    "direction_accuracy", math.nan
                ),
                "recommended_use": RECOMMENDED_USE.get(str(key), "Context-specific preset."),
            }
        )
    return pd.DataFrame(rows)


def _plot_parameter_presets(recs: pd.DataFrame, output_dir: str) -> None:
    plot_df = recs[recs["scheme"].isin(["participant_type", "topic_category"])].copy()
    if plot_df.empty:
        return
    colors = {
        "participant_type": "#2f6f9f",
        "topic_category": "#c96f2d",
    }
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for _, row in plot_df.iterrows():
        ax.scatter(
            row["recommended_update_s"],
            row["recommended_anchor_s"],
            s=120,
            color=colors.get(row["scheme"], "#4c566a"),
            edgecolor="white",
            linewidth=1.2,
            label=row["scheme"],
        )
        ax.annotate(
            row["preset_label"],
            (row["recommended_update_s"], row["recommended_anchor_s"]),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=9,
        )
    handles, labels = ax.get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    ax.legend(dedup.values(), dedup.keys(), frameon=False)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("recommended update_s")
    ax.set_ylabel("recommended anchor_s")
    ax.set_title("Held-out Belief Engine Parameter Presets")
    ax.grid(True, which="both", alpha=0.18)
    fig.tight_layout()
    for ext in ["pdf", "png", "svg"]:
        fig.savefig(os.path.join(output_dir, f"figure_parameter_presets.{ext}"), dpi=200)
    plt.close(fig)


def _plot_scheme_comparison(metrics: pd.DataFrame, output_dir: str) -> None:
    if metrics.empty:
        return
    model_order = [
        "prior_only_no_change",
        "preset_global",
        "preset_participant_type",
        "preset_topic_category",
    ]
    labels = {
        "prior_only_no_change": "Prior only",
        "preset_global": "Global preset",
        "preset_participant_type": "Type presets",
        "preset_topic_category": "Topic presets",
    }
    plot_df = metrics[metrics["model"].isin(model_order)].copy()
    plot_df["order"] = plot_df["model"].map({m: i for i, m in enumerate(model_order)})
    plot_df = plot_df.sort_values("order")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    x = np.arange(len(plot_df))
    axes[0].bar(x, plot_df["change_rmse"], color="#2f6f9f")
    axes[0].set_xticks(x, [labels[m] for m in plot_df["model"]], rotation=20, ha="right")
    axes[0].set_ylabel("held-out RMSE")
    axes[0].set_title("Individual Fit")
    if "convergence_error" in plot_df:
        axes[1].bar(x, plot_df["convergence_error"], color="#c96f2d")
    axes[1].set_xticks(x, [labels[m] for m in plot_df["model"]], rotation=20, ha="right")
    axes[1].set_ylabel("convergence error")
    axes[1].set_title("Collective Realism")
    for ax in axes:
        ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    for ext in ["pdf", "png", "svg"]:
        fig.savefig(os.path.join(output_dir, f"figure_global_vs_type_specific.{ext}"), dpi=200)
    plt.close(fig)


def _save_report(
    output_dir: str,
    summary: dict,
    metrics_df: pd.DataFrame,
    recs_df: pd.DataFrame,
) -> None:
    lines = [
        "# Parameter Recommendation Report",
        "",
        f"- Run ID: `{summary['run_id']}`",
        f"- Split: `{summary['split_by']}`",
        f"- Clean mode: `{summary['clean_mode']}`",
        f"- Exposure mode: `{summary['exposure_mode']}`",
        f"- Uptake mode: `{summary.get('uptake_mode', 'raw')}`",
        f"- Strength source: `{summary.get('strength_source', 'active')}`",
        f"- Participants: `{summary['n_participants']}`",
        f"- Folds: `{summary['n_folds']}`",
        f"- Objective: `{summary['objective']}`",
        "",
        "## Interpretation",
        "",
        "This run selects `update_s` and `anchor_s` on train folds only, then",
        "evaluates the selected presets on held-out participants. Participant",
        "types are assigned from pre-debate prior extremity and initial-text",
        "consistency, so type-specific presets are deployable before a debate",
        "starts.",
        "",
        "## Held-Out Scheme Metrics",
        "",
    ]
    if metrics_df.empty:
        lines.append("_No metrics produced._")
    else:
        cols = [
            "model",
            "change_n",
            "change_mae",
            "change_rmse",
            "change_pearson_r",
            "change_direction_accuracy",
            "convergence_error",
        ]
        lines.append(metrics_df[[c for c in cols if c in metrics_df]].to_markdown(index=False))
    lines.extend(["", "## Recommended Presets", ""])
    if recs_df.empty:
        lines.append("_No recommendations produced._")
    else:
        cols = [
            "scheme",
            "preset_label",
            "recommended_update_s",
            "recommended_anchor_s",
            "heldout_rmse",
            "heldout_pearson_r",
            "heldout_direction_accuracy",
            "recommended_use",
        ]
        lines.append(recs_df[[c for c in cols if c in recs_df]].to_markdown(index=False))
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `parameter_recommendation_summary.json`",
            "- `individual_results.csv`",
            "- `heldout_metrics.csv`",
            "- `stratified_metrics.csv`",
            "- `group_results.csv`",
            "- `fold_selected_params.csv`",
            "- `parameter_recommendations.csv`",
            "- `figure_parameter_presets.pdf`",
            "- `figure_global_vs_type_specific.pdf`",
        ]
    )
    with open(os.path.join(output_dir, "parameter_recommendation_report.md"), "w") as f:
        f.write("\n".join(lines))


def _append_ledger(output_dir: str, summary: dict, metrics_df: pd.DataFrame, recs_df: pd.DataFrame) -> None:
    ledger = "validation/docs/validation_results.md"
    lines = [
        "",
        "---",
        "",
        f"## {datetime.now().strftime('%Y-%m-%d')} - {summary['run_id']}",
        "",
        f"**Directory**: `{output_dir}/`  ",
        f"**Report**: `{output_dir}/parameter_recommendation_report.md`",
        "",
        "### Purpose",
        "",
        "Nested held-out calibration of Belief Engine parameter presets for",
        "`update_s` and `anchor_s` by participant type and topic context.",
        "",
        "### Held-Out Scheme Metrics",
        "",
    ]
    if not metrics_df.empty:
        cols = [
            "model",
            "change_rmse",
            "change_pearson_r",
            "change_direction_accuracy",
            "convergence_error",
        ]
        lines.append(metrics_df[[c for c in cols if c in metrics_df]].to_markdown(index=False))
    lines.extend(["", "### Recommended Presets", ""])
    if not recs_df.empty:
        cols = [
            "scheme",
            "preset_label",
            "recommended_update_s",
            "recommended_anchor_s",
            "heldout_rmse",
        ]
        lines.append(recs_df[[c for c in cols if c in recs_df]].to_markdown(index=False))
    with open(ledger, "a") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Held-out Belief Engine parameter recommendation run."
    )
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-by",
        choices=["group", "topic", "participant"],
        default="group",
    )
    parser.add_argument(
        "--clean-mode",
        choices=["quality", "movers"],
        default="movers",
    )
    parser.add_argument(
        "--exposure-mode",
        choices=["received", "all"],
        default="received",
        help="received uses only directed exposure; all also includes own outputs.",
    )
    parser.add_argument(
        "--uptake-mode",
        choices=["raw", "human_like"],
        default="raw",
        help="raw replays all extracted arguments; human_like applies vector dedup before replay.",
    )
    parser.add_argument(
        "--strength-source",
        choices=["active", "classifier", "llm"],
        default="active",
        help="Argument strength field used during replay.",
    )
    parser.add_argument("--argument-similarity-threshold", type=float, default=0.60)
    parser.add_argument("--self-similarity-threshold", type=float, default=0.45)
    parser.add_argument(
        "--objective",
        choices=["rmse", "mae", "balanced"],
        default="rmse",
    )
    parser.add_argument("--update-grid", type=str, default=None)
    parser.add_argument("--anchor-grid", type=str, default=None)
    parser.add_argument("--min-train", type=int, default=30)
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
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-ledger", action="store_true")
    args = parser.parse_args()

    update_grid = _parse_grid(args.update_grid)
    anchor_grid = _parse_grid(args.anchor_grid)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"parameter_recommendations_{args.split_by}_{timestamp}"
    output_dir = args.output_dir or os.path.join("validation/results", run_id)
    os.makedirs(output_dir, exist_ok=True)

    extraction_cache = _load_json_if_exists(args.extraction_cache)
    reclass_cache = _load_json_if_exists(args.reclass_cache)

    print("Loading DEBATE dataset...")
    timelines = build_participant_timelines(
        load_debate_dataset(), max_participants=args.sample
    )
    excluded = _load_excluded_ids(args.exclusion_registry)
    if excluded:
        before = len(timelines)
        timelines = [t for t in timelines if str(t.worker_id) not in excluded]
        print(f"Applied exclusions: {before} -> {len(timelines)} timelines")

    records_all = build_records(timelines, extraction_cache, reclass_cache=None)
    records = filter_records(records_all, args.clean_mode)
    print(f"Clean mode '{args.clean_mode}': {len(records_all)} -> {len(records)} records")
    print(f"Exposure mode: {args.exposure_mode}")
    print(f"Uptake mode: {args.uptake_mode}")
    print(f"Strength source: {args.strength_source}")

    update_args_by_worker = _prepare_update_args(
        records,
        extraction_cache,
        reclass_cache,
        args.exposure_mode,
        args.uptake_mode,
        args.strength_source,
        args.argument_similarity_threshold,
        args.self_similarity_threshold,
    )
    folds = make_folds(records, args.n_folds, args.seed, split_by=args.split_by)
    print(f"Fold sizes: {[len(fold) for fold in folds]}")

    individual_rows: list[dict] = []
    calibration_rows: list[dict] = []
    schemes = [
        "global",
        "participant_type",
        "response_type",
        "evidence_alignment",
        "topic_category",
    ]

    for fold_idx, test_records in enumerate(folds, start=1):
        train_records = [
            record
            for idx, fold in enumerate(folds, start=1)
            if idx != fold_idx
            for record in fold
        ]
        print(
            f"Fold {fold_idx}/{args.n_folds}: "
            f"train={len(train_records)} test={len(test_records)}"
        )
        individual_rows.extend(_prior_only_rows(test_records, fold_idx))
        for scheme in schemes:
            rows, params = _evaluate_scheme(
                scheme,
                fold_idx,
                train_records,
                test_records,
                update_args_by_worker,
                update_grid,
                anchor_grid,
                args.objective,
                args.min_train,
            )
            individual_rows.extend(rows)
            calibration_rows.extend(params)

    individual_df = pd.DataFrame(individual_rows)
    calibration_df = pd.DataFrame(calibration_rows)
    metrics_df = _metric_summary(individual_rows)
    stratified_df = summarize_prediction_rows(
        individual_rows, ["model", "participant_type"]
    )
    group_df = group_dynamics(individual_rows)
    recs_df = _recommendations(individual_df, calibration_df)

    individual_df.to_csv(os.path.join(output_dir, "individual_results.csv"), index=False)
    metrics_df.to_csv(os.path.join(output_dir, "heldout_metrics.csv"), index=False)
    stratified_df.to_csv(os.path.join(output_dir, "stratified_metrics.csv"), index=False)
    group_df.to_csv(os.path.join(output_dir, "group_results.csv"), index=False)
    calibration_df.to_csv(os.path.join(output_dir, "fold_selected_params.csv"), index=False)
    recs_df.to_csv(os.path.join(output_dir, "parameter_recommendations.csv"), index=False)

    _plot_parameter_presets(recs_df, output_dir)
    _plot_scheme_comparison(metrics_df, output_dir)

    summary = {
        "run_id": run_id,
        "timestamp": timestamp,
        "methodology": "heldout_parameter_recommendations",
        "split_by": args.split_by,
        "clean_mode": args.clean_mode,
        "exposure_mode": args.exposure_mode,
        "uptake_mode": args.uptake_mode,
        "strength_source": args.strength_source,
        "argument_similarity_threshold": args.argument_similarity_threshold,
        "self_similarity_threshold": args.self_similarity_threshold,
        "objective": args.objective,
        "n_folds": args.n_folds,
        "seed": args.seed,
        "n_participants": len(records),
        "n_participants_before_filter": len(records_all),
        "update_grid": update_grid,
        "anchor_grid": anchor_grid,
        "min_train": args.min_train,
        "outputs": {
            "individual_results": os.path.join(output_dir, "individual_results.csv"),
            "heldout_metrics": os.path.join(output_dir, "heldout_metrics.csv"),
            "fold_selected_params": os.path.join(output_dir, "fold_selected_params.csv"),
            "parameter_recommendations": os.path.join(output_dir, "parameter_recommendations.csv"),
        },
    }
    with open(os.path.join(output_dir, "parameter_recommendation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    _save_report(output_dir, summary, metrics_df, recs_df)
    if not args.no_ledger and output_dir.startswith("validation/results"):
        _append_ledger(output_dir, summary, metrics_df, recs_df)

    print(f"Saved parameter recommendation outputs to: {output_dir}")


if __name__ == "__main__":
    main()
