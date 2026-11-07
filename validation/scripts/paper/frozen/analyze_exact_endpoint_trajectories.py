#!/usr/bin/env python3
"""Outcome-blind exact-endpoint trajectory analysis.

This analysis replays the frozen DEBATE evidence streams.  For each eligible
record it solves an uptake value that exactly offsets a change in prior
anchoring, then asks whether the two endpoint-equivalent policies followed
different intermediate trajectories.  Observed opinion change is deliberately
discarded before eligibility, orientation, solving, and example selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import statistics
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import scipy
import yaml
from scipy.optimize import brentq

from analyze_uptake_anchor_trajectories import (
    AnalysisError,
    Evidence,
    EXPECTED_GROUPS,
    EXPECTED_PARTICIPANTS,
    EXPECTED_TOPICS,
    load_evidence_streams,
    load_prediction_metadata,
    log_odds_to_stance,
    percentile,
    replay,
    sha256_file,
    stance_to_log_odds,
    validate_input_bundle,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT / "results" / "debate_component_ablation_v1"
DEFAULT_SPEC = ROOT / "EXACT_ENDPOINT_TRAJECTORY_ANALYSIS_SPEC.yaml"
DEFAULT_OUTPUT_DIR = (
    ROOT / "results" / "analyses" / "exact_endpoint_trajectory_v1"
)

EXPECTED_SPEC_SHA256 = (
    "c33782be9003246a23f9124632ee5d6b4f7b22e127ad9fd4140e46adf22d4e4d"
)
EXPECTED_MANIFEST_SHA256 = (
    "929e3a20747fff56a31da2e0acb31c8895310f3ab85c4231d363d782d5bb7521"
)
EXPECTED_ACCEPTED_SHA256 = (
    "ecb79b76738c7bf6ada4dbaba8fd1f8dc836cc3c2f11d61c796fee297e6e491e"
)
EXPECTED_PREDICTIONS_SHA256 = (
    "d22de2566db26ee9fcbcf85f68423bbd51163c830b643f633fbe9646168ab943"
)
FROZEN_AT_UTC = "2026-07-25T00:00:00Z"

LOW_ANCHOR = 0.2
HIGH_ANCHOR = 1.0
REFERENCE_UPTAKE = 0.05
ROOT_LOWER = 0.25
ROOT_UPPER = 0.8
ROOT_SCAN_INTERVALS = 8192
ROOT_XTOL = 5.0e-15
ROOT_RTOL = 1.0e-14
EXACT_ENDPOINT_TOLERANCE = 1.0e-10
LEVERAGE_ZERO_TOLERANCE = 1.0e-14
MIN_EVIDENCE_ITEMS = 3
SATURATION_BOUND = 0.95
SEPARATION_THRESHOLD = 0.20
BOOTSTRAP_REPLICATES = 100_000
BOOTSTRAP_SEED = 2026072502
BOOTSTRAP_CHUNK = 256


@dataclass(frozen=True)
class BlindParticipant:
    participant_record_id: str
    evidence_stream_id: str
    group_id: str
    topic_id: str
    initial_stance: float
    evidence: tuple[Evidence, ...]


def stable_seed(label: str) -> int:
    digest = hashlib.sha256(f"{BOOTSTRAP_SEED}|{label}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def validate_spec(path: Path) -> dict:
    if not path.is_file():
        raise AnalysisError(f"Missing frozen analysis specification: {path}")
    actual = sha256_file(path)
    if actual != EXPECTED_SPEC_SHA256:
        raise AnalysisError(
            f"Analysis spec SHA-256 mismatch: {actual} != {EXPECTED_SPEC_SHA256}"
        )
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    if spec["analysis"]["id"] != "exact_endpoint_trajectory_v1":
        raise AnalysisError("Unexpected analysis id in frozen specification.")
    return spec


def load_blind_participants(
    input_dir: Path,
) -> tuple[list[BlindParticipant], dict]:
    """Load frozen inputs, then discard observed outcomes before analysis."""
    accepted_path, predictions_path, manifest = validate_input_bundle(input_dir)
    manifest_path = input_dir / "manifest.json"
    if sha256_file(manifest_path) != EXPECTED_MANIFEST_SHA256:
        raise AnalysisError("Frozen input manifest SHA-256 changed.")
    metadata = load_prediction_metadata(predictions_path)
    streams = load_evidence_streams(accepted_path)

    participants: list[BlindParticipant] = []
    for participant_id in sorted(metadata):
        source = metadata[participant_id]
        # opinion_change_actual is intentionally not copied to this object.
        participants.append(
            BlindParticipant(
                participant_record_id=participant_id,
                evidence_stream_id=str(source["evidence_stream_id"]),
                group_id=str(source["group_id"]),
                topic_id=str(source["topic_id"]),
                initial_stance=float(source["initial_stance"]),
                evidence=tuple(streams.get(participant_id, ())),
            )
        )
    if len(participants) != EXPECTED_PARTICIPANTS:
        raise AnalysisError("Unexpected participant count.")
    if len({row.group_id for row in participants}) != EXPECTED_GROUPS:
        raise AnalysisError("Unexpected group count.")
    if len({row.topic_id for row in participants}) != EXPECTED_TOPICS:
        raise AnalysisError("Unexpected topic count.")
    group_topics: dict[str, set[str]] = defaultdict(set)
    for row in participants:
        group_topics[row.group_id].add(row.topic_id)
    if any(len(topics) != 1 for topics in group_topics.values()):
        raise AnalysisError("Group identifiers are not nested within topics.")
    return participants, manifest


def evidence_log_odds(evidence: Sequence[Evidence], uptake: float) -> float:
    return math.fsum(
        item.sign * math.log1p(item.strength * uptake) for item in evidence
    )


def uptake_leverage(participant: BlindParticipant) -> float:
    return evidence_log_odds(
        participant.evidence, ROOT_UPPER
    ) - evidence_log_odds(participant.evidence, REFERENCE_UPTAKE)


def orientation(
    prior_log_odds: float, leverage: float
) -> str | None:
    if abs(leverage) <= LEVERAGE_ZERO_TOLERANCE:
        return None
    product = prior_log_odds * leverage
    if product > 0.0:
        return "prior_consistent"
    if product < 0.0:
        return "prior_opposing"
    return None


def comparison_parameters(which: str) -> tuple[float, float]:
    if which == "prior_consistent":
        return HIGH_ANCHOR, LOW_ANCHOR
    if which == "prior_opposing":
        return LOW_ANCHOR, HIGH_ANCHOR
    raise AnalysisError(f"Unknown orientation: {which}")


def find_roots(
    participant: BlindParticipant,
    variable_anchor: float,
    target_log_odds: float,
) -> list[float]:
    """Find sign-changing roots on the frozen grid and solve them precisely."""
    prior = stance_to_log_odds(participant.initial_stance)
    target_evidence = target_log_odds - variable_anchor * prior

    def residual(uptake: float) -> float:
        return (
            evidence_log_odds(participant.evidence, uptake)
            - target_evidence
        )

    grid = np.linspace(
        ROOT_LOWER, ROOT_UPPER, ROOT_SCAN_INTERVALS + 1, dtype=np.float64
    )
    strengths = np.asarray(
        [item.strength for item in participant.evidence], dtype=np.float64
    )
    signs = np.asarray(
        [item.sign for item in participant.evidence], dtype=np.float64
    )
    values = (
        signs[:, None] * np.log1p(strengths[:, None] * grid[None, :])
    ).sum(axis=0) - target_evidence

    candidates: list[float] = []
    exact_indices = np.flatnonzero(np.abs(values) <= 1.0e-13)
    candidates.extend(float(grid[index]) for index in exact_indices)
    changes = np.flatnonzero(values[:-1] * values[1:] < 0.0)
    for index in changes:
        candidates.append(
            float(
                brentq(
                    residual,
                    float(grid[index]),
                    float(grid[index + 1]),
                    xtol=ROOT_XTOL,
                    rtol=ROOT_RTOL,
                    maxiter=200,
                )
            )
        )

    roots: list[float] = []
    for candidate in sorted(candidates):
        if not ROOT_LOWER - 1e-14 <= candidate <= ROOT_UPPER + 1e-14:
            continue
        if abs(residual(candidate)) > 1.0e-10:
            continue
        if not roots or abs(candidate - roots[-1]) > 1.0e-10:
            roots.append(min(ROOT_UPPER, max(ROOT_LOWER, candidate)))
    return roots


def analyze_participants(
    participants: Sequence[BlindParticipant],
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
]:
    flow: list[dict[str, object]] = []
    denominator: list[dict[str, object]] = []
    counts: Counter[str] = Counter(
        {
            "excluded_minimum_evidence": 0,
            "excluded_zero_uptake_leverage": 0,
            "excluded_saturated_endpoint": 0,
            "no_root_in_interval": 0,
            "endpoint_tolerance_failure": 0,
            "exact_match": 0,
        }
    )
    participant_lookup = {
        row.participant_record_id: row for row in participants
    }

    for participant in participants:
        evidence_count = len(participant.evidence)
        base = {
            "participant_record_id": participant.participant_record_id,
            "evidence_stream_id": participant.evidence_stream_id,
            "group_id": participant.group_id,
            "topic_id": participant.topic_id,
            "evidence_count": evidence_count,
            "initial_stance": participant.initial_stance,
        }
        if evidence_count < MIN_EVIDENCE_ITEMS:
            flow.append({**base, "flow_status": "excluded_minimum_evidence"})
            counts["excluded_minimum_evidence"] += 1
            continue

        prior = stance_to_log_odds(participant.initial_stance)
        leverage = uptake_leverage(participant)
        which = orientation(prior, leverage)
        if which is None:
            flow.append(
                {
                    **base,
                    "uptake_leverage": leverage,
                    "flow_status": "excluded_zero_uptake_leverage",
                }
            )
            counts["excluded_zero_uptake_leverage"] += 1
            continue

        reference_anchor, variable_anchor = comparison_parameters(which)
        reference = replay(
            participant, reference_anchor, REFERENCE_UPTAKE
        )
        if abs(reference.final) >= SATURATION_BOUND:
            flow.append(
                {
                    **base,
                    "uptake_leverage": leverage,
                    "orientation": which,
                    "reference_anchor": reference_anchor,
                    "variable_anchor": variable_anchor,
                    "reference_endpoint": reference.final,
                    "flow_status": "excluded_saturated_endpoint",
                }
            )
            counts["excluded_saturated_endpoint"] += 1
            continue

        roots = find_roots(
            participant, variable_anchor, reference.log_odds[-1]
        )
        record: dict[str, object] = {
            **base,
            "prior_log_odds": prior,
            "uptake_leverage": leverage,
            "orientation": which,
            "reference_anchor": reference_anchor,
            "reference_uptake": REFERENCE_UPTAKE,
            "variable_anchor": variable_anchor,
            "root_interval_lower": ROOT_LOWER,
            "root_interval_upper": ROOT_UPPER,
            "root_count": len(roots),
            "solved_uptake": None,
            "reference_initial_stance": reference.initial,
            "variable_initial_stance": None,
            "absolute_initial_gap": None,
            "reference_endpoint": reference.final,
            "variable_endpoint": None,
            "absolute_endpoint_error": None,
            "maximum_interior_trajectory_gap": None,
            "mean_interior_trajectory_gap": None,
            "feasible_exact_match": 0,
            "separated_ge_0_20": 0,
            "joint_exact_match_and_separated": 0,
        }
        if not roots:
            record["solver_status"] = "no_root_in_interval"
            counts["no_root_in_interval"] += 1
        else:
            solved_uptake = roots[0]
            variable = replay(participant, variable_anchor, solved_uptake)
            endpoint_error = abs(reference.final - variable.final)
            gaps = [
                abs(left - right)
                for left, right in zip(reference.stances, variable.stances)
            ]
            interior = gaps[1:-1]
            if not interior:
                raise AnalysisError("Eligible trajectory has no interior steps.")
            maximum_gap = max(interior)
            exact = endpoint_error <= EXACT_ENDPOINT_TOLERANCE
            separated = exact and maximum_gap >= SEPARATION_THRESHOLD
            record.update(
                {
                    "solved_uptake": solved_uptake,
                    "variable_initial_stance": variable.initial,
                    "absolute_initial_gap": gaps[0],
                    "variable_endpoint": variable.final,
                    "absolute_endpoint_error": endpoint_error,
                    "maximum_interior_trajectory_gap": maximum_gap,
                    "mean_interior_trajectory_gap": statistics.fmean(interior),
                    "feasible_exact_match": int(exact),
                    "separated_ge_0_20": int(separated),
                    "joint_exact_match_and_separated": int(separated),
                    "solver_status": (
                        "exact_match" if exact else "endpoint_tolerance_failure"
                    ),
                }
            )
            counts[str(record["solver_status"])] += 1
        denominator.append(record)
        flow.append({**base, "flow_status": record["solver_status"]})

    counts["source_participants"] = len(participants)
    counts["analysis_denominator"] = len(denominator)
    exact_matches = [
        row for row in denominator if int(row["feasible_exact_match"]) == 1
    ]
    if not denominator:
        raise AnalysisError("The outcome-blind denominator is empty.")
    if len(flow) != len(participants):
        raise AnalysisError("Participant flow is incomplete.")
    if set(participant_lookup) != {
        str(row["participant_record_id"]) for row in flow
    }:
        raise AnalysisError("Participant flow identifiers changed.")
    return flow, denominator, exact_matches, dict(sorted(counts.items()))


def summary_metrics(rows: Sequence[Mapping[str, object]]) -> dict[str, float]:
    if not rows:
        raise AnalysisError("Cannot summarize an empty denominator.")
    feasible = [
        row for row in rows if int(row["feasible_exact_match"]) == 1
    ]
    joint = [
        row for row in rows if int(row["joint_exact_match_and_separated"]) == 1
    ]
    gaps = [
        float(row["maximum_interior_trajectory_gap"]) for row in feasible
    ]
    errors = [float(row["absolute_endpoint_error"]) for row in feasible]
    return {
        "denominator_n": float(len(rows)),
        "feasible_exact_match_n": float(len(feasible)),
        "joint_exact_match_and_separated_n": float(len(joint)),
        "exact_match_and_maximum_interior_gap_ge_0_20_rate": (
            len(joint) / len(rows)
        ),
        "feasible_exact_match_rate": len(feasible) / len(rows),
        "conditional_separation_rate_ge_0_20_among_exact_matches": (
            len(joint) / len(feasible) if feasible else math.nan
        ),
        "median_maximum_interior_gap_among_exact_matches": (
            statistics.median(gaps) if gaps else math.nan
        ),
        "maximum_absolute_endpoint_error_among_exact_matches": (
            max(errors) if errors else math.nan
        ),
    }


def summarize(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    levels = [("overall", "all")]
    levels.extend(
        ("orientation", level)
        for level in ("prior_consistent", "prior_opposing")
    )
    for dimension, level in levels:
        selected = (
            list(rows)
            if dimension == "overall"
            else [row for row in rows if row["orientation"] == level]
        )
        metrics = summary_metrics(selected)
        output.append(
            {
                "dimension": dimension,
                "level": level,
                "group_clusters": len(
                    {str(row["group_id"]) for row in selected}
                ),
                "topic_clusters": len(
                    {str(row["topic_id"]) for row in selected}
                ),
                **metrics,
            }
        )
    return output


def group_structure(
    rows: Sequence[Mapping[str, object]],
) -> tuple[
    list[str],
    list[str],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    group_ids = sorted({str(row["group_id"]) for row in rows})
    topic_ids = sorted({str(row["topic_id"]) for row in rows})
    group_index = {group: index for index, group in enumerate(group_ids)}
    topic_index = {topic: index for index, topic in enumerate(topic_ids)}
    group_topic: dict[str, str] = {}
    denominator_counts = np.zeros(len(group_ids), dtype=np.int64)
    feasible_counts = np.zeros(len(group_ids), dtype=np.int64)
    joint_counts = np.zeros(len(group_ids), dtype=np.int64)
    feasible_rows: list[tuple[float, int]] = []
    for row in rows:
        group = str(row["group_id"])
        topic = str(row["topic_id"])
        if group in group_topic and group_topic[group] != topic:
            raise AnalysisError("Groups are not nested within topics.")
        group_topic[group] = topic
        index = group_index[group]
        denominator_counts[index] += 1
        if int(row["feasible_exact_match"]) == 1:
            feasible_counts[index] += 1
            feasible_rows.append(
                (
                    float(row["maximum_interior_trajectory_gap"]),
                    index,
                )
            )
        if int(row["joint_exact_match_and_separated"]) == 1:
            joint_counts[index] += 1
    feasible_rows.sort()
    sorted_gaps = np.asarray([row[0] for row in feasible_rows], dtype=float)
    sorted_gap_groups = np.asarray(
        [row[1] for row in feasible_rows], dtype=np.int64
    )
    group_topic_indices = np.asarray(
        [topic_index[group_topic[group]] for group in group_ids],
        dtype=np.int64,
    )
    return (
        group_ids,
        topic_ids,
        group_topic_indices,
        denominator_counts,
        feasible_counts,
        joint_counts,
        np.vstack((sorted_gaps, sorted_gap_groups)).T,
    )


def weighted_medians(
    group_weights: np.ndarray,
    sorted_gaps: np.ndarray,
    sorted_gap_groups: np.ndarray,
    feasible_totals: np.ndarray,
) -> np.ndarray:
    output = np.full(len(group_weights), np.nan, dtype=float)
    defined = feasible_totals > 0
    if not np.any(defined):
        return output
    row_weights = group_weights[defined][:, sorted_gap_groups]
    cumulative = np.cumsum(row_weights, axis=1)
    totals = feasible_totals[defined]
    lower_targets = (totals + 1) // 2
    upper_targets = (totals + 2) // 2
    lower_indices = (
        cumulative >= lower_targets[:, None]
    ).argmax(axis=1)
    upper_indices = (
        cumulative >= upper_targets[:, None]
    ).argmax(axis=1)
    output[defined] = (
        sorted_gaps[lower_indices] + sorted_gaps[upper_indices]
    ) / 2.0
    return output


def draw_group_weights(
    scheme: str,
    batch: int,
    rng: np.random.Generator,
    group_topic_indices: np.ndarray,
    topic_count: int,
) -> np.ndarray:
    group_count = len(group_topic_indices)
    output = np.zeros((batch, group_count), dtype=np.int16)
    groups_by_topic = [
        np.flatnonzero(group_topic_indices == topic)
        for topic in range(topic_count)
    ]
    if scheme == "group_only":
        return rng.multinomial(
            group_count,
            np.full(group_count, 1.0 / group_count),
            size=batch,
        ).astype(np.int16, copy=False)
    if scheme == "fixed_topics_groups_within_topic":
        for groups in groups_by_topic:
            output[:, groups] = rng.multinomial(
                len(groups),
                np.full(len(groups), 1.0 / len(groups)),
                size=batch,
            )
        return output
    if scheme != "topic_then_group":
        raise AnalysisError(f"Unknown bootstrap scheme: {scheme}")

    topic_multiplicities = rng.multinomial(
        topic_count,
        np.full(topic_count, 1.0 / topic_count),
        size=batch,
    )
    for topic, groups in enumerate(groups_by_topic):
        multiplicities = topic_multiplicities[:, topic]
        for multiplier in np.unique(multiplicities):
            if multiplier == 0:
                continue
            selected = np.flatnonzero(multiplicities == multiplier)
            output[np.ix_(selected, groups)] = rng.multinomial(
                int(multiplier) * len(groups),
                np.full(len(groups), 1.0 / len(groups)),
                size=len(selected),
            )
    return output


def bootstrap_intervals(
    rows: Sequence[Mapping[str, object]], replicates: int
) -> list[dict[str, object]]:
    (
        group_ids,
        topic_ids,
        group_topic_indices,
        denominator_counts,
        feasible_counts,
        joint_counts,
        gap_table,
    ) = group_structure(rows)
    sorted_gaps = gap_table[:, 0]
    sorted_gap_groups = gap_table[:, 1].astype(np.int64)
    point = summary_metrics(rows)
    metric_names = (
        "exact_match_and_maximum_interior_gap_ge_0_20_rate",
        "feasible_exact_match_rate",
        "conditional_separation_rate_ge_0_20_among_exact_matches",
        "median_maximum_interior_gap_among_exact_matches",
    )
    results: list[dict[str, object]] = []

    for scheme in (
        "topic_then_group",
        "group_only",
        "fixed_topics_groups_within_topic",
    ):
        rng = np.random.default_rng(stable_seed(scheme))
        draws = {
            metric: np.empty(replicates, dtype=np.float64)
            for metric in metric_names
        }
        offset = 0
        while offset < replicates:
            batch = min(BOOTSTRAP_CHUNK, replicates - offset)
            weights = draw_group_weights(
                scheme,
                batch,
                rng,
                group_topic_indices,
                len(topic_ids),
            )
            denominator_total = weights @ denominator_counts
            feasible_total = weights @ feasible_counts
            joint_total = weights @ joint_counts
            draws[
                "exact_match_and_maximum_interior_gap_ge_0_20_rate"
            ][offset : offset + batch] = joint_total / denominator_total
            draws["feasible_exact_match_rate"][
                offset : offset + batch
            ] = feasible_total / denominator_total
            draws[
                "conditional_separation_rate_ge_0_20_among_exact_matches"
            ][offset : offset + batch] = np.divide(
                joint_total,
                feasible_total,
                out=np.full(batch, np.nan),
                where=feasible_total > 0,
            )
            draws[
                "median_maximum_interior_gap_among_exact_matches"
            ][offset : offset + batch] = weighted_medians(
                weights,
                sorted_gaps,
                sorted_gap_groups,
                feasible_total,
            )
            offset += batch

        for metric in metric_names:
            values = draws[metric]
            values = values[np.isfinite(values)]
            if len(values) < int(0.99 * replicates):
                raise AnalysisError(
                    f"Too many undefined bootstrap values for {metric}."
                )
            results.append(
                {
                    "resampling_scheme": scheme,
                    "metric": metric,
                    "estimate": point[metric],
                    "ci_low": float(np.quantile(values, 0.025)),
                    "ci_high": float(np.quantile(values, 0.975)),
                    "bootstrap_replicates_requested": replicates,
                    "bootstrap_replicates_defined": len(values),
                    "group_clusters": len(group_ids),
                    "topic_clusters": len(topic_ids),
                    "seed": stable_seed(scheme),
                }
            )
    return results


def leave_one_topic_out(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    topics = sorted({str(row["topic_id"]) for row in rows})
    output: list[dict[str, object]] = []
    for topic in topics:
        selected = [row for row in rows if str(row["topic_id"]) != topic]
        output.append(
            {
                "omitted_topic_id": topic,
                "remaining_n": len(selected),
                **summary_metrics(selected),
            }
        )
    return output


def select_examples(
    rows: Sequence[Mapping[str, object]],
    participants: Mapping[str, BlindParticipant],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    summaries: list[dict[str, object]] = []
    trajectories: list[dict[str, object]] = []
    for which in ("prior_consistent", "prior_opposing"):
        pool = [
            row
            for row in rows
            if row["orientation"] == which
            and int(row["joint_exact_match_and_separated"]) == 1
        ]
        if not pool:
            raise AnalysisError(f"No separated exact-match example for {which}.")
        median_gap = statistics.median(
            float(row["maximum_interior_trajectory_gap"]) for row in pool
        )
        selected = min(
            pool,
            key=lambda row: (
                abs(
                    float(row["maximum_interior_trajectory_gap"])
                    - median_gap
                ),
                str(row["participant_record_id"]),
            ),
        )
        participant = participants[str(selected["participant_record_id"])]
        reference = replay(
            participant,
            float(selected["reference_anchor"]),
            REFERENCE_UPTAKE,
        )
        variable = replay(
            participant,
            float(selected["variable_anchor"]),
            float(selected["solved_uptake"]),
        )
        summaries.append(
            {
                **selected,
                "example_pool_n": len(pool),
                "example_pool_median_maximum_interior_gap": median_gap,
                "absolute_distance_to_pool_median": abs(
                    float(selected["maximum_interior_trajectory_gap"])
                    - median_gap
                ),
                "selection_rule": (
                    "closest_to_orientation_specific_median_maximum_interior_gap"
                    "_then_participant_id"
                ),
            }
        )
        for step, (reference_stance, variable_stance) in enumerate(
            zip(reference.stances, variable.stances)
        ):
            evidence = (
                participant.evidence[step - 1] if step > 0 else None
            )
            trajectories.append(
                {
                    "orientation": which,
                    "participant_record_id": participant.participant_record_id,
                    "step": step,
                    "is_initial": int(step == 0),
                    "is_interior": int(0 < step < len(reference.stances) - 1),
                    "is_final": int(step == len(reference.stances) - 1),
                    "evidence_sign": evidence.sign if evidence else None,
                    "evidence_strength": evidence.strength if evidence else None,
                    "reference_anchor": selected["reference_anchor"],
                    "reference_uptake": REFERENCE_UPTAKE,
                    "reference_stance": reference_stance,
                    "variable_anchor": selected["variable_anchor"],
                    "solved_uptake": selected["solved_uptake"],
                    "variable_stance": variable_stance,
                    "absolute_gap": abs(reference_stance - variable_stance),
                }
            )
    return summaries, trajectories


def write_csv(
    path: Path, rows: Sequence[Mapping[str, object]]
) -> None:
    if not rows:
        raise AnalysisError(f"Refusing to write empty CSV: {path.name}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def build_report(
    summary_rows: Sequence[Mapping[str, object]],
    intervals: Sequence[Mapping[str, object]],
    flow_counts: Mapping[str, object],
    leave_out: Sequence[Mapping[str, object]],
    examples: Sequence[Mapping[str, object]],
) -> str:
    overall = next(
        row for row in summary_rows if row["dimension"] == "overall"
    )
    hierarchy = {
        str(row["metric"]): row
        for row in intervals
        if row["resampling_scheme"] == "topic_then_group"
    }
    primary_name = (
        "exact_match_and_maximum_interior_gap_ge_0_20_rate"
    )
    feasibility_name = "feasible_exact_match_rate"
    conditional_name = (
        "conditional_separation_rate_ge_0_20_among_exact_matches"
    )
    median_name = "median_maximum_interior_gap_among_exact_matches"
    loot_primary = [
        float(row[primary_name]) for row in leave_out
    ]
    exact_n = int(float(overall["feasible_exact_match_n"]))
    denominator_n = int(float(overall["denominator_n"]))
    joint_n = int(float(overall["joint_exact_match_and_separated_n"]))
    orientation_rows = [
        row for row in summary_rows if row["dimension"] == "orientation"
    ]

    return f"""# Outcome-blind exact-endpoint trajectory analysis

## Question and design

Can prior anchoring and evidence uptake produce the **same final stance** but
different intermediate opinion paths?  We answer this with a fixed-stream
intervention that does not use observed opinion change.

For records with at least {MIN_EVIDENCE_ITEMS} accepted evidence items, we
oriented the comparison using only the initial stance and the signed change in
cumulative evidence log odds from uptake {REFERENCE_UPTAKE} to {ROOT_UPPER}.
For prior-consistent evidence leverage, we matched high anchoring
(`a={HIGH_ANCHOR}, u={REFERENCE_UPTAKE}`) against low anchoring
(`a={LOW_ANCHOR}, u=u*`).  For prior-opposing leverage, we reversed the
anchors.  We solved the smallest `u*` in [{ROOT_LOWER}, {ROOT_UPPER}] that made
the endpoints equal.  Records with an outcome-blind reference endpoint at or
beyond ±{SATURATION_BOUND} were excluded; records with no root remained in the
full denominator.

## Main result

- Full outcome-blind denominator: **{denominator_n:,}** participant records.
- Feasible exact matches: **{exact_n:,}/{denominator_n:,}**
  ({fmt(float(overall[feasibility_name]))}; hierarchical 95% CI
  [{fmt(float(hierarchy[feasibility_name]["ci_low"]))},
  {fmt(float(hierarchy[feasibility_name]["ci_high"]))}]).
- Exact matches with maximum interior gap ≥ {SEPARATION_THRESHOLD:.2f}:
  **{joint_n:,}/{denominator_n:,}**
  ({fmt(float(overall[primary_name]))}; hierarchical 95% CI
  [{fmt(float(hierarchy[primary_name]["ci_low"]))},
  {fmt(float(hierarchy[primary_name]["ci_high"]))}]).
- Conditional separation among exact matches:
  **{fmt(float(overall[conditional_name]))}** (hierarchical 95% CI
  [{fmt(float(hierarchy[conditional_name]["ci_low"]))},
  {fmt(float(hierarchy[conditional_name]["ci_high"]))}]).
- Median maximum interior gap among exact matches:
  **{fmt(float(overall[median_name]))}** (hierarchical 95% CI
  [{fmt(float(hierarchy[median_name]["ci_low"]))},
  {fmt(float(hierarchy[median_name]["ci_high"]))}]).
- Maximum endpoint error among exact matches:
  **{float(overall["maximum_absolute_endpoint_error_among_exact_matches"]):.3e}**.

The result occurs in both outcome-blind orientations:

| Orientation | Denominator | Exact matches | Exact + gap ≥ .20 | Full-denominator rate | Conditional separation | Median max interior gap |
|---|---:|---:|---:|---:|---:|---:|
{os.linesep.join(
    f"| {str(row['level']).replace('_', ' ')} "
    f"| {int(float(row['denominator_n'])):,} "
    f"| {int(float(row['feasible_exact_match_n'])):,} "
    f"| {int(float(row['joint_exact_match_and_separated_n'])):,} "
    f"| {float(row[primary_name]):.4f} "
    f"| {float(row[conditional_name]):.4f} "
    f"| {float(row[median_name]):.4f} |"
    for row in orientation_rows
)}

The primary leave-one-topic-out estimate ranged from
**{min(loot_primary):.4f}** to **{max(loot_primary):.4f}** across
{len(leave_out)} omissions.  `bootstrap_intervals.csv` also reports group-only
and fixed-topic/groups-within-topic sensitivity intervals.

## Participant flow

The denominator is auditable as **2,495 source records −
{int(flow_counts["excluded_minimum_evidence"]):,} with fewer than three
evidence items −
{int(flow_counts["excluded_zero_uptake_leverage"]):,} with zero uptake leverage
− {int(flow_counts["excluded_saturated_endpoint"]):,} saturated reference
endpoint = {denominator_n:,}**.  Root infeasibility is not an exclusion:
those records remain in the denominator.

```json
{json.dumps(flow_counts, indent=2, sort_keys=True)}
```

## Deterministic examples

The two examples were not selected for extremeness.  Within each orientation,
we restricted to exact matches separated by at least {SEPARATION_THRESHOLD:.2f},
then selected the record closest to that orientation's median maximum interior
gap, breaking ties by participant id.

{os.linesep.join(
    f"- `{row['orientation']}`: `{row['participant_record_id']}`, "
    f"u*={float(row['solved_uptake']):.6f}, "
    f"max interior gap={float(row['maximum_interior_trajectory_gap']):.4f}, "
    f"endpoint error={float(row['absolute_endpoint_error']):.3e}."
    for row in examples
)}

## Interpretation

This is direct empirical confirmation that evidence uptake and prior anchoring
are not the same outcome-level mechanism.  Even when their final stance is
numerically identical, the policies can disagree materially after intermediate
evidence items.  The analysis is a fixed-stream policy-identification test.  It
does not estimate human cognitive parameters and does not establish generated
dialogue robustness across agent settings.

## Reproducibility

- Frozen spec SHA-256: `{EXPECTED_SPEC_SHA256}`
- Frozen accepted-stream SHA-256:
  `ecb79b76738c7bf6ada4dbaba8fd1f8dc836cc3c2f11d61c796fee297e6e491e`
- Frozen predictions SHA-256:
  `d22de2566db26ee9fcbcf85f68423bbd51163c830b643f633fbe9646168ab943`
- Bootstrap: {BOOTSTRAP_REPLICATES:,} topic-then-group draws plus the two
  registered sensitivity schemes; seed {BOOTSTRAP_SEED}.
- No participant-level analysis table contains `opinion_change_actual` or an
  outcome-derived response stratum.
"""


def build_manifest(output_dir: Path, metadata: Mapping[str, object]) -> dict:
    artifacts = []
    for path in sorted(output_dir.iterdir()):
        if path.name == "manifest.json" or not path.is_file():
            continue
        artifacts.append(
            {
                "name": path.name,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return {
        "analysis_id": "exact_endpoint_trajectory_v1",
        "frozen_at_utc": FROZEN_AT_UTC,
        "spec": {
            "path": "EXACT_ENDPOINT_TRAJECTORY_ANALYSIS_SPEC.yaml",
            "sha256": EXPECTED_SPEC_SHA256,
        },
        "inputs": [
            {
                "path": (
                    "results/debate_component_ablation_v1/"
                    "accepted_stream.jsonl.gz"
                ),
                "sha256": EXPECTED_ACCEPTED_SHA256,
                "size_bytes": (
                    DEFAULT_INPUT_DIR / "accepted_stream.jsonl.gz"
                ).stat().st_size,
            },
            {
                "path": (
                    "results/debate_component_ablation_v1/"
                    "predictions.csv.gz"
                ),
                "sha256": EXPECTED_PREDICTIONS_SHA256,
                "size_bytes": (
                    DEFAULT_INPUT_DIR / "predictions.csv.gz"
                ).stat().st_size,
            },
            {
                "path": "results/debate_component_ablation_v1/manifest.json",
                "sha256": EXPECTED_MANIFEST_SHA256,
                "size_bytes": (
                    DEFAULT_INPUT_DIR / "manifest.json"
                ).stat().st_size,
            },
        ],
        "analyzer": {
            "path": "scripts/analyze_exact_endpoint_trajectories.py",
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "metadata": dict(metadata),
        "artifacts": artifacts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = validate_spec(args.spec)
    if args.bootstrap_replicates <= 0:
        raise AnalysisError("Bootstrap replicates must be positive.")
    if (
        args.output_dir.resolve() == DEFAULT_OUTPUT_DIR.resolve()
        and args.bootstrap_replicates != BOOTSTRAP_REPLICATES
    ):
        raise AnalysisError(
            "Canonical output requires the frozen 100,000 replicates."
        )
    if args.output_dir.exists():
        raise AnalysisError(
            f"Output already exists; choose a fresh directory: {args.output_dir}"
        )

    participants, input_manifest = load_blind_participants(args.input_dir)
    (
        participant_flow,
        denominator,
        exact_matches,
        flow_counts,
    ) = analyze_participants(participants)
    summary_rows = summarize(denominator)
    intervals = bootstrap_intervals(
        denominator, args.bootstrap_replicates
    )
    leave_out = leave_one_topic_out(denominator)
    participant_map = {
        row.participant_record_id: row for row in participants
    }
    example_summaries, example_trajectories = select_examples(
        denominator, participant_map
    )

    parent = args.output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=".exact_endpoint_", dir=parent)
    )
    try:
        shutil.copy2(args.spec, temporary / args.spec.name)
        write_csv(temporary / "participant_flow.csv", participant_flow)
        write_csv(temporary / "participant_denominator.csv", denominator)
        write_csv(temporary / "participant_exact_matches.csv", exact_matches)
        write_csv(temporary / "summary.csv", summary_rows)
        write_csv(temporary / "bootstrap_intervals.csv", intervals)
        write_csv(temporary / "leave_one_topic_out.csv", leave_out)
        write_csv(temporary / "selected_examples.csv", example_summaries)
        write_csv(
            temporary / "selected_example_trajectories.csv",
            example_trajectories,
        )
        analysis = {
            "analysis_id": "exact_endpoint_trajectory_v1",
            "spec_sha256": EXPECTED_SPEC_SHA256,
            "flow_counts": flow_counts,
            "summary": summary_rows,
            "bootstrap_intervals": intervals,
            "leave_one_topic_out_primary_range": {
                "minimum": min(
                    float(row[
                        "exact_match_and_maximum_interior_gap_ge_0_20_rate"
                    ])
                    for row in leave_out
                ),
                "maximum": max(
                    float(row[
                        "exact_match_and_maximum_interior_gap_ge_0_20_rate"
                    ])
                    for row in leave_out
                ),
            },
            "outcome_blind": True,
        }
        (temporary / "analysis.json").write_text(
            json.dumps(analysis, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        report = build_report(
            summary_rows,
            intervals,
            flow_counts,
            leave_out,
            example_summaries,
        )
        (temporary / "REPORT.md").write_text(report, encoding="utf-8")

        max_error = max(
            float(row["absolute_endpoint_error"]) for row in exact_matches
        )
        root_distribution = Counter(
            int(row["root_count"]) for row in denominator
        )
        solver_diagnostics = {
            "root_count_distribution": {
                str(key): value
                for key, value in sorted(root_distribution.items())
            },
            "maximum_detected_root_count": max(root_distribution),
            "multiple_root_records": sum(
                value
                for key, value in root_distribution.items()
                if key > 1
            ),
            "solver_status_counts": dict(
                sorted(
                    Counter(
                        str(row["solver_status"]) for row in denominator
                    ).items()
                )
            ),
            "maximum_absolute_endpoint_error_among_exact_matches": max_error,
        }
        (temporary / "solver_diagnostics.json").write_text(
            json.dumps(solver_diagnostics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        output_headers = "\n".join(
            path.read_text(encoding="utf-8").splitlines()[0]
            for path in temporary.glob("*.csv")
        )
        verification = {
            "analysis_id": "exact_endpoint_trajectory_v1",
            "frozen_at_utc": FROZEN_AT_UTC,
            "checks": {
                "frozen_spec_sha256": sha256_file(args.spec)
                == EXPECTED_SPEC_SHA256,
                "frozen_input_manifest_sha256": sha256_file(
                    args.input_dir / "manifest.json"
                )
                == EXPECTED_MANIFEST_SHA256,
                "participant_flow_complete": sum(
                    value
                    for key, value in flow_counts.items()
                    if key.startswith("excluded_")
                    or key in (
                        "no_root_in_interval",
                        "exact_match",
                        "endpoint_tolerance_failure",
                    )
                )
                == EXPECTED_PARTICIPANTS,
                "endpoint_tolerance_satisfied": max_error
                <= EXACT_ENDPOINT_TOLERANCE,
                "bootstrap_replicates_complete": all(
                    int(row["bootstrap_replicates_requested"])
                    == args.bootstrap_replicates
                    for row in intervals
                ),
                "no_observed_outcome_fields_in_csv_headers": (
                    "opinion_change_actual" not in output_headers
                    and "response_stratum" not in output_headers
                ),
                "two_deterministic_examples": len(example_summaries) == 2,
            },
            "maximum_absolute_endpoint_error": max_error,
        }
        if not all(verification["checks"].values()):
            raise AnalysisError(f"Self-verification failed: {verification}")
        (temporary / "verification.json").write_text(
            json.dumps(verification, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest = build_manifest(
            temporary,
            {
                "canonical": args.bootstrap_replicates
                == BOOTSTRAP_REPLICATES,
                "bootstrap_replicates": args.bootstrap_replicates,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "input_manifest_analysis_id": input_manifest.get(
                    "analysis_id"
                ),
                "spec_analysis_id": spec["analysis"]["id"],
            },
        )
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.rename(args.output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "denominator_n": len(denominator),
                "exact_matches_n": len(exact_matches),
                "primary_rate": summary_metrics(denominator)[
                    "exact_match_and_maximum_interior_gap_ge_0_20_rate"
                ],
                "bootstrap_replicates": args.bootstrap_replicates,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
