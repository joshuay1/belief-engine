#!/usr/bin/env python3
"""Analyze the locked component factorial at the paired debate unit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = (
    REPO_ROOT / "rebuttal/component_necessity_factorial_v1/registry.csv"
)
SPEC_PATH = REPO_ROOT / "rebuttal/COMPONENT_NECESSITY_FACTORIAL_SPEC.yaml"

METRICS = (
    "final_gap",
    "trajectory_gap_auc",
    "mean_absolute_stance_shift",
    "absolute_midpoint_shift",
    "same_side_final",
    "final_side_switch_count",
    "any_side_switch_count",
)
PRIMARY = (
    "structured_representation_main",
    "response_retrieval_main",
    "persistence_main",
    "deduplication_at_full_setting",
    "graded_evidence_scoring_at_full_setting",
)
SECONDARY = (
    "structured_update_path_without_response_retrieval",
    "structured_effect_with_response_retrieval",
    "representation_by_retrieval",
    "representation_by_persistence",
    "retrieval_by_persistence",
    "representation_by_retrieval_by_persistence",
)
CORE_CONDITIONS = tuple(
    f"be_s{s}_r{r}_p{p}_d1"
    for s in (0, 1)
    for r in (0, 1)
    for p in (0, 1)
)
UNIFORM_STRENGTH_VALUE = 0.8475946340113878
EXACT_SIGN_PATTERNS = 1024


def read_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantile(values, probability):
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def normalized_trapezoid(values):
    return sum(
        (left + right) / 2
        for left, right in zip(values, values[1:])
    ) / (len(values) - 1)


def _hex64(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _score_summary(values):
    scores = [float(value) for value in values]
    if not scores:
        return {
            "count": 0,
            "mean": None,
            "sd": None,
            "q05": None,
            "q25": None,
            "q50": None,
            "q75": None,
            "q95": None,
            "mean_absolute_deviation_from_uniform": None,
        }
    if any(
        not math.isfinite(value) or not 0.0 <= value <= 1.0
        for value in scores
    ):
        raise ValueError("Evidence-score telemetry contains an invalid score")
    return {
        "count": len(scores),
        "mean": statistics.fmean(scores),
        "sd": statistics.stdev(scores) if len(scores) > 1 else 0.0,
        "q05": quantile(scores, 0.05),
        "q25": quantile(scores, 0.25),
        "q50": quantile(scores, 0.50),
        "q75": quantile(scores, 0.75),
        "q95": quantile(scores, 0.95),
        "mean_absolute_deviation_from_uniform": statistics.fmean(
            abs(value - UNIFORM_STRENGTH_VALUE) for value in scores
        ),
    }


def _rank_signature(fingerprints, scores):
    if len(fingerprints) != len(scores):
        raise ValueError("Ranking telemetry length mismatch")
    return tuple(
        fingerprint
        for _index, fingerprint, _score in sorted(
            (
                (index, str(fingerprint), float(score))
                for index, (fingerprint, score)
                in enumerate(zip(fingerprints, scores))
            ),
            key=lambda item: (-item[2], item[1], item[0]),
        )
    )


def _validated_extraction_events(extraction):
    events = extraction.get("manipulation_events", [])
    extract_calls = int(extraction.get("extract_calls", -1))
    if len(events) != extract_calls:
        raise ValueError("Extraction manipulation-event count mismatch")
    expected_mode = str(extraction.get("output_mode", ""))
    normalized = []
    total_records = 0
    for expected_index, event in enumerate(events, start=1):
        if int(event.get("call_index", -1)) != expected_index:
            raise ValueError("Extraction manipulation-event index mismatch")
        if str(event.get("output_mode", "")) != expected_mode:
            raise ValueError("Extraction manipulation-event mode mismatch")
        fingerprints = [
            str(value) for value in event.get("record_fingerprints", [])
        ]
        classifier_scores = [
            float(value) for value in event.get("classifier_strengths", [])
        ]
        active_scores = [
            float(value) for value in event.get("active_strengths", [])
        ]
        count = int(event.get("records_emitted", -1))
        if (
            count < 0
            or len(fingerprints) != count
            or len(classifier_scores) != count
            or len(active_scores) != count
            or any(not _hex64(value) for value in fingerprints)
        ):
            raise ValueError("Extraction manipulation-event shape mismatch")
        _score_summary(classifier_scores)
        _score_summary(active_scores)
        identity_sha256 = str(
            event.get("record_identity_sequence_sha256", "")
        )
        expected_identity_sha256 = hashlib.sha256(
            json.dumps(
                fingerprints,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if (
            not _hex64(identity_sha256)
            or identity_sha256 != expected_identity_sha256
        ):
            raise ValueError("Extraction identity fingerprint mismatch")
        normalized.append(
            {
                "call_index": expected_index,
                "output_mode": expected_mode,
                "source_role": str(event.get("source_role", "")),
                "records_emitted": count,
                "record_fingerprints": fingerprints,
                "record_identity_sequence_sha256": identity_sha256,
                "classifier_strengths": classifier_scores,
                "active_strengths": active_scores,
            }
        )
        total_records += count
    if total_records != int(extraction.get("records_emitted", -1)):
        raise ValueError("Extraction record telemetry does not reconcile")
    if extraction.get("manipulation_telemetry_use") != (
        "report_only_never_controls_generation_or_continuation"
    ):
        raise ValueError("Extraction manipulation telemetry use is not locked")
    return normalized


def component_telemetry(result):
    """Extract outcome-independent factor-manipulation telemetry."""
    metadata = result.get("metadata", {})
    extraction = metadata.get("argument_extraction", {})
    extraction_events = _validated_extraction_events(extraction)
    dedup_agents = metadata.get("deduplication", {}).get("agents", {})
    retrieval_agents = metadata.get("response_retrieval", {}).get(
        "agents",
        {},
    )
    if set(dedup_agents) != {"Pro", "Con"}:
        raise ValueError("Deduplication telemetry must contain Pro and Con")
    if set(retrieval_agents) != {"Pro", "Con"}:
        raise ValueError("Retrieval telemetry must contain Pro and Con")

    dynamic_status = defaultdict(int)
    final_dynamic_total = 0
    final_dynamic_active = 0
    final_dynamic_archived = 0
    canonical_state_events = []
    for agent_name in sorted(dedup_agents):
        agent = dedup_agents[agent_name]
        for status, count in agent.get(
            "dynamic_status_counts",
            {},
        ).items():
            dynamic_status[str(status)] += int(count)
        final_dynamic_total += int(
            agent.get("final_dynamic_records_total", 0)
        )
        final_dynamic_active += int(
            agent.get("final_dynamic_active_records", 0)
        )
        final_dynamic_archived += int(
            agent.get("final_dynamic_archived_records", 0)
        )
        for key in (
            "final_dynamic_canonical_state_sha256",
            "final_dynamic_active_canonical_state_sha256",
        ):
            if not _hex64(str(agent.get(key, ""))):
                raise ValueError(
                    f"Canonical dynamic-state telemetry missing: {key}"
                )
        canonical_state_events.append(
            {
                "agent": agent_name,
                "all": str(
                    agent["final_dynamic_canonical_state_sha256"]
                ),
                "active": str(
                    agent[
                        "final_dynamic_active_canonical_state_sha256"
                    ]
                ),
            }
        )
    if final_dynamic_active + final_dynamic_archived != final_dynamic_total:
        raise ValueError("Final dynamic-record telemetry is inconsistent")

    retrieval_records_returned = 0
    context_events = []
    for agent_name in sorted(retrieval_agents):
        agent = retrieval_agents[agent_name]
        retrieval_records_returned += int(
            agent.get("records_returned_total", 0)
        )
        events = agent.get("events", [])
        if len(events) != int(agent.get("reply_calls", -1)):
            raise ValueError("Retrieval event count does not match reply calls")
        for event in events:
            context_events.append(
                {
                    "agent": agent_name,
                    "call_index": int(event["call_index"]),
                    "retrieval_enabled": bool(
                        event["retrieval_enabled"]
                    ),
                    "records_returned": int(event["records_returned"]),
                    "memory_context_tokens": int(
                        event["memory_context_tokens"]
                    ),
                    "memory_context_budget_saturated": bool(
                        event["memory_context_budget_saturated"]
                    ),
                    "rendered_context_sha256": str(
                        event["rendered_context_sha256"]
                    ),
                    "retrieved_record_identity_sequence_sha256": str(
                        event[
                            "retrieved_record_identity_sequence_sha256"
                        ]
                    ),
                    "retrieved_record_identity_multiset_sha256": str(
                        event[
                            "retrieved_record_identity_multiset_sha256"
                        ]
                    ),
                    "retrieved_active_score_sequence_sha256": str(
                        event[
                            "retrieved_active_score_sequence_sha256"
                        ]
                    ),
                }
            )
            for key in (
                "rendered_context_sha256",
                "retrieved_record_identity_sequence_sha256",
                "retrieved_record_identity_multiset_sha256",
                "retrieved_active_score_sequence_sha256",
            ):
                if not _hex64(str(event.get(key, ""))):
                    raise ValueError(
                        f"Retrieval event fingerprint missing: {key}"
                    )
    context_payload = json.dumps(
        context_events,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    extraction_payload = json.dumps(
        extraction_events,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    classifier_scores = [
        score
        for event in extraction_events
        for score in event["classifier_strengths"]
    ]
    active_scores = [
        score
        for event in extraction_events
        for score in event["active_strengths"]
    ]
    record_counts = [
        int(event["records_emitted"]) for event in extraction_events
    ]
    classifier_summary = _score_summary(classifier_scores)
    active_summary = _score_summary(active_scores)
    internal_state_events = []
    for entry in sorted(
        result.get("stance_history", []),
        key=lambda item: int(item["round"]),
    ):
        if "Pro" not in entry or "Con" not in entry:
            raise ValueError("Internal-state trajectory is incomplete")
        pro = float(entry["Pro"])
        con = float(entry["Con"])
        if not math.isfinite(pro) or not math.isfinite(con):
            raise ValueError("Internal-state trajectory is non-finite")
        internal_state_events.append(
            {
                "round": int(entry["round"]),
                "Pro": pro,
                "Con": con,
            }
        )
    if len(internal_state_events) != 16:
        raise ValueError("Internal-state trajectory must contain 16 rows")
    state_payload = json.dumps(
        canonical_state_events,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    internal_payload = json.dumps(
        internal_state_events,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return {
        "dedup_dynamic_added": int(dynamic_status["added"]),
        "dedup_dynamic_ignored": int(dynamic_status["ignored"]),
        "dedup_dynamic_replaced": int(dynamic_status["replaced"]),
        "dedup_dynamic_other": int(
            sum(
                count
                for status, count in dynamic_status.items()
                if status not in {"added", "ignored", "replaced"}
            )
        ),
        "dedup_final_dynamic_total": final_dynamic_total,
        "dedup_final_dynamic_active": final_dynamic_active,
        "dedup_final_dynamic_archived": final_dynamic_archived,
        "retrieval_records_returned_total": retrieval_records_returned,
        "retrieval_context_event_count": len(context_events),
        "retrieval_context_sequence_sha256": hashlib.sha256(
            context_payload
        ).hexdigest(),
        "retrieval_events_json": context_payload.decode("utf-8"),
        "extraction_event_count": len(extraction_events),
        "extraction_records_emitted": sum(record_counts),
        "extraction_multi_record_messages": sum(
            count > 1 for count in record_counts
        ),
        "extraction_multi_record_fraction": (
            sum(count > 1 for count in record_counts) / len(record_counts)
            if record_counts
            else 0.0
        ),
        "extraction_record_count_mean": (
            statistics.fmean(record_counts) if record_counts else 0.0
        ),
        "extraction_record_count_sd": (
            statistics.stdev(record_counts)
            if len(record_counts) > 1
            else 0.0
        ),
        "extraction_record_count_q05": quantile(record_counts, 0.05),
        "extraction_record_count_q25": quantile(record_counts, 0.25),
        "extraction_record_count_q50": quantile(record_counts, 0.50),
        "extraction_record_count_q75": quantile(record_counts, 0.75),
        "extraction_record_count_q95": quantile(record_counts, 0.95),
        "extraction_events_json": extraction_payload.decode("utf-8"),
        "extraction_event_sequence_sha256": hashlib.sha256(
            extraction_payload
        ).hexdigest(),
        **{
            f"classifier_strength_{key}": value
            for key, value in classifier_summary.items()
        },
        **{
            f"active_strength_{key}": value
            for key, value in active_summary.items()
        },
        "canonical_dynamic_state_sequence_sha256": hashlib.sha256(
            state_payload
        ).hexdigest(),
        "internal_state_sequence_sha256": hashlib.sha256(
            internal_payload
        ).hexdigest(),
    }


def registry():
    with REGISTRY_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("trial", "seed", "S", "R", "P", "D", "E"):
            row[key] = int(row[key])
    return rows


def metrics_from_result(row, result):
    trajectory = []
    for entry in sorted(
        result["stance_history"],
        key=lambda item: int(item["round"]),
    ):
        if int(entry["round"]) == 0:
            pro, con = 0.75, -0.75
        else:
            pro = float(entry["Pro_evaluated"])
            con = float(entry["Con_evaluated"])
        trajectory.append((pro, con))
    if len(trajectory) != 16:
        raise ValueError(f"{row['run_id']}: incomplete trajectory")
    pro_final, con_final = trajectory[-1]
    gaps = [abs(pro - con) for pro, con in trajectory]
    pro_values = [pro for pro, _con in trajectory]
    con_values = [con for _pro, con in trajectory]
    return {
        **row,
        **component_telemetry(result),
        "pro_final": pro_final,
        "con_final": con_final,
        "final_gap": abs(pro_final - con_final),
        "trajectory_gap_auc": normalized_trapezoid(gaps),
        "mean_absolute_stance_shift": (
            abs(pro_final - 0.75) + abs(con_final + 0.75)
        ) / 2,
        "absolute_midpoint_shift": abs((pro_final + con_final) / 2),
        "same_side_final": int(pro_final * con_final > 0),
        "final_side_switch_count": int(pro_final < 0) + int(con_final > 0),
        "any_side_switch_count": (
            int(any(value < 0 for value in pro_values))
            + int(any(value > 0 for value in con_values))
        ),
    }


def _json_events(cell, field):
    events = json.loads(str(cell[field]))
    if not isinstance(events, list):
        raise ValueError(f"{field} is not an event list")
    return events


def _paired_extraction_diagnostics(left, right):
    left_events = _json_events(left, "extraction_events_json")
    right_events = _json_events(right, "extraction_events_json")
    if len(left_events) != len(right_events):
        raise ValueError("Paired extraction event counts differ")
    matched_messages = 0
    matched_records = 0
    classifier_score_diverged = 0
    active_score_diverged = 0
    active_ranking_diverged = 0
    for left_event, right_event in zip(left_events, right_events):
        if int(left_event["call_index"]) != int(right_event["call_index"]):
            raise ValueError("Paired extraction call indices differ")
        left_fingerprints = list(left_event["record_fingerprints"])
        right_fingerprints = list(right_event["record_fingerprints"])
        if left_fingerprints != right_fingerprints:
            continue
        matched_messages += 1
        matched_records += len(left_fingerprints)
        left_classifier = [
            float(value) for value in left_event["classifier_strengths"]
        ]
        right_classifier = [
            float(value) for value in right_event["classifier_strengths"]
        ]
        left_active = [
            float(value) for value in left_event["active_strengths"]
        ]
        right_active = [
            float(value) for value in right_event["active_strengths"]
        ]
        classifier_score_diverged += int(
            left_classifier != right_classifier
        )
        active_score_diverged += int(left_active != right_active)
        active_ranking_diverged += int(
            _rank_signature(left_fingerprints, left_active)
            != _rank_signature(right_fingerprints, right_active)
        )
    return {
        "identity_matched_messages": matched_messages,
        "identity_matched_records": matched_records,
        "classifier_score_diverged_matched_messages": (
            classifier_score_diverged
        ),
        "active_score_diverged_matched_messages": active_score_diverged,
        "active_ranking_diverged_matched_messages": active_ranking_diverged,
    }


def _paired_retrieval_diagnostics(left, right):
    left_events = _json_events(left, "retrieval_events_json")
    right_events = _json_events(right, "retrieval_events_json")
    if len(left_events) != len(right_events):
        raise ValueError("Paired retrieval event counts differ")
    same_multiset = 0
    ranking_diverged = 0
    score_sequence_diverged = 0
    rendered_context_diverged = 0
    for left_event, right_event in zip(left_events, right_events):
        identity = (
            str(left_event["agent"]),
            int(left_event["call_index"]),
        )
        if identity != (
            str(right_event["agent"]),
            int(right_event["call_index"]),
        ):
            raise ValueError("Paired retrieval event identities differ")
        same_records = (
            left_event["retrieved_record_identity_multiset_sha256"]
            == right_event[
                "retrieved_record_identity_multiset_sha256"
            ]
        )
        same_multiset += int(same_records)
        ranking_diverged += int(
            same_records
            and left_event[
                "retrieved_record_identity_sequence_sha256"
            ]
            != right_event[
                "retrieved_record_identity_sequence_sha256"
            ]
        )
        score_sequence_diverged += int(
            left_event["retrieved_active_score_sequence_sha256"]
            != right_event["retrieved_active_score_sequence_sha256"]
        )
        rendered_context_diverged += int(
            left_event["rendered_context_sha256"]
            != right_event["rendered_context_sha256"]
        )
    return {
        "retrieval_events": len(left_events),
        "retrieval_same_record_multiset_events": same_multiset,
        "retrieval_ranking_diverged_same_record_events": ranking_diverged,
        "retrieval_active_score_sequence_diverged_events": (
            score_sequence_diverged
        ),
        "rendered_context_diverged_events": rendered_context_diverged,
    }


def evidence_scoring_manipulation_rows(cells):
    """Pair E1 and E0 mechanism telemetry without stance outcomes."""
    indexed = {
        (row["topic_slug"], int(row["trial"]), row["condition"]): row
        for row in cells
    }
    output = []
    topics = sorted({row["topic_slug"] for row in cells})
    for topic_slug in topics:
        trials = sorted(
            int(row["trial"])
            for row in cells
            if row["topic_slug"] == topic_slug
            and row["condition"] == "be_s1_r1_p1_d1"
        )
        for trial in trials:
            e1 = indexed[(topic_slug, trial, "be_s1_r1_p1_d1")]
            e0 = indexed[
                (topic_slug, trial, "be_s1_r1_p1_d1_uniform")
            ]
            extraction_pair = _paired_extraction_diagnostics(e1, e0)
            retrieval_pair = _paired_retrieval_diagnostics(e1, e0)
            output.append(
                {
                    "topic": e1["topic"],
                    "topic_slug": topic_slug,
                    "trial": trial,
                    "seed": int(e1["seed"]),
                    "e1_dynamic_score_count": int(
                        e1["active_strength_count"]
                    ),
                    "e0_dynamic_score_count": int(
                        e0["active_strength_count"]
                    ),
                    "e1_dynamic_score_mean": e1[
                        "active_strength_mean"
                    ],
                    "e0_dynamic_score_mean": e0[
                        "active_strength_mean"
                    ],
                    "e1_dynamic_score_sd": e1["active_strength_sd"],
                    "e0_dynamic_score_sd": e0["active_strength_sd"],
                    "e1_dynamic_score_q05": e1[
                        "active_strength_q05"
                    ],
                    "e1_dynamic_score_q25": e1[
                        "active_strength_q25"
                    ],
                    "e1_dynamic_score_q50": e1[
                        "active_strength_q50"
                    ],
                    "e1_dynamic_score_q75": e1[
                        "active_strength_q75"
                    ],
                    "e1_dynamic_score_q95": e1[
                        "active_strength_q95"
                    ],
                    "e1_dynamic_score_mad_from_uniform": e1[
                        "active_strength_mean_absolute_deviation_from_uniform"
                    ],
                    "e0_dynamic_score_mad_from_uniform": e0[
                        "active_strength_mean_absolute_deviation_from_uniform"
                    ],
                    "e1_retained_classifier_score_mean": e1[
                        "classifier_strength_mean"
                    ],
                    "e0_retained_classifier_score_mean": e0[
                        "classifier_strength_mean"
                    ],
                    "e1_records_emitted": int(
                        e1["extraction_records_emitted"]
                    ),
                    "e0_records_emitted": int(
                        e0["extraction_records_emitted"]
                    ),
                    **extraction_pair,
                    "e1_dedup_replacements": int(
                        e1["dedup_dynamic_replaced"]
                    ),
                    "e0_dedup_replacements": int(
                        e0["dedup_dynamic_replaced"]
                    ),
                    "replacement_count_diverged": int(
                        int(e1["dedup_dynamic_replaced"])
                        != int(e0["dedup_dynamic_replaced"])
                    ),
                    **retrieval_pair,
                    "retrieval_context_sequence_diverged": int(
                        e1["retrieval_context_sequence_sha256"]
                        != e0["retrieval_context_sequence_sha256"]
                    ),
                    "canonical_dynamic_state_diverged": int(
                        e1[
                            "canonical_dynamic_state_sequence_sha256"
                        ]
                        != e0[
                            "canonical_dynamic_state_sequence_sha256"
                        ]
                    ),
                    "internal_state_sequence_diverged": int(
                        e1["internal_state_sequence_sha256"]
                        != e0["internal_state_sequence_sha256"]
                    ),
                }
            )
    if len(output) != 50:
        raise ValueError(
            f"Expected 50 evidence-scoring pairs, found {len(output)}"
        )
    return output


def _arm_scores(cells, condition, score_key):
    output = []
    selected = [
        row for row in cells if row["condition"] == condition
    ]
    if len(selected) != 50:
        raise ValueError(f"{condition}: expected 50 cells")
    for row in selected:
        for event in _json_events(row, "extraction_events_json"):
            output.extend(float(value) for value in event[score_key])
    return output


def summarize_evidence_scoring_manipulation(cells, rows):
    e1_active = _arm_scores(
        cells,
        "be_s1_r1_p1_d1",
        "active_strengths",
    )
    e0_active = _arm_scores(
        cells,
        "be_s1_r1_p1_d1_uniform",
        "active_strengths",
    )
    e1_classifier = _arm_scores(
        cells,
        "be_s1_r1_p1_d1",
        "classifier_strengths",
    )
    e0_classifier = _arm_scores(
        cells,
        "be_s1_r1_p1_d1_uniform",
        "classifier_strengths",
    )
    return {
        "schema_version": 1,
        "paired_units": len(rows),
        "uniform_strength_value": UNIFORM_STRENGTH_VALUE,
        "e1_realized_active_scores": _score_summary(e1_active),
        "e0_realized_active_scores": _score_summary(e0_active),
        "e1_retained_classifier_scores": _score_summary(e1_classifier),
        "e0_retained_classifier_scores": _score_summary(e0_classifier),
        "pairs_with_any_identity_matched_message": sum(
            int(row["identity_matched_messages"]) > 0 for row in rows
        ),
        "identity_matched_messages": sum(
            int(row["identity_matched_messages"]) for row in rows
        ),
        "identity_matched_records": sum(
            int(row["identity_matched_records"]) for row in rows
        ),
        "active_score_diverged_matched_messages": sum(
            int(row["active_score_diverged_matched_messages"])
            for row in rows
        ),
        "active_ranking_diverged_matched_messages": sum(
            int(row["active_ranking_diverged_matched_messages"])
            for row in rows
        ),
        "replacement_count_diverged_pairs": sum(
            int(row["replacement_count_diverged"]) for row in rows
        ),
        "retrieval_context_sequence_diverged_pairs": sum(
            int(row["retrieval_context_sequence_diverged"])
            for row in rows
        ),
        "canonical_dynamic_state_diverged_pairs": sum(
            int(row["canonical_dynamic_state_diverged"])
            for row in rows
        ),
        "internal_state_sequence_diverged_pairs": sum(
            int(row["internal_state_sequence_diverged"])
            for row in rows
        ),
        "continuation_rule": (
            "report_only_never_controls_campaign_continuation"
        ),
        "interpretation": (
            "E1 uses realized classifier scores; E0 retains the same score "
            "measurement but fixes the active update-time strength. Paired "
            "ranking statements are restricted to messages with identical "
            "record fingerprints."
        ),
    }


def structured_decomposition_manipulation_rows(cells):
    """Pair S1 atomic decomposition and S0 whole-message records."""
    indexed = {
        (row["topic_slug"], int(row["trial"]), row["condition"]): row
        for row in cells
    }
    output = []
    for topic_slug in sorted({row["topic_slug"] for row in cells}):
        for trial in (1, 2, 3, 4, 5):
            for r in (0, 1):
                for p in (0, 1):
                    s1 = indexed[
                        (
                            topic_slug,
                            trial,
                            core_condition(1, r, p),
                        )
                    ]
                    s0 = indexed[
                        (
                            topic_slug,
                            trial,
                            core_condition(0, r, p),
                        )
                    ]
                    retrieval_pair = _paired_retrieval_diagnostics(s1, s0)
                    output.append(
                        {
                            "topic": s1["topic"],
                            "topic_slug": topic_slug,
                            "trial": trial,
                            "seed": int(s1["seed"]),
                            "R": r,
                            "P": p,
                            "s1_messages": int(
                                s1["extraction_event_count"]
                            ),
                            "s1_claims": int(
                                s1["extraction_records_emitted"]
                            ),
                            "s1_claims_per_message_mean": s1[
                                "extraction_record_count_mean"
                            ],
                            "s1_multi_claim_messages": int(
                                s1["extraction_multi_record_messages"]
                            ),
                            "s1_multi_claim_fraction": s1[
                                "extraction_multi_record_fraction"
                            ],
                            "s0_messages": int(
                                s0["extraction_event_count"]
                            ),
                            "s0_records": int(
                                s0["extraction_records_emitted"]
                            ),
                            "s0_multi_record_messages": int(
                                s0["extraction_multi_record_messages"]
                            ),
                            "s1_final_dynamic_active": int(
                                s1["dedup_final_dynamic_active"]
                            ),
                            "s0_final_dynamic_active": int(
                                s0["dedup_final_dynamic_active"]
                            ),
                            "active_difference_s1_minus_s0": int(
                                s1["dedup_final_dynamic_active"]
                            )
                            - int(s0["dedup_final_dynamic_active"]),
                            "s1_final_dynamic_archived": int(
                                s1["dedup_final_dynamic_archived"]
                            ),
                            "s0_final_dynamic_archived": int(
                                s0["dedup_final_dynamic_archived"]
                            ),
                            "archived_difference_s1_minus_s0": int(
                                s1["dedup_final_dynamic_archived"]
                            )
                            - int(s0["dedup_final_dynamic_archived"]),
                            **retrieval_pair,
                            "retrieval_context_sequence_diverged": int(
                                s1["retrieval_context_sequence_sha256"]
                                != s0[
                                    "retrieval_context_sequence_sha256"
                                ]
                            ),
                            "canonical_dynamic_state_diverged": int(
                                s1[
                                    "canonical_dynamic_state_sequence_sha256"
                                ]
                                != s0[
                                    "canonical_dynamic_state_sequence_sha256"
                                ]
                            ),
                            "internal_state_sequence_diverged": int(
                                s1["internal_state_sequence_sha256"]
                                != s0["internal_state_sequence_sha256"]
                            ),
                        }
                    )
    if len(output) != 200:
        raise ValueError(
            f"Expected 200 structured-decomposition pairs, found {len(output)}"
        )
    return output


def summarize_structured_decomposition_manipulation(cells, rows):
    s1_counts = []
    s0_counts = []
    for row in cells:
        if row["condition"] not in set(CORE_CONDITIONS):
            continue
        destination = (
            s1_counts if int(row["S"]) == 1 else s0_counts
        )
        destination.extend(
            int(event["records_emitted"])
            for event in _json_events(row, "extraction_events_json")
        )
    def count_summary(values):
        return {
            "messages": len(values),
            "records": sum(values),
            "mean": statistics.fmean(values),
            "sd": statistics.stdev(values),
            "q05": quantile(values, 0.05),
            "q25": quantile(values, 0.25),
            "q50": quantile(values, 0.50),
            "q75": quantile(values, 0.75),
            "q95": quantile(values, 0.95),
            "messages_with_more_than_one_record": sum(
                value > 1 for value in values
            ),
            "fraction_with_more_than_one_record": (
                sum(value > 1 for value in values) / len(values)
            ),
        }
    retrieval_on = [row for row in rows if int(row["R"]) == 1]
    retrieval_off = [row for row in rows if int(row["R"]) == 0]
    return {
        "schema_version": 1,
        "paired_units": len(rows),
        "factor_label": (
            "structured_atomic_decomposition_vs_verbatim_whole_message"
        ),
        "s1_claims_per_message": count_summary(s1_counts),
        "s0_records_per_message": count_summary(s0_counts),
        "mean_active_difference_s1_minus_s0": statistics.fmean(
            int(row["active_difference_s1_minus_s0"]) for row in rows
        ),
        "mean_archived_difference_s1_minus_s0": statistics.fmean(
            int(row["archived_difference_s1_minus_s0"]) for row in rows
        ),
        "retrieval_on_context_diverged_pairs": sum(
            int(row["retrieval_context_sequence_diverged"])
            for row in retrieval_on
        ),
        "retrieval_on_pairs": len(retrieval_on),
        "retrieval_off_context_diverged_pairs": sum(
            int(row["retrieval_context_sequence_diverged"])
            for row in retrieval_off
        ),
        "retrieval_off_pairs": len(retrieval_off),
        "canonical_dynamic_state_diverged_pairs": sum(
            int(row["canonical_dynamic_state_diverged"])
            for row in rows
        ),
        "internal_state_sequence_diverged_pairs": sum(
            int(row["internal_state_sequence_diverged"])
            for row in rows
        ),
        "continuation_rule": (
            "report_only_never_controls_campaign_continuation"
        ),
        "interpretation": (
            "S1 versus S0 tests structured atomic decomposition, not memory "
            "versus no memory. Both arms retain update metadata."
        ),
    }


def deduplication_manipulation_rows(cells):
    """Pair D1 and D0 telemetry without consulting stance outcomes."""
    indexed = {
        (row["topic_slug"], row["trial"], row["condition"]): row
        for row in cells
    }
    output = []
    for topic_slug in sorted({row["topic_slug"] for row in cells}):
        trials = sorted(
            {
                int(row["trial"])
                for row in cells
                if row["topic_slug"] == topic_slug
            }
        )
        for trial in trials:
            d1 = indexed[(topic_slug, trial, "be_s1_r1_p1_d1")]
            d0 = indexed[(topic_slug, trial, "be_s1_r1_p1_d0")]
            d1_triggered = (
                int(d1["dedup_dynamic_ignored"])
                + int(d1["dedup_dynamic_replaced"])
            )
            d0_triggered = (
                int(d0["dedup_dynamic_ignored"])
                + int(d0["dedup_dynamic_replaced"])
            )
            output.append(
                {
                    "topic": d1["topic"],
                    "topic_slug": topic_slug,
                    "trial": trial,
                    "seed": int(d1["seed"]),
                    "d1_dynamic_added": int(
                        d1["dedup_dynamic_added"]
                    ),
                    "d1_dynamic_ignored": int(
                        d1["dedup_dynamic_ignored"]
                    ),
                    "d1_dynamic_replaced": int(
                        d1["dedup_dynamic_replaced"]
                    ),
                    "d1_dynamic_other": int(
                        d1["dedup_dynamic_other"]
                    ),
                    "d1_triggered_decisions": d1_triggered,
                    "d0_dynamic_added": int(
                        d0["dedup_dynamic_added"]
                    ),
                    "d0_dynamic_ignored": int(
                        d0["dedup_dynamic_ignored"]
                    ),
                    "d0_dynamic_replaced": int(
                        d0["dedup_dynamic_replaced"]
                    ),
                    "d0_dynamic_other": int(
                        d0["dedup_dynamic_other"]
                    ),
                    "d0_triggered_decisions": d0_triggered,
                    "d1_final_dynamic_active": int(
                        d1["dedup_final_dynamic_active"]
                    ),
                    "d0_final_dynamic_active": int(
                        d0["dedup_final_dynamic_active"]
                    ),
                    "active_difference_d0_minus_d1": int(
                        d0["dedup_final_dynamic_active"]
                    )
                    - int(d1["dedup_final_dynamic_active"]),
                    "d1_final_dynamic_archived": int(
                        d1["dedup_final_dynamic_archived"]
                    ),
                    "d0_final_dynamic_archived": int(
                        d0["dedup_final_dynamic_archived"]
                    ),
                    "archived_difference_d1_minus_d0": int(
                        d1["dedup_final_dynamic_archived"]
                    )
                    - int(d0["dedup_final_dynamic_archived"]),
                    "d1_retrieval_records_returned_total": int(
                        d1["retrieval_records_returned_total"]
                    ),
                    "d0_retrieval_records_returned_total": int(
                        d0["retrieval_records_returned_total"]
                    ),
                    "retrieval_records_difference_d0_minus_d1": int(
                        d0["retrieval_records_returned_total"]
                    )
                    - int(d1["retrieval_records_returned_total"]),
                    "d1_retrieval_context_sequence_sha256": d1[
                        "retrieval_context_sequence_sha256"
                    ],
                    "d0_retrieval_context_sequence_sha256": d0[
                        "retrieval_context_sequence_sha256"
                    ],
                    "retrieval_context_sequence_diverged": int(
                        d1["retrieval_context_sequence_sha256"]
                        != d0["retrieval_context_sequence_sha256"]
                    ),
                    "canonical_dynamic_state_diverged": int(
                        d1[
                            "canonical_dynamic_state_sequence_sha256"
                        ]
                        != d0[
                            "canonical_dynamic_state_sequence_sha256"
                        ]
                    ),
                    "internal_state_sequence_diverged": int(
                        d1["internal_state_sequence_sha256"]
                        != d0["internal_state_sequence_sha256"]
                    ),
                }
            )
    return output


def summarize_deduplication_manipulation(rows):
    paired_units = len(rows)
    if paired_units != 50:
        raise ValueError(
            f"Expected 50 deduplication pairs, found {paired_units}"
        )
    pairs_with_trigger = sum(
        int(row["d1_triggered_decisions"]) > 0 for row in rows
    )
    context_diverged = sum(
        int(row["retrieval_context_sequence_diverged"]) for row in rows
    )
    d0_triggered = sum(
        int(row["d0_triggered_decisions"]) for row in rows
    )
    return {
        "schema_version": 1,
        "paired_units": paired_units,
        "d1_total_dynamic_added": sum(
            int(row["d1_dynamic_added"]) for row in rows
        ),
        "d1_total_dynamic_ignored": sum(
            int(row["d1_dynamic_ignored"]) for row in rows
        ),
        "d1_total_dynamic_replaced": sum(
            int(row["d1_dynamic_replaced"]) for row in rows
        ),
        "d1_total_dynamic_other": sum(
            int(row["d1_dynamic_other"]) for row in rows
        ),
        "d1_pairs_with_any_triggered_decision": pairs_with_trigger,
        "d1_pair_exposure_fraction": pairs_with_trigger / paired_units,
        "d0_total_triggered_decisions": d0_triggered,
        "mean_active_difference_d0_minus_d1": statistics.fmean(
            int(row["active_difference_d0_minus_d1"]) for row in rows
        ),
        "mean_archived_difference_d1_minus_d0": statistics.fmean(
            int(row["archived_difference_d1_minus_d0"]) for row in rows
        ),
        "mean_retrieval_records_difference_d0_minus_d1": (
            statistics.fmean(
                int(row["retrieval_records_difference_d0_minus_d1"])
                for row in rows
            )
        ),
        "retrieval_context_sequence_diverged_pairs": context_diverged,
        "retrieval_context_sequence_diverged_fraction": (
            context_diverged / paired_units
        ),
        "canonical_dynamic_state_diverged_pairs": sum(
            int(row["canonical_dynamic_state_diverged"]) for row in rows
        ),
        "internal_state_sequence_diverged_pairs": sum(
            int(row["internal_state_sequence_diverged"]) for row in rows
        ),
        "natural_deduplication_exposure_observed": (
            pairs_with_trigger > 0
        ),
        "continuation_rule": (
            "report_only_never_controls_campaign_continuation"
        ),
        "interpretation": (
            "This is realized manipulation exposure, not an outcome or a "
            "deduplication-accuracy estimate."
        ),
    }


def load_cells(campaign, rows):
    cells = []
    inputs = []
    for row in rows:
        path = (
            campaign
            / "raw_runs"
            / row["topic_slug"]
            / row["run_id"]
            / "results.json"
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        result = read_json(path)
        cells.append(metrics_from_result(row, result))
        inputs.append(path)
    return cells, inputs


def core_condition(s, r, p):
    return f"be_s{s}_r{r}_p{p}_d1"


def contrast_weights(name):
    weights = {}
    if name == "structured_representation_main":
        for r in (0, 1):
            for p in (0, 1):
                weights[core_condition(0, r, p)] = 0.25
                weights[core_condition(1, r, p)] = -0.25
    elif name == "response_retrieval_main":
        for s in (0, 1):
            for p in (0, 1):
                weights[core_condition(s, 0, p)] = 0.25
                weights[core_condition(s, 1, p)] = -0.25
    elif name == "persistence_main":
        for s in (0, 1):
            for r in (0, 1):
                weights[core_condition(s, r, 0)] = 0.25
                weights[core_condition(s, r, 1)] = -0.25
    elif name == "deduplication_at_full_setting":
        weights = {"be_s1_r1_p1_d0": 1.0, "be_s1_r1_p1_d1": -1.0}
    elif name == "graded_evidence_scoring_at_full_setting":
        weights = {
            "be_s1_r1_p1_d1_uniform": 1.0,
            "be_s1_r1_p1_d1": -1.0,
        }
    elif name == "structured_update_path_without_response_retrieval":
        for p in (0, 1):
            weights[core_condition(0, 0, p)] = 0.5
            weights[core_condition(1, 0, p)] = -0.5
    elif name == "structured_effect_with_response_retrieval":
        for p in (0, 1):
            weights[core_condition(0, 1, p)] = 0.5
            weights[core_condition(1, 1, p)] = -0.5
    elif name == "representation_by_retrieval":
        for p in (0, 1):
            weights[core_condition(0, 1, p)] = 0.5
            weights[core_condition(1, 1, p)] = -0.5
            weights[core_condition(0, 0, p)] = -0.5
            weights[core_condition(1, 0, p)] = 0.5
    elif name == "representation_by_persistence":
        for r in (0, 1):
            weights[core_condition(0, r, 1)] = 0.5
            weights[core_condition(1, r, 1)] = -0.5
            weights[core_condition(0, r, 0)] = -0.5
            weights[core_condition(1, r, 0)] = 0.5
    elif name == "retrieval_by_persistence":
        for s in (0, 1):
            weights[core_condition(s, 0, 0)] = 0.5
            weights[core_condition(s, 1, 0)] = -0.5
            weights[core_condition(s, 0, 1)] = -0.5
            weights[core_condition(s, 1, 1)] = 0.5
    elif name == "representation_by_retrieval_by_persistence":
        for s in (0, 1):
            for r in (0, 1):
                for p in (0, 1):
                    coefficient = (
                        (1 if s == 0 else -1)
                        * (1 if r == 1 else -1)
                        * (1 if p == 1 else -1)
                    )
                    weights[core_condition(s, r, p)] = coefficient
    else:
        raise KeyError(name)
    if abs(sum(weights.values())) > 1e-12:
        raise ValueError(f"{name}: contrast weights do not sum to zero")
    return weights


def unit_contrasts(cells, contrast_names):
    indexed = {
        (row["topic_slug"], row["trial"], row["condition"]): row
        for row in cells
    }
    units = []
    topics = sorted({row["topic_slug"] for row in cells})
    for topic in topics:
        trials = sorted(
            {
                row["trial"]
                for row in cells
                if row["topic_slug"] == topic
            }
        )
        for trial in trials:
            block = {
                condition: indexed[(topic, trial, condition)]
                for condition in {
                    row["condition"]
                    for row in cells
                    if row["topic_slug"] == topic and row["trial"] == trial
                }
            }
            if len(block) != 10:
                raise ValueError(f"{topic} trial {trial}: incomplete block")
            for contrast in contrast_names:
                weights = contrast_weights(contrast)
                for metric in METRICS:
                    value = sum(
                        weight * float(block[condition][metric])
                        for condition, weight in weights.items()
                    )
                    units.append(
                        {
                            "topic_slug": topic,
                            "topic": next(iter(block.values()))["topic"],
                            "trial": trial,
                            "seed": next(iter(block.values()))["seed"],
                            "contrast": contrast,
                            "metric": metric,
                            "difference": value,
                        }
                    )
    return units


def hierarchical_draws(rows, repetitions, seed):
    by_topic = defaultdict(list)
    for row in rows:
        by_topic[row["topic_slug"]].append(float(row["difference"]))
    topics = sorted(by_topic)
    rng = random.Random(seed)
    draws = []
    for _ in range(repetitions):
        sampled = []
        for _ in topics:
            topic = rng.choice(topics)
            values = by_topic[topic]
            sampled.extend(rng.choice(values) for _ in values)
        draws.append(statistics.fmean(sampled))
    return draws


def bootstrap_seed(base_seed, contrast, metric):
    payload = (
        f"component_necessity_v1::{base_seed}::{contrast}::{metric}"
    ).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def summarize_contrasts(units, repetitions, seed):
    grouped = defaultdict(list)
    for row in units:
        grouped[(row["contrast"], row["metric"])].append(row)
    summaries = []
    topics = []
    for (contrast, metric), rows in sorted(grouped.items()):
        observed = statistics.fmean(row["difference"] for row in rows)
        draw_seed = bootstrap_seed(seed, contrast, metric)
        draws = hierarchical_draws(rows, repetitions, draw_seed)
        summaries.append(
            {
                "contrast": contrast,
                "metric": metric,
                "n_paired_units": len(rows),
                "estimate": observed,
                "ci95_low": quantile(draws, 0.025),
                "ci95_high": quantile(draws, 0.975),
                "familywise_ci99_low": (
                    quantile(draws, 0.005) if contrast in PRIMARY else ""
                ),
                "familywise_ci99_high": (
                    quantile(draws, 0.995) if contrast in PRIMARY else ""
                ),
                "bootstrap_repetitions": repetitions,
                "bootstrap_seed": draw_seed,
            }
        )
        by_topic = defaultdict(list)
        names = {}
        for row in rows:
            by_topic[row["topic_slug"]].append(row["difference"])
            names[row["topic_slug"]] = row["topic"]
        for topic_slug, values in sorted(by_topic.items()):
            mean = statistics.fmean(values)
            topics.append(
                {
                    "contrast": contrast,
                    "metric": metric,
                    "topic": names[topic_slug],
                    "topic_slug": topic_slug,
                    "n_trials": len(values),
                    "estimate": mean,
                    "sign": "positive" if mean > 0 else ("negative" if mean < 0 else "zero"),
                }
            )
    return summaries, topics


def exact_shared_topic_signflip_max_family(units):
    """Exact 2^10 shared-vector sign-flip sensitivity for primary gaps.

    The same topic sign is applied to all five contrast coordinates, retaining
    their within-topic dependence. The registered hierarchical bootstrap and
    family-wise intervals remain the primary inference.
    """
    topic_values = defaultdict(lambda: defaultdict(list))
    topic_names = {}
    for row in units:
        if (
            row["contrast"] in PRIMARY
            and row["metric"] == "final_gap"
        ):
            topic_values[row["topic_slug"]][row["contrast"]].append(
                float(row["difference"])
            )
            topic_names[row["topic_slug"]] = row["topic"]
    topics = sorted(topic_values)
    if len(topics) != 10:
        raise ValueError("Exact sign-flip sensitivity requires ten topics")
    vectors = {
        contrast: [
            statistics.fmean(topic_values[topic][contrast])
            for topic in topics
        ]
        for contrast in PRIMARY
    }
    if any(len(values) != 10 for values in vectors.values()):
        raise ValueError("Incomplete topic vector for exact sign flips")
    denominators = {
        contrast: math.sqrt(sum(value * value for value in values))
        for contrast, values in vectors.items()
    }
    observed = {
        contrast: (
            abs(sum(values)) / denominators[contrast]
            if denominators[contrast] > 0.0
            else 0.0
        )
        for contrast, values in vectors.items()
    }
    marginal_counts = {contrast: 0 for contrast in PRIMARY}
    max_counts = {contrast: 0 for contrast in PRIMARY}
    tolerance = 1e-15
    for mask in range(1 << len(topics)):
        signs = [
            1.0 if mask & (1 << index) else -1.0
            for index in range(len(topics))
        ]
        statistics_by_contrast = {}
        for contrast in PRIMARY:
            numerator = abs(
                sum(
                    sign * value
                    for sign, value in zip(
                        signs,
                        vectors[contrast],
                    )
                )
            )
            statistics_by_contrast[contrast] = (
                numerator / denominators[contrast]
                if denominators[contrast] > 0.0
                else 0.0
            )
        maximum = max(statistics_by_contrast.values())
        for contrast in PRIMARY:
            marginal_counts[contrast] += int(
                statistics_by_contrast[contrast]
                >= observed[contrast] - tolerance
            )
            max_counts[contrast] += int(
                maximum >= observed[contrast] - tolerance
            )
    rows = []
    for contrast in PRIMARY:
        values = vectors[contrast]
        estimate = statistics.fmean(values)
        rows.append(
            {
                "contrast": contrast,
                "metric": "final_gap",
                "n_topics": len(topics),
                "topic_trials_averaged": 5,
                "estimate": estimate,
                "positive_topic_estimates": sum(
                    value > 0 for value in values
                ),
                "negative_topic_estimates": sum(
                    value < 0 for value in values
                ),
                "zero_topic_estimates": sum(
                    value == 0 for value in values
                ),
                "self_normalized_absolute_statistic": observed[contrast],
                "exact_marginal_signflip_p": (
                    marginal_counts[contrast] / EXACT_SIGN_PATTERNS
                ),
                "exact_max_family_signflip_p": (
                    max_counts[contrast] / EXACT_SIGN_PATTERNS
                ),
                "enumerated_shared_sign_vectors": EXACT_SIGN_PATTERNS,
            }
        )
    metadata = {
        "schema_version": 1,
        "method": (
            "exact_shared_topic_vector_signflip_with_max_family_adjustment"
        ),
        "metric": "final_gap",
        "topics": topics,
        "topic_names": [topic_names[topic] for topic in topics],
        "contrasts": list(PRIMARY),
        "enumerated_shared_sign_vectors": EXACT_SIGN_PATTERNS,
        "statistic": "absolute_sum_divided_by_root_sum_squares",
        "assumptions": [
            "topic vectors are independent units",
            "each joint five-contrast topic vector is sign-symmetric under the joint null",
            "the same sign is applied to all five coordinates within a topic",
        ],
        "status": (
            "offline_sensitivity_registered_hierarchical_bootstrap_remains_primary"
        ),
    }
    return rows, metadata


def condition_summary(cells):
    grouped = defaultdict(list)
    for row in cells:
        grouped[row["condition"]].append(row)
    output = []
    for condition, rows in sorted(grouped.items()):
        item = {"condition": condition, "n": len(rows)}
        for metric in METRICS:
            values = [float(row[metric]) for row in rows]
            item[f"{metric}_mean"] = statistics.fmean(values)
            item[f"{metric}_sd"] = statistics.stdev(values)
        output.append(item)
    return output


def leave_one_topic_out(units):
    output = []
    all_topics = sorted({row["topic_slug"] for row in units})
    for contrast in PRIMARY:
        for metric in ("final_gap", "trajectory_gap_auc"):
            subset = [
                row for row in units
                if row["contrast"] == contrast and row["metric"] == metric
            ]
            for omitted in all_topics:
                kept = [
                    row["difference"] for row in subset
                    if row["topic_slug"] != omitted
                ]
                output.append(
                    {
                        "contrast": contrast,
                        "metric": metric,
                        "omitted_topic_slug": omitted,
                        "n_paired_units": len(kept),
                        "estimate": statistics.fmean(kept),
                    }
                )
    return output


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report(
    primary_rows,
    topic_rows,
    dedup_summary,
    scoring_summary,
    structured_summary,
    exact_rows,
):
    def format_score(value):
        return "NA" if value is None else f"{float(value):.3f}"

    final = {
        row["contrast"]: row
        for row in primary_rows
        if row["metric"] == "final_gap" and row["contrast"] in PRIMARY
    }
    lines = [
        "# Component-necessity factorial",
        "",
        "All outcomes use the same external evaluator. Positive primary final-gap "
        "contrasts mean the named component reduced expressed disagreement.",
        "",
        "| Primary contrast | Estimate | 95% CI | family-wise 99% CI |",
        "|---|---:|---:|---:|",
    ]
    for name in PRIMARY:
        row = final[name]
        lines.append(
            f"| {name} | {row['estimate']:+.3f} | "
            f"[{row['ci95_low']:+.3f}, {row['ci95_high']:+.3f}] | "
            f"[{row['familywise_ci99_low']:+.3f}, "
            f"{row['familywise_ci99_high']:+.3f}] |"
        )
    lines.extend(
        [
            "",
            "The representation×retrieval interaction separates the incremental "
            "prompt-exposure pathway from the representation effect that remains "
            "when response retrieval is disabled. Deduplication is conditional "
            "on the full S1/R1/P1 setting; no dedup interaction is claimed.",
            "",
            "Every topic estimate and sign is available in `topic_contrasts.csv`.",
            "",
            "## Deduplication manipulation exposure",
            "",
            "The D1 gate made at least one dynamic ignore/replace decision in "
            f"{dedup_summary['d1_pairs_with_any_triggered_decision']}/"
            f"{dedup_summary['paired_units']} paired cells "
            f"({dedup_summary['d1_pair_exposure_fraction']:.1%}). "
            "The D0 arm made "
            f"{dedup_summary['d0_total_triggered_decisions']} such decisions. "
            "Hashed retrieved-context sequences differed in "
            f"{dedup_summary['retrieval_context_sequence_diverged_pairs']}/"
            f"{dedup_summary['paired_units']} pairs. These are report-only "
            "manipulation checks; they never control continuation. If natural "
            "D1 exposure is sparse, a null stance contrast is a weak-exposure "
            "result rather than evidence that deduplication is unnecessary.",
            "",
            "## Evidence-scoring manipulation",
            "",
            "The graded E1 arm realized "
            f"{scoring_summary['e1_realized_active_scores']['count']} "
            "dynamic scores with mean "
            f"{format_score(scoring_summary['e1_realized_active_scores']['mean'])}, "
            "SD "
            f"{format_score(scoring_summary['e1_realized_active_scores']['sd'])}, "
            "and mean absolute deviation "
            f"{format_score(scoring_summary['e1_realized_active_scores']['mean_absolute_deviation_from_uniform'])} "
            "from the frozen scale-matched value. The E0 arm's realized "
            "active-score SD and deviation were "
            f"{format_score(scoring_summary['e0_realized_active_scores']['sd'])} "
            "and "
            f"{format_score(scoring_summary['e0_realized_active_scores']['mean_absolute_deviation_from_uniform'])}. "
            "Ranking comparisons are made only for messages whose hashed "
            "record identities match across arms. All checks are report-only.",
            "",
            "## Structured-decomposition manipulation",
            "",
            "S1 emitted a mean of "
            f"{structured_summary['s1_claims_per_message']['mean']:.2f} "
            "atomic claims per message; "
            f"{structured_summary['s1_claims_per_message']['fraction_with_more_than_one_record']:.1%} "
            "of messages produced more than one claim. S0 retained exactly "
            "one verbatim whole-message record per message if its registered "
            "contract passed. This is structured decomposition versus a "
            "verbatim record, not memory versus no memory.",
            "",
            "## Exact topic-vector sensitivity",
            "",
            "The registered hierarchical bootstrap and family-wise 99% "
            "intervals above remain primary. As an offline sensitivity check, "
            "we also enumerated all 1,024 shared sign vectors over the ten "
            "topic-level five-contrast vectors and used the maximum statistic "
            "for family adjustment.",
            "",
            "| Primary contrast | exact marginal p | exact max-family p |",
            "|---|---:|---:|",
        ]
    )
    for row in exact_rows:
        lines.append(
            f"| {row['contrast']} | "
            f"{row['exact_marginal_signflip_p']:.4f} | "
            f"{row['exact_max_family_signflip_p']:.4f} |"
        )
    lines.extend(
        [
            "",
            "This exact sensitivity assumes independent topic vectors and "
            "joint sign symmetry under the null; it is not a randomized-trial "
            "test and does not replace the registered bootstrap.",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=100000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260725)
    args = parser.parse_args()
    if args.bootstrap_repetitions < 10000:
        raise SystemExit("At least 10,000 bootstrap repetitions are required.")
    output = args.output_dir.resolve()
    if output.exists():
        raise SystemExit(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)

    rows = registry()
    if len(rows) != 500:
        raise ValueError(f"Expected 500 registered cells, found {len(rows)}")
    cells, inputs = load_cells(args.campaign.resolve(), rows)
    contrasts = PRIMARY + SECONDARY
    units = unit_contrasts(cells, contrasts)
    summaries, topics = summarize_contrasts(
        units,
        args.bootstrap_repetitions,
        args.bootstrap_seed,
    )
    conditions = condition_summary(cells)
    loto = leave_one_topic_out(units)
    dedup_rows = deduplication_manipulation_rows(cells)
    dedup_summary = summarize_deduplication_manipulation(dedup_rows)
    scoring_rows = evidence_scoring_manipulation_rows(cells)
    scoring_summary = summarize_evidence_scoring_manipulation(
        cells,
        scoring_rows,
    )
    structured_rows = structured_decomposition_manipulation_rows(cells)
    structured_summary = (
        summarize_structured_decomposition_manipulation(
            cells,
            structured_rows,
        )
    )
    exact_rows, exact_metadata = (
        exact_shared_topic_signflip_max_family(units)
    )
    write_csv(output / "cell_outcomes.csv", cells)
    write_csv(output / "condition_summary.csv", conditions)
    write_csv(output / "unit_contrasts.csv", units)
    write_csv(
        output / "primary_contrasts.csv",
        [row for row in summaries if row["contrast"] in PRIMARY],
    )
    write_csv(
        output / "secondary_contrasts.csv",
        [row for row in summaries if row["contrast"] in SECONDARY],
    )
    write_csv(output / "topic_contrasts.csv", topics)
    write_csv(output / "leave_one_topic_out.csv", loto)
    write_csv(
        output / "deduplication_manipulation_checks.csv",
        dedup_rows,
    )
    (output / "deduplication_manipulation_summary.json").write_text(
        json.dumps(dedup_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(
        output / "evidence_scoring_manipulation_checks.csv",
        scoring_rows,
    )
    (output / "evidence_scoring_manipulation_summary.json").write_text(
        json.dumps(scoring_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(
        output / "structured_decomposition_manipulation_checks.csv",
        structured_rows,
    )
    (
        output / "structured_decomposition_manipulation_summary.json"
    ).write_text(
        json.dumps(structured_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(
        output / "primary_topic_vector_signflip_max_family.csv",
        exact_rows,
    )
    (output / "primary_topic_vector_signflip_metadata.json").write_text(
        json.dumps(exact_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "REPORT.md").write_text(
        report(
            summaries,
            topics,
            dedup_summary,
            scoring_summary,
            structured_summary,
            exact_rows,
        ),
        encoding="utf-8",
    )
    summary = {
        "schema_version": 1,
        "experiment_id": "neurips_2026_component_necessity_factorial_v1",
        "cells": len(cells),
        "paired_units": len(
            {(row["topic_slug"], row["trial"]) for row in cells}
        ),
        "topics": len({row["topic_slug"] for row in cells}),
        "primary_final_gap": {
            row["contrast"]: {
                key: row[key]
                for key in (
                    "estimate",
                    "ci95_low",
                    "ci95_high",
                    "familywise_ci99_low",
                    "familywise_ci99_high",
                )
            }
            for row in summaries
            if row["metric"] == "final_gap" and row["contrast"] in PRIMARY
        },
        "deduplication_manipulation": dedup_summary,
        "evidence_scoring_manipulation": scoring_summary,
        "structured_decomposition_manipulation": structured_summary,
        "primary_topic_vector_signflip_max_family": exact_rows,
        "primary_topic_vector_signflip_metadata": exact_metadata,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "analysis_script": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
        "analysis_script_sha256": file_sha256(Path(__file__).resolve()),
        "spec_sha256": file_sha256(SPEC_PATH),
        "registry_sha256": file_sha256(REGISTRY_PATH),
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "bootstrap_seed": args.bootstrap_seed,
        "inputs": [
            {
                "path": str(path.resolve().relative_to(REPO_ROOT)),
                "sha256": file_sha256(path),
            }
            for path in inputs
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote verified analysis inputs and outputs to {output}.")


if __name__ == "__main__":
    main()
