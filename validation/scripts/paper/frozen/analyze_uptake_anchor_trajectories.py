#!/usr/bin/env python3
"""Group-aware fixed-stream anchoring-by-uptake trajectory analysis.

This is a transparent post-hoc robustness analysis over the verifier-approved
DEBATE component-ablation outputs. It never calls a model, reruns extraction,
or changes the frozen accepted-evidence streams.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import random
import shutil
import statistics
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence


REBUTTAL_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = REBUTTAL_ROOT / "results" / "debate_component_ablation_v1"
DEFAULT_SPEC = REBUTTAL_ROOT / "UPTAKE_ANCHOR_TRAJECTORY_ANALYSIS_SPEC_V2.yaml"
DEFAULT_OUTPUT_DIR = (
    REBUTTAL_ROOT
    / "results"
    / "analyses"
    / "uptake_anchor_trajectory_posthoc_v2"
)

EXPECTED_ACCEPTED_SHA256 = (
    "ecb79b76738c7bf6ada4dbaba8fd1f8dc836cc3c2f11d61c796fee297e6e491e"
)
EXPECTED_PREDICTIONS_SHA256 = (
    "d22de2566db26ee9fcbcf85f68423bbd51163c830b643f633fbe9646168ab943"
)
EXPECTED_SPEC_SHA256 = (
    "60e96c05e24ad324ca059b04851cca4e0ce2782343e280dc00f06528976382fd"
)
EXPECTED_PARTICIPANTS = 2495
EXPECTED_PREDICTION_ROWS = 29940
EXPECTED_ACCEPTED_ROWS = 21695
EXPECTED_NONEMPTY_STREAMS = 2494
EXPECTED_EVIDENCE_STREAM_IDS = 2482
EXPECTED_GROUPS = 704
EXPECTED_TOPICS = 107

LOW_ANCHOR = 0.2
HIGH_ANCHOR = 1.0
LOW_UPTAKE = 0.05
HIGH_UPTAKE = 0.6
ENDPOINT_TOLERANCES = (0.01, 0.02, 0.05)
PRIMARY_ENDPOINT_TOLERANCE = 0.02
SEPARATION_THRESHOLDS = (0.1, 0.2)
MIN_ARGUMENTS = 3
SATURATION_BOUND = 0.95
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260724
STABLE_TOLERANCE = 1e-12
WEAK_PRESSURE_THRESHOLD = 1.0

UPDATE_GRID = (0.005, 0.01, 0.02, 0.035, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8)
ANCHOR_GRID = (0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5)
GRID_MIN_ANCHOR_DIFFERENCE = 0.4
GRID_MIN_UPTAKE_DIFFERENCE = 0.2

STRATA = (
    "evidence_aligned_mover",
    "evidence_opposed_mover",
    "weak_signal_mover",
    "stable_despite_pressure",
    "stable_low_pressure",
)


class AnalysisError(RuntimeError):
    """Raised when an input or analysis invariant fails."""


@dataclass(frozen=True)
class Evidence:
    sequence_index: int
    sign: int
    strength: float


@dataclass(frozen=True)
class Participant:
    participant_record_id: str
    evidence_stream_id: str
    group_id: str
    topic_id: str
    initial_stance: float
    opinion_change_actual: float
    evidence: tuple[Evidence, ...]
    evidence_pressure: float
    evidence_relation: str
    uptake_leverage_orientation: str
    initial_stance_cell: str
    prior_extremity: str
    response_stratum: str


@dataclass(frozen=True)
class Trajectory:
    anchor: float
    uptake: float
    stances: tuple[float, ...]
    log_odds: tuple[float, ...]

    @property
    def initial(self) -> float:
        return self.stances[0]

    @property
    def final(self) -> float:
        return self.stances[-1]

    @property
    def signed_shift(self) -> float:
        return self.final - self.initial

    @property
    def absolute_shift(self) -> float:
        return abs(self.signed_shift)

    @property
    def mean_absolute_step(self) -> float:
        if len(self.stances) == 1:
            return 0.0
        return statistics.fmean(
            abs(right - left)
            for left, right in zip(self.stances, self.stances[1:])
        )

    @property
    def trajectory_mean(self) -> float:
        return statistics.fmean(self.stances)

    @property
    def first_switch_index(self) -> int | None:
        start_sign = signum(self.initial)
        if start_sign == 0:
            return None
        for index, value in enumerate(self.stances[1:], start=1):
            if signum(value) != start_sign:
                return index
        return None


@dataclass(frozen=True)
class EndpointRow:
    comparison: str
    participant_record_id: str
    group_id: str
    topic_id: str
    response_stratum: str
    evidence_relation: str
    uptake_leverage_orientation: str
    initial_stance_cell: str
    prior_extremity: str
    evidence_count: int
    final_a: float
    final_b: float
    final_gap: float
    initial_gap: float
    maximum_trajectory_gap: float
    mean_trajectory_gap: float
    maximum_interior_trajectory_gap: float
    mean_interior_trajectory_gap: float
    switch_a: int | None
    switch_b: int | None

    @property
    def switch_discordant(self) -> bool:
        return (self.switch_a is None) != (self.switch_b is None)

    @property
    def switch_gap(self) -> float | None:
        if self.switch_a is None or self.switch_b is None:
            return None
        return float(abs(self.switch_a - self.switch_b))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def signum(value: float, tolerance: float = 1e-15) -> int:
    if value > tolerance:
        return 1
    if value < -tolerance:
        return -1
    return 0


def stance_to_log_odds(stance: float) -> float:
    bounded = min(0.999, max(-0.999, float(stance)))
    return math.log((1.0 + bounded) / (1.0 - bounded))


def log_odds_to_stance(log_odds: float) -> float:
    return math.tanh(log_odds / 2.0)


def percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise AnalysisError("Cannot take a percentile of an empty sample.")
    ordered = sorted(values)
    location = (len(ordered) - 1) * probability
    lower = math.floor(location)
    upper = math.ceil(location)
    if lower == upper:
        return ordered[lower]
    weight = location - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        raise AnalysisError("Wilson interval requires a positive denominator.")
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def median_or_none(values: Iterable[float]) -> float | None:
    materialized = list(values)
    return statistics.median(materialized) if materialized else None


def mean_or_none(values: Iterable[float]) -> float | None:
    materialized = list(values)
    return statistics.fmean(materialized) if materialized else None


def derive_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    suffix = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return BOOTSTRAP_SEED ^ suffix


def validate_input_bundle(input_dir: Path) -> tuple[Path, Path, dict]:
    manifest_path = input_dir / "manifest.json"
    accepted_path = input_dir / "accepted_stream.jsonl.gz"
    predictions_path = input_dir / "predictions.csv.gz"
    for path in (manifest_path, accepted_path, predictions_path):
        if not path.is_file():
            raise AnalysisError(f"Missing required input: {path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    registered = {
        row["name"]: (row["sha256"], int(row["size_bytes"]))
        for row in manifest.get("artifacts", [])
    }
    expected = {
        "accepted_stream.jsonl.gz": EXPECTED_ACCEPTED_SHA256,
        "predictions.csv.gz": EXPECTED_PREDICTIONS_SHA256,
    }
    for name, expected_hash in expected.items():
        if name not in registered:
            raise AnalysisError(f"Input manifest does not register {name}.")
        registered_hash, registered_size = registered[name]
        path = input_dir / name
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash or registered_hash != expected_hash:
            raise AnalysisError(
                f"{name} SHA-256 mismatch: actual={actual_hash}, "
                f"manifest={registered_hash}, expected={expected_hash}"
            )
        if path.stat().st_size != registered_size:
            raise AnalysisError(f"{name} size disagrees with its manifest.")
    return accepted_path, predictions_path, manifest


def load_prediction_metadata(predictions_path: Path) -> dict[str, dict[str, object]]:
    metadata: dict[str, dict[str, object]] = {}
    row_counts: Counter[str] = Counter()
    total_rows = 0
    with gzip.open(predictions_path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "participant_record_id",
            "evidence_stream_id",
            "group_id",
            "topic_id",
            "initial_stance",
            "opinion_change_actual",
        }
        if not required.issubset(reader.fieldnames or ()):
            raise AnalysisError("Predictions file is missing required columns.")
        for row in reader:
            total_rows += 1
            participant_id = row["participant_record_id"]
            candidate = {
                "evidence_stream_id": row["evidence_stream_id"],
                "group_id": row["group_id"],
                "topic_id": row["topic_id"],
                "initial_stance": float(row["initial_stance"]),
                "opinion_change_actual": float(row["opinion_change_actual"]),
            }
            prior = metadata.setdefault(participant_id, candidate)
            for key in (
                "evidence_stream_id",
                "group_id",
                "topic_id",
            ):
                if prior[key] != candidate[key]:
                    raise AnalysisError(
                        f"Inconsistent {key} for participant {participant_id}."
                    )
            for key in ("initial_stance", "opinion_change_actual"):
                if not math.isclose(
                    float(prior[key]),
                    float(candidate[key]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise AnalysisError(
                        f"Inconsistent {key} for participant {participant_id}."
                    )
            row_counts[participant_id] += 1

    if total_rows != EXPECTED_PREDICTION_ROWS:
        raise AnalysisError(
            f"Expected {EXPECTED_PREDICTION_ROWS} prediction rows, got {total_rows}."
        )
    if len(metadata) != EXPECTED_PARTICIPANTS:
        raise AnalysisError(
            f"Expected {EXPECTED_PARTICIPANTS} participants, got {len(metadata)}."
        )
    if set(row_counts.values()) != {12}:
        raise AnalysisError(
            "Every participant must appear exactly 12 times in predictions."
        )
    return metadata


def load_evidence_streams(
    accepted_path: Path,
) -> dict[str, tuple[Evidence, ...]]:
    streams: dict[str, list[Evidence]] = defaultdict(list)
    accepted_ids: set[str] = set()
    sequence_indices: set[int] = set()
    total_rows = 0
    with gzip.open(accepted_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            total_rows += 1
            accepted_id = str(row["accepted_id"])
            if accepted_id in accepted_ids:
                raise AnalysisError(f"Duplicate accepted_id: {accepted_id}")
            accepted_ids.add(accepted_id)
            sequence_index = int(row["sequence_index"])
            if sequence_index in sequence_indices:
                raise AnalysisError(f"Duplicate global sequence_index: {sequence_index}")
            sequence_indices.add(sequence_index)
            sign = int(row["sign"])
            strength = float(row["classifier_strength"])
            if sign not in (-1, 1):
                raise AnalysisError(f"Invalid evidence sign {sign}.")
            if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
                raise AnalysisError(f"Invalid classifier strength {strength}.")
            streams[str(row["participant_record_id"])].append(
                Evidence(sequence_index, sign, strength)
            )

    if total_rows != EXPECTED_ACCEPTED_ROWS:
        raise AnalysisError(
            f"Expected {EXPECTED_ACCEPTED_ROWS} accepted rows, got {total_rows}."
        )
    if len(streams) != EXPECTED_NONEMPTY_STREAMS:
        raise AnalysisError(
            f"Expected {EXPECTED_NONEMPTY_STREAMS} nonempty streams, "
            f"got {len(streams)}."
        )
    if sequence_indices != set(range(EXPECTED_ACCEPTED_ROWS)):
        raise AnalysisError("Global sequence_index values are not exactly contiguous.")
    return {
        participant_id: tuple(sorted(rows, key=lambda row: row.sequence_index))
        for participant_id, rows in streams.items()
    }


def response_stratum(actual_change: float, evidence: Sequence[Evidence]) -> str:
    pressure = sum(item.sign * item.strength for item in evidence)
    weak_pressure = abs(pressure) < WEAK_PRESSURE_THRESHOLD
    if abs(actual_change) <= STABLE_TOLERANCE:
        return "stable_low_pressure" if weak_pressure else "stable_despite_pressure"
    if weak_pressure:
        return "weak_signal_mover"
    if actual_change * pressure > 0.0:
        return "evidence_aligned_mover"
    return "evidence_opposed_mover"


def evidence_relation(initial_stance: float, pressure: float) -> str:
    if abs(pressure) < WEAK_PRESSURE_THRESHOLD:
        return "weak_or_balanced_pressure"
    return (
        "strong_prior_consistent_pressure"
        if initial_stance * pressure > 0.0
        else "strong_prior_opposing_pressure"
    )


def initial_stance_cell(initial_stance: float) -> str:
    return f"{initial_stance:+.15f}"


def prior_extremity(initial_stance: float) -> str:
    magnitude = abs(initial_stance)
    if magnitude < 0.4:
        return "open_weak"
    if magnitude < 0.8:
        return "leaning_moderate"
    return "committed_strong"


def uptake_leverage(evidence: Sequence[Evidence]) -> float:
    """Return the fixed-pair change in cumulative evidence log odds."""
    return sum(
        item.sign
        * (
            math.log1p(HIGH_UPTAKE * item.strength)
            - math.log1p(LOW_UPTAKE * item.strength)
        )
        for item in evidence
    )


def uptake_leverage_orientation(
    initial_stance: float, evidence: Sequence[Evidence]
) -> str:
    product = stance_to_log_odds(initial_stance) * uptake_leverage(evidence)
    direction = signum(product)
    if direction > 0:
        return "prior_consistent_uptake_leverage"
    if direction < 0:
        return "prior_opposing_uptake_leverage"
    return "unorientable_zero_uptake_leverage"


def load_participants(input_dir: Path) -> tuple[list[Participant], dict]:
    accepted_path, predictions_path, manifest = validate_input_bundle(input_dir)
    metadata = load_prediction_metadata(predictions_path)
    streams = load_evidence_streams(accepted_path)
    participants: list[Participant] = []
    for participant_id in sorted(metadata):
        row = metadata[participant_id]
        evidence = streams.get(participant_id, ())
        actual_change = float(row["opinion_change_actual"])
        initial = float(row["initial_stance"])
        pressure = sum(item.sign * item.strength for item in evidence)
        participants.append(
            Participant(
                participant_record_id=participant_id,
                evidence_stream_id=str(row["evidence_stream_id"]),
                group_id=str(row["group_id"]),
                topic_id=str(row["topic_id"]),
                initial_stance=initial,
                opinion_change_actual=actual_change,
                evidence=evidence,
                evidence_pressure=pressure,
                evidence_relation=evidence_relation(initial, pressure),
                uptake_leverage_orientation=uptake_leverage_orientation(
                    initial, evidence
                ),
                initial_stance_cell=initial_stance_cell(initial),
                prior_extremity=prior_extremity(initial),
                response_stratum=response_stratum(actual_change, evidence),
            )
        )

    if len({row.group_id for row in participants}) != EXPECTED_GROUPS:
        raise AnalysisError("Unexpected number of DEBATE group clusters.")
    if len({row.topic_id for row in participants}) != EXPECTED_TOPICS:
        raise AnalysisError("Unexpected number of DEBATE topic clusters.")
    if (
        len({row.evidence_stream_id for row in participants})
        != EXPECTED_EVIDENCE_STREAM_IDS
    ):
        raise AnalysisError("Unexpected number of evidence-stream identifiers.")
    if Counter(row.response_stratum for row in participants) != Counter(
        {
            "evidence_aligned_mover": 542,
            "evidence_opposed_mover": 400,
            "weak_signal_mover": 174,
            "stable_despite_pressure": 1151,
            "stable_low_pressure": 228,
        }
    ):
        raise AnalysisError("Observed-response stratum counts changed.")
    if Counter(row.evidence_relation for row in participants) != Counter(
        {
            "strong_prior_consistent_pressure": 1211,
            "strong_prior_opposing_pressure": 882,
            "weak_or_balanced_pressure": 402,
        }
    ):
        raise AnalysisError("Prior/evidence relation counts changed.")
    if Counter(row.prior_extremity for row in participants) != Counter(
        {
            "open_weak": 897,
            "leaning_moderate": 821,
            "committed_strong": 777,
        }
    ):
        raise AnalysisError("Prior-extremity counts changed.")
    if Counter(row.uptake_leverage_orientation for row in participants) != Counter(
        {
            "prior_consistent_uptake_leverage": 1422,
            "prior_opposing_uptake_leverage": 1072,
            "unorientable_zero_uptake_leverage": 1,
        }
    ):
        raise AnalysisError("Exact uptake-leverage orientation counts changed.")
    return participants, manifest


def replay(participant: Participant, anchor: float, uptake: float) -> Trajectory:
    latent = anchor * stance_to_log_odds(participant.initial_stance)
    log_odds = [latent]
    stances = [log_odds_to_stance(latent)]
    for item in participant.evidence:
        latent += item.sign * math.log1p(item.strength * uptake)
        log_odds.append(latent)
        stances.append(log_odds_to_stance(latent))
    return Trajectory(
        anchor=anchor,
        uptake=uptake,
        stances=tuple(stances),
        log_odds=tuple(log_odds),
    )


def trajectory_cell_name(anchor: float, uptake: float) -> str:
    anchor_name = "low_anchor" if anchor == LOW_ANCHOR else "high_anchor"
    uptake_name = "low_uptake" if uptake == LOW_UPTAKE else "high_uptake"
    return f"{anchor_name}_{uptake_name}"


def condition_rows(
    participants: Sequence[Participant],
) -> tuple[list[dict[str, object]], dict[str, dict[str, Trajectory]]]:
    output: list[dict[str, object]] = []
    trajectories: dict[str, dict[str, Trajectory]] = {}
    for participant in participants:
        cells: dict[str, Trajectory] = {}
        for anchor in (LOW_ANCHOR, HIGH_ANCHOR):
            for uptake in (LOW_UPTAKE, HIGH_UPTAKE):
                name = trajectory_cell_name(anchor, uptake)
                trajectory = replay(participant, anchor, uptake)
                cells[name] = trajectory
                output.append(
                    {
                        "participant_record_id": participant.participant_record_id,
                        "group_id": participant.group_id,
                        "topic_id": participant.topic_id,
                        "response_stratum": participant.response_stratum,
                        "condition": name,
                        "anchor": anchor,
                        "uptake": uptake,
                        "evidence_count": len(participant.evidence),
                        "anchored_initial_state": trajectory.initial,
                        "final_stance": trajectory.final,
                        "absolute_anchored_initial_state": abs(trajectory.initial),
                        "absolute_final_stance": abs(trajectory.final),
                        "signed_stance_shift": trajectory.signed_shift,
                        "absolute_stance_shift": trajectory.absolute_shift,
                        "mean_absolute_step": trajectory.mean_absolute_step,
                        "trajectory_mean": trajectory.trajectory_mean,
                        "side_switch": int(
                            trajectory.first_switch_index is not None
                        ),
                        "first_switch_index": trajectory.first_switch_index,
                    }
                )
        trajectories[participant.participant_record_id] = cells
    return output, trajectories


def subset_by_stratum(
    rows: Sequence[dict[str, object]], stratum: str
) -> list[dict[str, object]]:
    if stratum == "all":
        return list(rows)
    return [row for row in rows if row["response_stratum"] == stratum]


def summarize_factorial_cells(rows: Sequence[dict[str, object]]) -> list[dict]:
    result: list[dict] = []
    for stratum in ("all", *STRATA):
        selected = subset_by_stratum(rows, stratum)
        by_condition: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in selected:
            by_condition[str(row["condition"])].append(row)
        for condition in sorted(by_condition):
            cells = by_condition[condition]
            result.append(
                {
                    "response_stratum": stratum,
                    "condition": condition,
                    "n": len(cells),
                    "group_clusters": len({str(row["group_id"]) for row in cells}),
                    "topic_clusters": len({str(row["topic_id"]) for row in cells}),
                    "anchor": cells[0]["anchor"],
                    "uptake": cells[0]["uptake"],
                    "mean_anchored_initial_state": statistics.fmean(
                        float(row["anchored_initial_state"]) for row in cells
                    ),
                    "mean_absolute_anchored_initial_state": statistics.fmean(
                        float(row["absolute_anchored_initial_state"])
                        for row in cells
                    ),
                    "mean_final_stance": statistics.fmean(
                        float(row["final_stance"]) for row in cells
                    ),
                    "mean_absolute_final_stance": statistics.fmean(
                        float(row["absolute_final_stance"]) for row in cells
                    ),
                    "mean_signed_stance_shift": statistics.fmean(
                        float(row["signed_stance_shift"]) for row in cells
                    ),
                    "mean_absolute_stance_shift": statistics.fmean(
                        float(row["absolute_stance_shift"]) for row in cells
                    ),
                    "mean_absolute_step": statistics.fmean(
                        float(row["mean_absolute_step"]) for row in cells
                    ),
                    "side_switch_rate": statistics.fmean(
                        float(row["side_switch"]) for row in cells
                    ),
                }
            )
    return result


def paired_contrast_records(
    factorial_rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    by_participant: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for row in factorial_rows:
        by_participant[str(row["participant_record_id"])][str(row["condition"])] = row

    definitions = (
        (
            "anchor_effect_on_absolute_initial_at_low_uptake",
            "high_anchor_low_uptake",
            "low_anchor_low_uptake",
            "absolute_anchored_initial_state",
        ),
        (
            "anchor_effect_on_absolute_initial_at_high_uptake",
            "high_anchor_high_uptake",
            "low_anchor_high_uptake",
            "absolute_anchored_initial_state",
        ),
        (
            "uptake_effect_on_anchored_initial_state_at_low_anchor",
            "low_anchor_high_uptake",
            "low_anchor_low_uptake",
            "anchored_initial_state",
        ),
        (
            "uptake_effect_on_anchored_initial_state_at_high_anchor",
            "high_anchor_high_uptake",
            "high_anchor_low_uptake",
            "anchored_initial_state",
        ),
        (
            "uptake_effect_on_mean_absolute_step_at_low_anchor",
            "low_anchor_high_uptake",
            "low_anchor_low_uptake",
            "mean_absolute_step",
        ),
        (
            "uptake_effect_on_mean_absolute_step_at_high_anchor",
            "high_anchor_high_uptake",
            "high_anchor_low_uptake",
            "mean_absolute_step",
        ),
    )
    records: list[dict[str, object]] = []
    for participant_id, cells in by_participant.items():
        sample = next(iter(cells.values()))
        for name, left, right, field in definitions:
            records.append(
                {
                    "participant_record_id": participant_id,
                    "group_id": sample["group_id"],
                    "topic_id": sample["topic_id"],
                    "response_stratum": sample["response_stratum"],
                    "contrast": name,
                    "value": float(cells[left][field]) - float(cells[right][field]),
                }
            )
    return records


def endpoint_row(
    comparison: str,
    participant: Participant,
    left: Trajectory,
    right: Trajectory,
) -> EndpointRow:
    if len(left.stances) != len(right.stances):
        raise AnalysisError("Endpoint comparison trajectories have unequal lengths.")
    gaps = [
        abs(left_value - right_value)
        for left_value, right_value in zip(left.stances, right.stances)
    ]
    interior_gaps = gaps[1:-1]
    maximum_interior_gap = max(interior_gaps) if interior_gaps else 0.0
    mean_interior_gap = (
        statistics.fmean(interior_gaps) if interior_gaps else 0.0
    )
    return EndpointRow(
        comparison=comparison,
        participant_record_id=participant.participant_record_id,
        group_id=participant.group_id,
        topic_id=participant.topic_id,
        response_stratum=participant.response_stratum,
        evidence_relation=participant.evidence_relation,
        uptake_leverage_orientation=participant.uptake_leverage_orientation,
        initial_stance_cell=participant.initial_stance_cell,
        prior_extremity=participant.prior_extremity,
        evidence_count=len(participant.evidence),
        final_a=left.final,
        final_b=right.final,
        final_gap=abs(left.final - right.final),
        initial_gap=abs(left.initial - right.initial),
        maximum_trajectory_gap=max(gaps),
        mean_trajectory_gap=statistics.fmean(gaps),
        maximum_interior_trajectory_gap=maximum_interior_gap,
        mean_interior_trajectory_gap=mean_interior_gap,
        switch_a=left.first_switch_index,
        switch_b=right.first_switch_index,
    )


def build_endpoint_comparisons(
    participants: Sequence[Participant],
    trajectories: Mapping[str, Mapping[str, Trajectory]],
) -> dict[str, list[EndpointRow]]:
    definitions = {
        "cross_compensating": (
            "low_anchor_high_uptake",
            "high_anchor_low_uptake",
        ),
        "low_low_vs_high_high": (
            "low_anchor_low_uptake",
            "high_anchor_high_uptake",
        ),
    }
    output: dict[str, list[EndpointRow]] = {}
    for name, (left_name, right_name) in definitions.items():
        rows = []
        for participant in participants:
            cells = trajectories[participant.participant_record_id]
            rows.append(
                endpoint_row(
                    name,
                    participant,
                    cells[left_name],
                    cells[right_name],
                )
            )
        output[name] = rows
    return output


def build_direction_oriented_comparison(
    participants: Sequence[Participant],
    comparisons: Mapping[str, Sequence[EndpointRow]],
) -> list[EndpointRow]:
    cross = {
        row.participant_record_id: row
        for row in comparisons["cross_compensating"]
    }
    diagonal = {
        row.participant_record_id: row
        for row in comparisons["low_low_vs_high_high"]
    }
    output: list[EndpointRow] = []
    for participant in participants:
        orientation = participant.uptake_leverage_orientation
        if orientation == "prior_consistent_uptake_leverage":
            output.append(cross[participant.participant_record_id])
        elif orientation == "prior_opposing_uptake_leverage":
            output.append(diagonal[participant.participant_record_id])
        elif orientation != "unorientable_zero_uptake_leverage":
            raise AnalysisError(f"Unknown uptake orientation: {orientation}")
    if len(output) != EXPECTED_PARTICIPANTS - 1:
        raise AnalysisError("Unexpected number of direction-oriented records.")
    if len({row.participant_record_id for row in output}) != len(output):
        raise AnalysisError("Direction-oriented comparison duplicated a record.")
    return output


def endpoint_eligible(
    row: EndpointRow, tolerance: float, stratum: str
) -> bool:
    if stratum != "all" and row.response_stratum != stratum:
        return False
    if row.evidence_count < MIN_ARGUMENTS:
        return False
    if max(abs(row.final_a), abs(row.final_b)) >= SATURATION_BOUND:
        return False
    return row.final_gap <= tolerance


def endpoint_denominators(
    rows: Sequence[EndpointRow], stratum: str
) -> tuple[int, int]:
    base = [
        row
        for row in rows
        if (stratum == "all" or row.response_stratum == stratum)
        and row.evidence_count >= MIN_ARGUMENTS
    ]
    nonsaturated = [
        row
        for row in base
        if max(abs(row.final_a), abs(row.final_b)) < SATURATION_BOUND
    ]
    return len(base), len(nonsaturated)


def summarize_endpoint_rows(
    comparisons: Mapping[str, Sequence[EndpointRow]]
) -> tuple[list[dict], list[dict]]:
    summaries: list[dict] = []
    primary_rows: list[dict] = []
    for comparison, rows in comparisons.items():
        for tolerance in ENDPOINT_TOLERANCES:
            for stratum in ("all", *STRATA):
                base_n, nonsaturated_n = endpoint_denominators(rows, stratum)
                matched = [
                    row
                    for row in rows
                    if endpoint_eligible(row, tolerance, stratum)
                ]
                separated_n = sum(
                    row.maximum_interior_trajectory_gap >= 0.2
                    for row in matched
                )
                match_wilson = (
                    wilson_interval(len(matched), nonsaturated_n)
                    if nonsaturated_n
                    else (None, None)
                )
                joint_wilson = (
                    wilson_interval(separated_n, nonsaturated_n)
                    if nonsaturated_n
                    else (None, None)
                )
                conditional_wilson = (
                    wilson_interval(separated_n, len(matched))
                    if matched
                    else (None, None)
                )
                summary = {
                    "comparison": comparison,
                    "endpoint_tolerance": tolerance,
                    "response_stratum": stratum,
                    "minimum_arguments": MIN_ARGUMENTS,
                    "base_n": base_n,
                    "nonsaturated_n": nonsaturated_n,
                    "endpoint_matched_n": len(matched),
                    "endpoint_match_rate_among_nonsaturated": (
                        len(matched) / nonsaturated_n if nonsaturated_n else None
                    ),
                    "endpoint_match_rate_wilson_low": match_wilson[0],
                    "endpoint_match_rate_wilson_high": match_wilson[1],
                    "endpoint_matched_and_separated_n": separated_n,
                    "unconditional_match_and_separation_rate": (
                        separated_n / nonsaturated_n if nonsaturated_n else None
                    ),
                    "unconditional_joint_rate_wilson_low": joint_wilson[0],
                    "unconditional_joint_rate_wilson_high": joint_wilson[1],
                    "matched_group_clusters": len({row.group_id for row in matched}),
                    "matched_topic_clusters": len({row.topic_id for row in matched}),
                    "mean_final_gap": mean_or_none(row.final_gap for row in matched),
                    "median_initial_gap": median_or_none(
                        row.initial_gap for row in matched
                    ),
                    "mean_maximum_trajectory_gap": mean_or_none(
                        row.maximum_trajectory_gap for row in matched
                    ),
                    "median_maximum_trajectory_gap": median_or_none(
                        row.maximum_trajectory_gap for row in matched
                    ),
                    "mean_mean_trajectory_gap": mean_or_none(
                        row.mean_trajectory_gap for row in matched
                    ),
                    "median_mean_trajectory_gap": median_or_none(
                        row.mean_trajectory_gap for row in matched
                    ),
                    "mean_maximum_interior_trajectory_gap": mean_or_none(
                        row.maximum_interior_trajectory_gap for row in matched
                    ),
                    "median_maximum_interior_trajectory_gap": median_or_none(
                        row.maximum_interior_trajectory_gap for row in matched
                    ),
                    "mean_mean_interior_trajectory_gap": mean_or_none(
                        row.mean_interior_trajectory_gap for row in matched
                    ),
                    "median_mean_interior_trajectory_gap": median_or_none(
                        row.mean_interior_trajectory_gap for row in matched
                    ),
                    "separation_rate_ge_0_1": (
                        statistics.fmean(
                            row.maximum_interior_trajectory_gap >= 0.1
                            for row in matched
                        )
                        if matched
                        else None
                    ),
                    "separation_rate_ge_0_2": (
                        separated_n / len(matched)
                        if matched
                        else None
                    ),
                    "conditional_separation_wilson_low": conditional_wilson[0],
                    "conditional_separation_wilson_high": conditional_wilson[1],
                    "switch_discordance_rate": (
                        statistics.fmean(row.switch_discordant for row in matched)
                        if matched
                        else None
                    ),
                    "median_switch_gap_when_both_switch": median_or_none(
                        row.switch_gap
                        for row in matched
                        if row.switch_gap is not None
                    ),
                }
                summaries.append(summary)
                if (
                    comparison == "direction_oriented"
                    and tolerance == PRIMARY_ENDPOINT_TOLERANCE
                    and stratum == "all"
                ):
                    for row in matched:
                        primary_rows.append(
                            {
                                "selected_fixed_comparison": row.comparison,
                                "participant_record_id": row.participant_record_id,
                                "group_id": row.group_id,
                                "topic_id": row.topic_id,
                                "response_stratum": row.response_stratum,
                                "evidence_relation": row.evidence_relation,
                                "uptake_leverage_orientation": (
                                    row.uptake_leverage_orientation
                                ),
                                "initial_stance_cell": row.initial_stance_cell,
                                "prior_extremity": row.prior_extremity,
                                "evidence_count": row.evidence_count,
                                "final_condition_a": row.final_a,
                                "final_condition_b": row.final_b,
                                "absolute_final_gap": row.final_gap,
                                "absolute_anchored_initial_state_gap": (
                                    row.initial_gap
                                ),
                                "maximum_trajectory_gap": row.maximum_trajectory_gap,
                                "mean_trajectory_gap": row.mean_trajectory_gap,
                                "maximum_interior_trajectory_gap": (
                                    row.maximum_interior_trajectory_gap
                                ),
                                "mean_interior_trajectory_gap": (
                                    row.mean_interior_trajectory_gap
                                ),
                                "switch_condition_a": row.switch_a,
                                "switch_condition_b": row.switch_b,
                                "switch_discordant": int(row.switch_discordant),
                                "absolute_switch_gap_when_both": row.switch_gap,
                            }
                        )
    return summaries, primary_rows


def statistic_value(rows: Sequence[EndpointRow], metric: str) -> float:
    if not rows:
        raise AnalysisError("Cannot compute an endpoint statistic on no rows.")
    if metric == "separation_rate_ge_0_2":
        return statistics.fmean(
            row.maximum_interior_trajectory_gap >= 0.2 for row in rows
        )
    if metric == "mean_maximum_trajectory_gap":
        return statistics.fmean(row.maximum_trajectory_gap for row in rows)
    if metric == "median_maximum_trajectory_gap":
        return statistics.median(row.maximum_trajectory_gap for row in rows)
    if metric == "mean_maximum_interior_trajectory_gap":
        return statistics.fmean(
            row.maximum_interior_trajectory_gap for row in rows
        )
    if metric == "median_maximum_interior_trajectory_gap":
        return statistics.median(
            row.maximum_interior_trajectory_gap for row in rows
        )
    if metric == "mean_mean_interior_trajectory_gap":
        return statistics.fmean(row.mean_interior_trajectory_gap for row in rows)
    if metric == "mean_mean_trajectory_gap":
        return statistics.fmean(row.mean_trajectory_gap for row in rows)
    if metric == "switch_discordance_rate":
        return statistics.fmean(row.switch_discordant for row in rows)
    raise AnalysisError(f"Unknown endpoint statistic {metric}.")


def cluster_bootstrap(
    rows: Sequence[object],
    cluster_getter: Callable[[object], str],
    statistic: Callable[[Sequence[object]], float],
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    by_cluster: dict[str, list[object]] = defaultdict(list)
    for row in rows:
        by_cluster[cluster_getter(row)].append(row)
    clusters = sorted(by_cluster)
    if len(rows) < 10 or len(clusters) < 3:
        raise AnalysisError("Sparse bootstrap request violated the analysis rule.")
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(replicates):
        sample: list[object] = []
        for _cluster_index in range(len(clusters)):
            sampled_cluster = clusters[rng.randrange(len(clusters))]
            sample.extend(by_cluster[sampled_cluster])
        values.append(statistic(sample))
    return percentile(values, 0.025), percentile(values, 0.975)


def endpoint_rate_statistics(
    rows: Sequence[EndpointRow],
    tolerance: float = PRIMARY_ENDPOINT_TOLERANCE,
) -> dict[str, float]:
    base = [row for row in rows if row.evidence_count >= MIN_ARGUMENTS]
    nonsaturated = [
        row
        for row in base
        if max(abs(row.final_a), abs(row.final_b)) < SATURATION_BOUND
    ]
    if not nonsaturated:
        raise AnalysisError("Endpoint-rate statistic has no eligible denominator.")
    matched = [row for row in nonsaturated if row.final_gap <= tolerance]
    separated = [
        row
        for row in matched
        if row.maximum_interior_trajectory_gap >= 0.2
    ]
    return {
        "endpoint_match_rate": len(matched) / len(nonsaturated),
        "unconditional_match_and_separation_rate": (
            len(separated) / len(nonsaturated)
        ),
        "conditional_separation_rate": (
            len(separated) / len(matched) if matched else math.nan
        ),
        "median_maximum_interior_trajectory_gap": (
            statistics.median(
                row.maximum_interior_trajectory_gap for row in matched
            )
            if matched
            else math.nan
        ),
    }


def directional_rate_bootstrap_intervals(
    rows: Sequence[EndpointRow],
    replicates: int,
) -> list[dict[str, object]]:
    """Bootstrap the full denominator under registered group/topic schemes."""
    by_topic: dict[str, list[EndpointRow]] = defaultdict(list)
    by_topic_group: dict[str, dict[str, list[EndpointRow]]] = defaultdict(
        lambda: defaultdict(list)
    )
    group_topics: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_topic[row.topic_id].append(row)
        by_topic_group[row.topic_id][row.group_id].append(row)
        group_topics[row.group_id].add(row.topic_id)
    if any(len(topics) != 1 for topics in group_topics.values()):
        raise AnalysisError("Group identifiers are not nested within topics.")

    topics = sorted(by_topic)
    point = endpoint_rate_statistics(rows)
    schemes = (
        "fixed_topics_groups_within_topic",
        "whole_topic_cluster",
        "hierarchical_topic_then_group",
    )
    output: list[dict[str, object]] = []
    for scheme in schemes:
        rng = random.Random(derive_seed("directional_rate_bootstrap", scheme))
        values: dict[str, list[float]] = {
            metric: [] for metric in point
        }
        for _ in range(replicates):
            sample: list[EndpointRow] = []
            if scheme == "fixed_topics_groups_within_topic":
                sampled_topics = topics
            else:
                sampled_topics = [
                    topics[rng.randrange(len(topics))]
                    for _topic_index in range(len(topics))
                ]
            for topic in sampled_topics:
                if scheme == "whole_topic_cluster":
                    sample.extend(by_topic[topic])
                    continue
                groups = sorted(by_topic_group[topic])
                for _group_index in range(len(groups)):
                    sampled_group = groups[rng.randrange(len(groups))]
                    sample.extend(by_topic_group[topic][sampled_group])
            replicate = endpoint_rate_statistics(sample)
            for metric, value in replicate.items():
                if math.isfinite(value):
                    values[metric].append(value)
        for metric, estimate in point.items():
            metric_values = values[metric]
            if len(metric_values) < max(1, int(0.95 * replicates)):
                raise AnalysisError(
                    f"Too many undefined {metric} bootstrap replicates."
                )
            output.append(
                {
                    "comparison": "direction_oriented",
                    "endpoint_tolerance": PRIMARY_ENDPOINT_TOLERANCE,
                    "resampling_scheme": scheme,
                    "metric": metric,
                    "estimate": estimate,
                    "ci_low": percentile(metric_values, 0.025),
                    "ci_high": percentile(metric_values, 0.975),
                    "bootstrap_replicates_requested": replicates,
                    "bootstrap_replicates_defined": len(metric_values),
                }
            )
    return output


def endpoint_bootstrap_intervals(
    comparisons: Mapping[str, Sequence[EndpointRow]],
    replicates: int,
) -> list[dict]:
    metrics = (
        "separation_rate_ge_0_2",
        "mean_maximum_interior_trajectory_gap",
        "median_maximum_interior_trajectory_gap",
        "mean_mean_interior_trajectory_gap",
        "switch_discordance_rate",
    )
    output: list[dict] = []
    for comparison, rows in comparisons.items():
        for tolerance in ENDPOINT_TOLERANCES:
            for stratum in ("all", *STRATA):
                matched = [
                    row
                    for row in rows
                    if endpoint_eligible(row, tolerance, stratum)
                ]
                for cluster_field in ("group_id", "topic_id"):
                    clusters = {
                        getattr(row, cluster_field)
                        for row in matched
                    }
                    for metric in metrics:
                        record = {
                            "comparison": comparison,
                            "endpoint_tolerance": tolerance,
                            "response_stratum": stratum,
                            "cluster_field": cluster_field,
                            "metric": metric,
                            "n": len(matched),
                            "clusters": len(clusters),
                            "estimate": (
                                statistic_value(matched, metric) if matched else None
                            ),
                            "ci_low": None,
                            "ci_high": None,
                            "bootstrap_replicates": 0,
                        }
                        if len(matched) >= 10 and len(clusters) >= 3:
                            low, high = cluster_bootstrap(
                                matched,
                                lambda row, field=cluster_field: str(
                                    getattr(row, field)
                                ),
                                lambda sample, name=metric: statistic_value(
                                    sample, name
                                ),
                                replicates,
                                derive_seed(
                                    comparison,
                                    tolerance,
                                    stratum,
                                    cluster_field,
                                    metric,
                                ),
                            )
                            record.update(
                                {
                                    "ci_low": low,
                                    "ci_high": high,
                                    "bootstrap_replicates": replicates,
                                }
                            )
                        output.append(record)
    return output


def contrast_bootstrap_summaries(
    records: Sequence[dict[str, object]], replicates: int
) -> list[dict]:
    output: list[dict] = []
    contrasts = sorted({str(row["contrast"]) for row in records})
    for contrast in contrasts:
        contrast_rows = [row for row in records if row["contrast"] == contrast]
        for stratum in ("all", *STRATA):
            selected = [
                row
                for row in contrast_rows
                if stratum == "all" or row["response_stratum"] == stratum
            ]
            for cluster_field in ("group_id", "topic_id"):
                clusters = {str(row[cluster_field]) for row in selected}
                estimate = statistics.fmean(float(row["value"]) for row in selected)
                low, high = cluster_bootstrap(
                    selected,
                    lambda row, field=cluster_field: str(row[field]),
                    lambda sample: statistics.fmean(
                        float(row["value"]) for row in sample
                    ),
                    replicates,
                    derive_seed(contrast, stratum, cluster_field),
                )
                output.append(
                    {
                        "contrast": contrast,
                        "response_stratum": stratum,
                        "cluster_field": cluster_field,
                        "n": len(selected),
                        "clusters": len(clusters),
                        "estimate": estimate,
                        "ci_low": low,
                        "ci_high": high,
                        "bootstrap_replicates": replicates,
                    }
                )
    return output


def mechanism_checks(
    participants: Sequence[Participant],
    trajectories: Mapping[str, Mapping[str, Trajectory]],
) -> dict[str, object]:
    max_initial_uptake_error = 0.0
    max_increment_anchor_error = 0.0
    max_round_trip_error = 0.0
    max_cross_endpoint_decomposition_error = 0.0
    max_diagonal_endpoint_decomposition_error = 0.0
    raw_pressure_orientation_disagreements = 0
    for participant in participants:
        cells = trajectories[participant.participant_record_id]
        max_initial_uptake_error = max(
            max_initial_uptake_error,
            abs(
                cells["low_anchor_low_uptake"].initial
                - cells["low_anchor_high_uptake"].initial
            ),
            abs(
                cells["high_anchor_low_uptake"].initial
                - cells["high_anchor_high_uptake"].initial
            ),
        )
        for uptake_name in ("low_uptake", "high_uptake"):
            low = cells[f"low_anchor_{uptake_name}"].log_odds
            high = cells[f"high_anchor_{uptake_name}"].log_odds
            for index in range(1, len(low)):
                low_increment = low[index] - low[index - 1]
                high_increment = high[index] - high[index - 1]
                max_increment_anchor_error = max(
                    max_increment_anchor_error,
                    abs(low_increment - high_increment),
                )
        restored = log_odds_to_stance(
            stance_to_log_odds(participant.initial_stance)
        )
        max_round_trip_error = max(
            max_round_trip_error, abs(restored - participant.initial_stance)
        )
        prior = stance_to_log_odds(participant.initial_stance)
        leverage = uptake_leverage(participant.evidence)
        cross_observed = (
            cells["low_anchor_high_uptake"].log_odds[-1]
            - cells["high_anchor_low_uptake"].log_odds[-1]
        )
        diagonal_observed = (
            cells["low_anchor_low_uptake"].log_odds[-1]
            - cells["high_anchor_high_uptake"].log_odds[-1]
        )
        max_cross_endpoint_decomposition_error = max(
            max_cross_endpoint_decomposition_error,
            abs(cross_observed - (leverage - 0.8 * prior)),
        )
        max_diagonal_endpoint_decomposition_error = max(
            max_diagonal_endpoint_decomposition_error,
            abs(diagonal_observed - (-leverage - 0.8 * prior)),
        )
        raw_orientation = signum(prior * participant.evidence_pressure)
        exact_orientation = signum(prior * leverage)
        raw_pressure_orientation_disagreements += (
            raw_orientation != exact_orientation
        )
    return {
        "max_anchored_initial_state_change_when_only_uptake_changes": (
            max_initial_uptake_error
        ),
        "max_log_odds_increment_change_when_only_anchor_changes": (
            max_increment_anchor_error
        ),
        "max_stance_log_odds_round_trip_error": max_round_trip_error,
        "max_cross_endpoint_decomposition_error": (
            max_cross_endpoint_decomposition_error
        ),
        "max_diagonal_endpoint_decomposition_error": (
            max_diagonal_endpoint_decomposition_error
        ),
        "raw_pressure_vs_exact_orientation_disagreements": (
            raw_pressure_orientation_disagreements
        ),
        "high_anchor_high_uptake_cell_executed": True,
        "parameters_sum_constraint_present": False,
    }


def compensation_ratio_rows(
    participants: Sequence[Participant],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    participant_rows: list[dict[str, object]] = []
    for participant in participants:
        prior_log_odds = stance_to_log_odds(participant.initial_stance)
        if prior_log_odds == 0.0:
            raise AnalysisError("Compensation ratio is undefined at zero prior.")
        leverage = uptake_leverage(participant.evidence)
        participant_rows.append(
            {
                "participant_record_id": participant.participant_record_id,
                "group_id": participant.group_id,
                "topic_id": participant.topic_id,
                "initial_stance_cell": participant.initial_stance_cell,
                "evidence_relation": participant.evidence_relation,
                "uptake_leverage_orientation": (
                    participant.uptake_leverage_orientation
                ),
                "prior_log_odds": prior_log_odds,
                "uptake_leverage_low_to_high": leverage,
                "required_anchor_difference_for_cross_endpoint_match": (
                    leverage / prior_log_odds
                ),
            }
        )

    selectors: list[tuple[str, str, Callable[[dict[str, object]], bool]]] = [
        ("all", "all", lambda _row: True)
    ]
    for field in (
        "uptake_leverage_orientation",
        "evidence_relation",
        "initial_stance_cell",
    ):
        levels = sorted({str(row[field]) for row in participant_rows})
        selectors.extend(
            (
                field,
                level,
                lambda row, key=field, value=level: str(row[key]) == value,
            )
            for level in levels
        )

    summaries: list[dict[str, object]] = []
    for dimension, level, selector in selectors:
        selected = [row for row in participant_rows if selector(row)]
        ratios = [
            float(row["required_anchor_difference_for_cross_endpoint_match"])
            for row in selected
        ]
        summaries.append(
            {
                "dimension": dimension,
                "level": level,
                "participant_records": len(selected),
                "negative_ratio_n": sum(value < 0.0 for value in ratios),
                "zero_ratio_n": sum(value == 0.0 for value in ratios),
                "positive_ratio_n": sum(value > 0.0 for value in ratios),
                "minimum": min(ratios),
                "q25": percentile(ratios, 0.25),
                "median": statistics.median(ratios),
                "q75": percentile(ratios, 0.75),
                "maximum": max(ratios),
            }
        )
    return summaries, participant_rows


def grid_robustness(
    participants: Sequence[Participant],
) -> tuple[list[dict], list[dict]]:
    participant_rows: list[dict] = []
    for participant in participants:
        if len(participant.evidence) < MIN_ARGUMENTS:
            continue
        grid = [
            replay(participant, anchor, uptake)
            for anchor in ANCHOR_GRID
            for uptake in UPDATE_GRID
        ]
        for tolerance in ENDPOINT_TOLERANCES:
            pair_count = 0
            separated_count = 0
            maximum_gaps: list[float] = []
            mean_gaps: list[float] = []
            for left_index, left in enumerate(grid):
                if abs(left.final) >= SATURATION_BOUND:
                    continue
                for right in grid[left_index + 1 :]:
                    if abs(right.final) >= SATURATION_BOUND:
                        continue
                    if (
                        abs(left.anchor - right.anchor)
                        < GRID_MIN_ANCHOR_DIFFERENCE
                        or abs(left.uptake - right.uptake)
                        < GRID_MIN_UPTAKE_DIFFERENCE
                    ):
                        continue
                    if abs(left.final - right.final) > tolerance:
                        continue
                    pair_count += 1
                    gaps = [
                        abs(a - b)
                        for a, b in zip(left.stances, right.stances)
                    ][1:-1]
                    if not gaps:
                        raise AnalysisError(
                            "Grid robustness requires interior trajectory points."
                        )
                    maximum_gap = max(gaps)
                    maximum_gaps.append(maximum_gap)
                    mean_gaps.append(statistics.fmean(gaps))
                    separated_count += maximum_gap >= 0.2
            participant_rows.append(
                {
                    "participant_record_id": participant.participant_record_id,
                    "group_id": participant.group_id,
                    "topic_id": participant.topic_id,
                    "response_stratum": participant.response_stratum,
                    "endpoint_tolerance": tolerance,
                    "eligible_pair_count": pair_count,
                    "separated_pair_count": separated_count,
                    "separated_pair_fraction": (
                        separated_count / pair_count if pair_count else None
                    ),
                    "median_maximum_trajectory_gap": (
                        statistics.median(maximum_gaps)
                        if maximum_gaps
                        else None
                    ),
                    "median_mean_trajectory_gap": (
                        statistics.median(mean_gaps) if mean_gaps else None
                    ),
                }
            )

    summaries: list[dict] = []
    for tolerance in ENDPOINT_TOLERANCES:
        tolerance_rows = [
            row
            for row in participant_rows
            if row["endpoint_tolerance"] == tolerance
        ]
        for stratum in ("all", *STRATA):
            selected = [
                row
                for row in tolerance_rows
                if stratum == "all" or row["response_stratum"] == stratum
            ]
            with_pairs = [
                row for row in selected if int(row["eligible_pair_count"]) > 0
            ]
            summaries.append(
                {
                    "endpoint_tolerance": tolerance,
                    "response_stratum": stratum,
                    "participants": len(selected),
                    "participants_with_endpoint_equivalent_pairs": len(with_pairs),
                    "participant_coverage_rate": (
                        len(with_pairs) / len(selected) if selected else None
                    ),
                    "median_participant_pair_count": median_or_none(
                        float(row["eligible_pair_count"]) for row in with_pairs
                    ),
                    "mean_participant_separated_pair_fraction": mean_or_none(
                        float(row["separated_pair_fraction"]) for row in with_pairs
                    ),
                    "median_of_participant_median_maximum_gap": median_or_none(
                        float(row["median_maximum_trajectory_gap"])
                        for row in with_pairs
                    ),
                    "median_of_participant_median_mean_gap": median_or_none(
                        float(row["median_mean_trajectory_gap"])
                        for row in with_pairs
                    ),
                }
            )
    return summaries, participant_rows


def cluster_primary_summary(
    participants: Sequence[Participant],
    rows: Sequence[EndpointRow],
    cluster_field: str,
) -> list[dict]:
    by_cluster: dict[str, list[EndpointRow]] = defaultdict(list)
    for row in rows:
        by_cluster[str(getattr(row, cluster_field))].append(row)
    source_by_cluster: Counter[str] = Counter(
        str(getattr(row, cluster_field)) for row in participants
    )
    output: list[dict] = []
    for cluster in sorted(source_by_cluster):
        cluster_rows = by_cluster.get(cluster, [])
        base = [row for row in cluster_rows if row.evidence_count >= MIN_ARGUMENTS]
        nonsaturated = [
            row
            for row in base
            if max(abs(row.final_a), abs(row.final_b)) < SATURATION_BOUND
        ]
        matched = [
            row
            for row in nonsaturated
            if row.final_gap <= PRIMARY_ENDPOINT_TOLERANCE
        ]
        output.append(
            {
                "cluster_field": cluster_field,
                "cluster_id": cluster,
                "source_participant_records": source_by_cluster[cluster],
                "oriented_comparison_records": len(cluster_rows),
                "minimum_argument_eligible": len(base),
                "nonsaturated": len(nonsaturated),
                "endpoint_matched": len(matched),
                "trajectory_separated_ge_0_2": sum(
                    row.maximum_interior_trajectory_gap >= 0.2 for row in matched
                ),
                "mean_maximum_interior_trajectory_gap": mean_or_none(
                    row.maximum_interior_trajectory_gap for row in matched
                ),
            }
        )
    return output


def count_level(size: int) -> str:
    return f"count_{size}"


def primary_heterogeneity_summary(
    participants: Sequence[Participant],
    rows: Sequence[EndpointRow],
) -> list[dict]:
    by_id = {row.participant_record_id: row for row in participants}
    group_record_counts = Counter(row.group_id for row in participants)
    group_streams: dict[str, set[str]] = defaultdict(set)
    for participant in participants:
        group_streams[participant.group_id].add(participant.evidence_stream_id)
    dimensions: dict[str, Callable[[Participant], str]] = {
        "uptake_leverage_orientation": (
            lambda row: row.uptake_leverage_orientation
        ),
        "evidence_relation": lambda row: row.evidence_relation,
        "prior_extremity": lambda row: row.prior_extremity,
        "initial_stance_cell": lambda row: row.initial_stance_cell,
        "prior_extremity_x_evidence_relation": (
            lambda row: f"{row.prior_extremity}__{row.evidence_relation}"
        ),
        "distinct_evidence_stream_count_per_group": (
            lambda row: count_level(len(group_streams[row.group_id]))
        ),
        "participant_record_count_per_group": (
            lambda row: count_level(group_record_counts[row.group_id])
        ),
        "observed_response_stratum": lambda row: row.response_stratum,
    }
    output: list[dict] = []
    for dimension, key_function in dimensions.items():
        levels = sorted({key_function(row) for row in participants})
        for level in levels:
            source_selected = [
                row for row in participants if key_function(row) == level
            ]
            selected = [
                row
                for row in rows
                if key_function(by_id[row.participant_record_id]) == level
            ]
            base = [row for row in selected if row.evidence_count >= MIN_ARGUMENTS]
            nonsaturated = [
                row
                for row in base
                if max(abs(row.final_a), abs(row.final_b)) < SATURATION_BOUND
            ]
            matched = [
                row
                for row in nonsaturated
                if row.final_gap <= PRIMARY_ENDPOINT_TOLERANCE
            ]
            separated = [
                row
                for row in matched
                if row.maximum_interior_trajectory_gap >= 0.2
            ]
            output.append(
                {
                    "dimension": dimension,
                    "level": level,
                    "source_participant_records": len(source_selected),
                    "oriented_comparison_records": len(selected),
                    "minimum_argument_eligible": len(base),
                    "nonsaturated": len(nonsaturated),
                    "endpoint_matched": len(matched),
                    "endpoint_match_rate": (
                        len(matched) / len(nonsaturated)
                        if nonsaturated
                        else None
                    ),
                    "endpoint_matched_and_separated": len(separated),
                    "unconditional_match_and_separation_rate": (
                        len(separated) / len(nonsaturated)
                        if nonsaturated
                        else None
                    ),
                    "conditional_separation_rate": (
                        len(separated) / len(matched) if matched else None
                    ),
                    "median_maximum_interior_trajectory_gap": median_or_none(
                        row.maximum_interior_trajectory_gap for row in matched
                    ),
                    "source_group_clusters": len(
                        {row.group_id for row in source_selected}
                    ),
                    "source_topic_clusters": len(
                        {row.topic_id for row in source_selected}
                    ),
                    "oriented_group_clusters": len(
                        {row.group_id for row in selected}
                    ),
                    "oriented_topic_clusters": len(
                        {row.topic_id for row in selected}
                    ),
                }
            )
    return output


def cluster_weighted_mean_rate(
    numerator_rows: Sequence[EndpointRow],
    denominator_rows: Sequence[EndpointRow],
    cluster_field: str,
) -> float | None:
    numerators = Counter(getattr(row, cluster_field) for row in numerator_rows)
    denominators = Counter(getattr(row, cluster_field) for row in denominator_rows)
    rates = [
        numerators[cluster] / denominator
        for cluster, denominator in denominators.items()
        if denominator > 0
    ]
    return statistics.fmean(rates) if rates else None


def weighting_summaries(
    comparisons: Mapping[str, Sequence[EndpointRow]]
) -> list[dict]:
    output: list[dict] = []
    for comparison, rows in comparisons.items():
        for tolerance in ENDPOINT_TOLERANCES:
            base = [row for row in rows if row.evidence_count >= MIN_ARGUMENTS]
            nonsaturated = [
                row
                for row in base
                if max(abs(row.final_a), abs(row.final_b)) < SATURATION_BOUND
            ]
            matched = [row for row in nonsaturated if row.final_gap <= tolerance]
            separated = [
                row
                for row in matched
                if row.maximum_interior_trajectory_gap >= 0.2
            ]
            for weighting in ("participant", "equal_group", "equal_topic"):
                if weighting == "participant":
                    match_rate = (
                        len(matched) / len(nonsaturated)
                        if nonsaturated
                        else None
                    )
                    joint_rate = (
                        len(separated) / len(nonsaturated)
                        if nonsaturated
                        else None
                    )
                    conditional = (
                        len(separated) / len(matched) if matched else None
                    )
                else:
                    field = "group_id" if weighting == "equal_group" else "topic_id"
                    match_rate = cluster_weighted_mean_rate(
                        matched, nonsaturated, field
                    )
                    joint_rate = cluster_weighted_mean_rate(
                        separated, nonsaturated, field
                    )
                    conditional = (
                        joint_rate / match_rate
                        if match_rate not in (None, 0.0)
                        and joint_rate is not None
                        else None
                    )
                output.append(
                    {
                        "comparison": comparison,
                        "endpoint_tolerance": tolerance,
                        "weighting": weighting,
                        "nonsaturated_participant_records": len(nonsaturated),
                        "endpoint_matched_participant_records": len(matched),
                        "endpoint_match_rate": match_rate,
                        "unconditional_match_and_separation_rate": joint_rate,
                        "conditional_separation_rate": conditional,
                    }
                )
    return output


def duplicate_stream_sensitivity(
    participants: Sequence[Participant],
    rows: Sequence[EndpointRow],
) -> list[dict]:
    by_participant = {row.participant_record_id: row for row in participants}
    representative: dict[str, str] = {}
    for participant in participants:
        current = representative.get(participant.evidence_stream_id)
        if current is None or participant.participant_record_id < current:
            representative[participant.evidence_stream_id] = (
                participant.participant_record_id
            )
    keep = set(representative.values())
    output = []
    for sample_name, selected in (
        ("all_oriented_records", list(rows)),
        (
            "one_oriented_record_per_evidence_stream_id",
            [row for row in rows if row.participant_record_id in keep],
        ),
    ):
        base = [row for row in selected if row.evidence_count >= MIN_ARGUMENTS]
        nonsaturated = [
            row
            for row in base
            if max(abs(row.final_a), abs(row.final_b)) < SATURATION_BOUND
        ]
        matched = [
            row
            for row in nonsaturated
            if row.final_gap <= PRIMARY_ENDPOINT_TOLERANCE
        ]
        separated = [
            row
            for row in matched
            if row.maximum_interior_trajectory_gap >= 0.2
        ]
        output.append(
            {
                "sample": sample_name,
                "source_participant_records": len(participants),
                "source_evidence_stream_ids": len(
                    {row.evidence_stream_id for row in participants}
                ),
                "oriented_comparison_records": len(selected),
                "oriented_evidence_stream_ids": len(
                    {
                        by_participant[row.participant_record_id].evidence_stream_id
                        for row in selected
                    }
                ),
                "nonsaturated": len(nonsaturated),
                "endpoint_matched": len(matched),
                "endpoint_matched_and_separated": len(separated),
                "endpoint_match_rate": (
                    len(matched) / len(nonsaturated) if nonsaturated else None
                ),
                "unconditional_match_and_separation_rate": (
                    len(separated) / len(nonsaturated) if nonsaturated else None
                ),
                "conditional_separation_rate": (
                    len(separated) / len(matched) if matched else None
                ),
                "median_maximum_interior_trajectory_gap": median_or_none(
                    row.maximum_interior_trajectory_gap for row in matched
                ),
            }
        )
    return output


def all_four_cell_nonsaturation_sensitivity(
    participants: Sequence[Participant],
    rows: Sequence[EndpointRow],
    trajectories: Mapping[str, Mapping[str, Trajectory]],
) -> list[dict[str, object]]:
    participant_by_id = {
        row.participant_record_id: row for row in participants
    }
    output: list[dict[str, object]] = []
    for tolerance in ENDPOINT_TOLERANCES:
        for orientation in (
            "all",
            "prior_consistent_uptake_leverage",
            "prior_opposing_uptake_leverage",
        ):
            base = [
                row
                for row in rows
                if row.evidence_count >= MIN_ARGUMENTS
                and (
                    orientation == "all"
                    or participant_by_id[
                        row.participant_record_id
                    ].uptake_leverage_orientation
                    == orientation
                )
            ]
            pair_nonsaturated = [
                row
                for row in base
                if max(abs(row.final_a), abs(row.final_b)) < SATURATION_BOUND
            ]
            all_four_nonsaturated = []
            for row in base:
                cells = trajectories[row.participant_record_id]
                if all(
                    abs(trajectory.final) < SATURATION_BOUND
                    for trajectory in cells.values()
                ):
                    all_four_nonsaturated.append(row)
            for rule, denominator in (
                ("selected_pair_nonsaturated", pair_nonsaturated),
                ("all_four_cells_nonsaturated", all_four_nonsaturated),
            ):
                matched = [
                    row for row in denominator if row.final_gap <= tolerance
                ]
                separated = [
                    row
                    for row in matched
                    if row.maximum_interior_trajectory_gap >= 0.2
                ]
                output.append(
                    {
                        "endpoint_tolerance": tolerance,
                        "uptake_leverage_orientation": orientation,
                        "nonsaturation_rule": rule,
                        "minimum_argument_eligible": len(base),
                        "nonsaturated": len(denominator),
                        "endpoint_matched": len(matched),
                        "endpoint_matched_and_separated": len(separated),
                        "endpoint_match_rate": (
                            len(matched) / len(denominator)
                            if denominator
                            else None
                        ),
                        "unconditional_match_and_separation_rate": (
                            len(separated) / len(denominator)
                            if denominator
                            else None
                        ),
                        "conditional_separation_rate": (
                            len(separated) / len(matched) if matched else None
                        ),
                        "median_maximum_interior_trajectory_gap": (
                            median_or_none(
                                row.maximum_interior_trajectory_gap
                                for row in matched
                            )
                        ),
                    }
                )
    return output


def trajectory_examples(
    rows: Sequence[EndpointRow],
    trajectories: Mapping[str, Mapping[str, Trajectory]],
) -> tuple[list[dict[str, object]], str]:
    definitions = {
        "prior_consistent_uptake_leverage": (
            "low_anchor_high_uptake",
            "high_anchor_low_uptake",
        ),
        "prior_opposing_uptake_leverage": (
            "low_anchor_low_uptake",
            "high_anchor_high_uptake",
        ),
    }
    examples: list[tuple[str, EndpointRow, str, str]] = []
    for orientation, (condition_a, condition_b) in definitions.items():
        candidates = [
            row
            for row in rows
            if row.uptake_leverage_orientation == orientation
            and row.evidence_count >= MIN_ARGUMENTS
            and max(abs(row.final_a), abs(row.final_b)) < SATURATION_BOUND
            and row.final_gap <= PRIMARY_ENDPOINT_TOLERANCE
            and row.maximum_interior_trajectory_gap >= 0.2
        ]
        if not candidates:
            raise AnalysisError(f"No trajectory example for {orientation}.")
        selected = min(
            candidates,
            key=lambda row: (row.final_gap, row.participant_record_id),
        )
        examples.append((orientation, selected, condition_a, condition_b))

    csv_rows: list[dict[str, object]] = []
    for example_index, (orientation, selected, condition_a, condition_b) in enumerate(
        examples, start=1
    ):
        cells = trajectories[selected.participant_record_id]
        for condition in (condition_a, condition_b):
            trajectory = cells[condition]
            for step, stance in enumerate(trajectory.stances):
                csv_rows.append(
                    {
                        "example": example_index,
                        "selection_rule": (
                            "smallest_absolute_final_gap_among_registered_separated_matches_within_orientation"
                        ),
                        "participant_record_id": selected.participant_record_id,
                        "uptake_leverage_orientation": orientation,
                        "selected_fixed_comparison": selected.comparison,
                        "condition": condition,
                        "evidence_step": step,
                        "stance": stance,
                        "absolute_final_gap": selected.final_gap,
                        "maximum_interior_trajectory_gap": (
                            selected.maximum_interior_trajectory_gap
                        ),
                    }
                )
    return csv_rows, trajectory_examples_svg(examples, trajectories)


def trajectory_examples_svg(
    examples: Sequence[tuple[str, EndpointRow, str, str]],
    trajectories: Mapping[str, Mapping[str, Trajectory]],
) -> str:
    width = 960
    height = 410
    margin_left = 58
    margin_right = 24
    margin_top = 76
    margin_bottom = 56
    panel_gap = 54
    panel_width = (
        width - margin_left - margin_right - panel_gap
    ) / len(examples)
    plot_height = height - margin_top - margin_bottom
    colors = ("#0072B2", "#D55E00")

    def y_position(value: float) -> float:
        return margin_top + (1.0 - (value + 1.0) / 2.0) * plot_height

    svg = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" '
            'role="img" aria-labelledby="title desc">'
        ),
        "<title id=\"title\">Endpoint-matched stance trajectories</title>",
        (
            "<desc id=\"desc\">Two fixed-stream examples selected only by "
            "smallest final endpoint gap among matches that meet the registered "
            "interior-separation threshold within each algebraic "
            "uptake-leverage direction.</desc>"
        ),
        "<rect width=\"100%\" height=\"100%\" fill=\"#ffffff\"/>",
        (
            '<text x="480" y="24" text-anchor="middle" '
            'font-family="sans-serif" font-size="17" font-weight="600">'
            "Same endpoint, different evidence-response path</text>"
        ),
    ]
    for y_tick in (-1.0, 0.0, 1.0):
        y = y_position(y_tick)
        svg.append(
            f'<text x="{margin_left - 10}" y="{y + 4:.2f}" text-anchor="end" '
            'font-family="sans-serif" font-size="11" fill="#444444">'
            f"{y_tick:+.0f}</text>"
        )
    svg.append(
        f'<text x="16" y="{margin_top + plot_height / 2:.2f}" '
        'text-anchor="middle" font-family="sans-serif" font-size="12" '
        'fill="#222222" transform="rotate(-90 16 '
        f'{margin_top + plot_height / 2:.2f})">stance</text>'
    )

    for panel_index, (orientation, row, condition_a, condition_b) in enumerate(
        examples
    ):
        x0 = margin_left + panel_index * (panel_width + panel_gap)
        cells = trajectories[row.participant_record_id]
        evidence_steps = len(cells[condition_a].stances) - 1

        def x_position(step: int) -> float:
            return x0 + (step / evidence_steps) * panel_width

        for y_tick in (-1.0, 0.0, 1.0):
            y = y_position(y_tick)
            svg.append(
                f'<line x1="{x0:.2f}" y1="{y:.2f}" '
                f'x2="{x0 + panel_width:.2f}" y2="{y:.2f}" '
                'stroke="#d8d8d8" stroke-width="1"/>'
            )
        svg.append(
            f'<line x1="{x0:.2f}" y1="{margin_top:.2f}" '
            f'x2="{x0:.2f}" y2="{margin_top + plot_height:.2f}" '
            'stroke="#555555" stroke-width="1"/>'
        )
        svg.append(
            f'<line x1="{x0:.2f}" y1="{margin_top + plot_height:.2f}" '
            f'x2="{x0 + panel_width:.2f}" '
            f'y2="{margin_top + plot_height:.2f}" '
            'stroke="#555555" stroke-width="1"/>'
        )
        if orientation == "prior_consistent_uptake_leverage":
            panel_title = "Prior-consistent uptake leverage"
            series_labels = (
                "A: low anchor / high uptake",
                "B: high anchor / low uptake",
            )
        else:
            panel_title = "Prior-opposing uptake leverage"
            series_labels = (
                "A: low anchor / low uptake",
                "B: high anchor / high uptake",
            )
        svg.append(
            f'<text x="{x0 + panel_width / 2:.2f}" y="43" '
            'text-anchor="middle" font-family="sans-serif" font-size="13" '
            f'font-weight="600">{escape(panel_title)}</text>'
        )
        label_start = x0 + 10
        for label_index, label in enumerate(series_labels):
            label_x = label_start + label_index * (panel_width / 2)
            svg.append(
                f'<line x1="{label_x:.2f}" y1="61" '
                f'x2="{label_x + 18:.2f}" y2="61" '
                f'stroke="{colors[label_index]}" stroke-width="2.5"/>'
            )
            svg.append(
                f'<text x="{label_x + 24:.2f}" y="65" '
                'font-family="sans-serif" font-size="10.5" fill="#333333">'
                f"{escape(label)}</text>"
            )
        for series_index, condition in enumerate((condition_a, condition_b)):
            trajectory = cells[condition]
            points = " ".join(
                f"{x_position(step):.2f},{y_position(value):.2f}"
                for step, value in enumerate(trajectory.stances)
            )
            svg.append(
                f'<polyline points="{points}" fill="none" '
                f'stroke="{colors[series_index]}" stroke-width="2.5"/>'
            )
            for step, value in enumerate(trajectory.stances):
                svg.append(
                    f'<circle cx="{x_position(step):.2f}" '
                    f'cy="{y_position(value):.2f}" r="2.7" '
                    f'fill="{colors[series_index]}"/>'
                )
        svg.append(
            f'<text x="{x0 + panel_width / 2:.2f}" '
            f'y="{height - 33}" text-anchor="middle" '
            'font-family="sans-serif" font-size="11" fill="#333333">'
            f"evidence step (0–{evidence_steps})</text>"
        )
        svg.append(
            f'<text x="{x0 + panel_width / 2:.2f}" '
            f'y="{height - 14}" text-anchor="middle" '
            'font-family="sans-serif" font-size="10.5" fill="#555555">'
            f"final gap {row.final_gap:.4f}; max interior gap "
            f"{row.maximum_interior_trajectory_gap:.3f}</text>"
        )

    svg.append("</svg>")
    return "\n".join(svg) + "\n"


def leave_one_topic_out(rows: Sequence[EndpointRow]) -> list[dict]:
    topics = sorted({row.topic_id for row in rows})
    output: list[dict] = []
    for omitted in topics:
        selected = [row for row in rows if row.topic_id != omitted]
        base = [row for row in selected if row.evidence_count >= MIN_ARGUMENTS]
        nonsaturated = [
            row
            for row in base
            if max(abs(row.final_a), abs(row.final_b)) < SATURATION_BOUND
        ]
        matched = [
            row
            for row in nonsaturated
            if row.final_gap <= PRIMARY_ENDPOINT_TOLERANCE
        ]
        separated = [
            row
            for row in matched
            if row.maximum_interior_trajectory_gap >= 0.2
        ]
        output.append(
            {
                "omitted_topic_id": omitted,
                "nonsaturated": len(nonsaturated),
                "endpoint_matched": len(matched),
                "endpoint_matched_and_separated": len(separated),
                "endpoint_match_rate": len(matched) / len(nonsaturated),
                "unconditional_match_and_separation_rate": (
                    len(separated) / len(nonsaturated)
                ),
                "conditional_separation_rate": (
                    len(separated) / len(matched) if matched else None
                ),
                "median_maximum_interior_trajectory_gap": statistics.median(
                    row.maximum_interior_trajectory_gap for row in matched
                ),
            }
        )
    return output


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise AnalysisError(f"Refusing to write empty CSV {path}.")
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def find_summary(
    summaries: Sequence[Mapping[str, object]],
    comparison: str,
    tolerance: float,
    stratum: str,
) -> Mapping[str, object]:
    matches = [
        row
        for row in summaries
        if row["comparison"] == comparison
        and float(row["endpoint_tolerance"]) == tolerance
        and row["response_stratum"] == stratum
    ]
    if len(matches) != 1:
        raise AnalysisError("Could not select one endpoint summary row.")
    return matches[0]


def find_interval(
    intervals: Sequence[Mapping[str, object]],
    comparison: str,
    tolerance: float,
    stratum: str,
    cluster_field: str,
    metric: str,
) -> Mapping[str, object]:
    matches = [
        row
        for row in intervals
        if row["comparison"] == comparison
        and float(row["endpoint_tolerance"]) == tolerance
        and row["response_stratum"] == stratum
        and row["cluster_field"] == cluster_field
        and row["metric"] == metric
    ]
    if len(matches) != 1:
        raise AnalysisError("Could not select one endpoint interval row.")
    return matches[0]


def find_rate_interval(
    rows: Sequence[Mapping[str, object]],
    scheme: str,
    metric: str,
) -> Mapping[str, object]:
    matches = [
        row
        for row in rows
        if row["resampling_scheme"] == scheme and row["metric"] == metric
    ]
    if len(matches) != 1:
        raise AnalysisError("Could not select one full-denominator interval.")
    return matches[0]


def build_report(
    participants: Sequence[Participant],
    endpoint_summaries: Sequence[Mapping[str, object]],
    rate_intervals: Sequence[Mapping[str, object]],
    grid_summary: Sequence[Mapping[str, object]],
    heterogeneity_rows: Sequence[Mapping[str, object]],
    weighting_rows: Sequence[Mapping[str, object]],
    duplicate_rows: Sequence[Mapping[str, object]],
    leave_topic_rows: Sequence[Mapping[str, object]],
    all_four_rows: Sequence[Mapping[str, object]],
    ratio_summary: Sequence[Mapping[str, object]],
    checks: Mapping[str, object],
    bootstrap_replicates: int,
) -> str:
    primary = find_summary(
        endpoint_summaries,
        "direction_oriented",
        PRIMARY_ENDPOINT_TOLERANCE,
        "all",
    )
    tolerance_rows = {
        tolerance: find_summary(
            endpoint_summaries, "direction_oriented", tolerance, "all"
        )
        for tolerance in ENDPOINT_TOLERANCES
    }
    fixed_group_ci = {
        metric: find_rate_interval(
            rate_intervals,
            "fixed_topics_groups_within_topic",
            metric,
        )
        for metric in (
            "endpoint_match_rate",
            "unconditional_match_and_separation_rate",
            "conditional_separation_rate",
            "median_maximum_interior_trajectory_gap",
        )
    }
    hierarchical_ci = {
        metric: find_rate_interval(
            rate_intervals,
            "hierarchical_topic_then_group",
            metric,
        )
        for metric in (
            "endpoint_match_rate",
            "unconditional_match_and_separation_rate",
            "conditional_separation_rate",
        )
    }
    primary_weighting = {
        str(row["weighting"]): row
        for row in weighting_rows
        if row["comparison"] == "direction_oriented"
        and float(row["endpoint_tolerance"]) == PRIMARY_ENDPOINT_TOLERANCE
    }
    orientation_rows = sorted(
        (
            row
            for row in heterogeneity_rows
            if row["dimension"] == "uptake_leverage_orientation"
            and row["level"] != "unorientable_zero_uptake_leverage"
        ),
        key=lambda row: str(row["level"]),
    )
    relation_rows = sorted(
        (
            row
            for row in heterogeneity_rows
            if row["dimension"] == "evidence_relation"
        ),
        key=lambda row: str(row["level"]),
    )
    initial_rows = sorted(
        (
            row
            for row in heterogeneity_rows
            if row["dimension"] == "initial_stance_cell"
        ),
        key=lambda row: float(row["level"]),
    )
    ratio_all = next(
        row
        for row in ratio_summary
        if row["dimension"] == "all" and row["level"] == "all"
    )
    all_four_primary = next(
        row
        for row in all_four_rows
        if float(row["endpoint_tolerance"]) == PRIMARY_ENDPOINT_TOLERANCE
        and row["uptake_leverage_orientation"] == "all"
        and row["nonsaturation_rule"] == "all_four_cells_nonsaturated"
    )
    grid_primary = next(
        row
        for row in grid_summary
        if float(row["endpoint_tolerance"]) == PRIMARY_ENDPOINT_TOLERANCE
        and row["response_stratum"] == "all"
    )
    strata_counts = Counter(row.response_stratum for row in participants)
    shared_stream = {
        str(row["sample"]): row for row in duplicate_rows
    }["one_oriented_record_per_evidence_stream_id"]
    leave_conditional = [
        float(row["conditional_separation_rate"])
        for row in leave_topic_rows
        if row["conditional_separation_rate"] is not None
    ]
    leave_joint = [
        float(row["unconditional_match_and_separation_rate"])
        for row in leave_topic_rows
    ]

    lines = [
        "# Uptake–anchoring trajectory analysis",
        "",
        "## Status",
        "",
        "This is a transparent **post-hoc fixed-stream robustness analysis**, "
        "not a preregistered experiment. The v2 specification records that its "
        "direction rule was amended after inspection of the first smoke run.",
    ]
    if bootstrap_replicates != BOOTSTRAP_REPLICATES:
        lines.extend(
            [
                "",
                f"> **SMOKE OUTPUT:** {bootstrap_replicates} bootstrap "
                f"replicates were used instead of the registered "
                f"{BOOTSTRAP_REPLICATES}; do not quote its intervals as final.",
            ]
        )
    lines.extend(
        [
            "",
            "## Straight answer",
            "",
            "No: strong evidence uptake is not automatically weak prior "
            "anchoring. Anchoring changes the prior term once; uptake changes "
            "each later evidence increment. Their effects can cancel at the "
            "final step, but that does not make the controls—or their paths—the "
            "same.",
            "",
            f"Of {primary['nonsaturated_n']} orientable, minimum-three-item, "
            f"pair-nonsaturated records, {primary['endpoint_matched_n']} ended "
            "within 0.02 under the algebraically appropriate pair of the four "
            "factorial cells. "
            f"{primary['endpoint_matched_and_separated_n']}/"
            f"{primary['endpoint_matched_n']} "
            f"({fmt(100 * float(primary['separation_rate_ge_0_2']), 1)}%) "
            "still differed by at least 0.20 at an interior evidence step. "
            f"The median maximum interior gap was "
            f"{fmt(primary['median_maximum_interior_trajectory_gap'])}. These "
            f"matches span {primary['matched_group_clusters']} group IDs and "
            f"{primary['matched_topic_clusters']} topics.",
            "",
            "The registered fixed-topic, groups-within-topic bootstrap gives:",
            "",
            "| Estimand | Estimate | 95% interval |",
            "|---|---:|---:|",
            (
                "| Endpoint match rate | "
                f"{fmt(100 * float(fixed_group_ci['endpoint_match_rate']['estimate']), 2)}% | "
                f"[{fmt(100 * float(fixed_group_ci['endpoint_match_rate']['ci_low']), 2)}%, "
                f"{fmt(100 * float(fixed_group_ci['endpoint_match_rate']['ci_high']), 2)}%] |"
            ),
            (
                "| Match + interior separation rate | "
                f"{fmt(100 * float(fixed_group_ci['unconditional_match_and_separation_rate']['estimate']), 2)}% | "
                f"[{fmt(100 * float(fixed_group_ci['unconditional_match_and_separation_rate']['ci_low']), 2)}%, "
                f"{fmt(100 * float(fixed_group_ci['unconditional_match_and_separation_rate']['ci_high']), 2)}%] |"
            ),
            (
                "| Separation conditional on matching | "
                f"{fmt(100 * float(fixed_group_ci['conditional_separation_rate']['estimate']), 1)}% | "
                f"[{fmt(100 * float(fixed_group_ci['conditional_separation_rate']['ci_low']), 1)}%, "
                f"{fmt(100 * float(fixed_group_ci['conditional_separation_rate']['ci_high']), 1)}%] |"
            ),
            (
                "| Median maximum interior gap | "
                f"{fmt(fixed_group_ci['median_maximum_interior_trajectory_gap']['estimate'])} | "
                f"[{fmt(fixed_group_ci['median_maximum_interior_trajectory_gap']['ci_low'])}, "
                f"{fmt(fixed_group_ci['median_maximum_interior_trajectory_gap']['ci_high'])}] |"
            ),
            "",
            "The endpoint-match rate is a design-specific frequency. The "
            "conditional result is a practical endpoint-equivalence "
            "demonstration at resolution 0.02: when the outcome is effectively "
            "the same at that resolution, the trajectories usually are not. "
            "Structural many-to-one behavior follows from the analytic mapping "
            "below.",
            "",
            "![Two endpoint-matched fixed-stream trajectory examples](trajectory_examples.svg)",
            "",
            "## Exact reason the compensating pair changes by group",
            "",
            "For prior log odds `P` and cumulative evidence `E_t(u)`, the "
            "policy state is:",
            "",
            "`L_t(a,u) = a P + E_t(u)`.",
            "",
            "Let `D = E_T(0.6) - E_T(0.05)`. Then low-anchor/high-uptake "
            "versus high-anchor/low-uptake can match when `D = 0.8P` "
            "(prior-consistent uptake leverage). When uptake leverage opposes "
            "the prior, the compensating comparison flips: low/low versus "
            "high/high can match when `D = -0.8P`.",
            "",
            "The required cross-pair anchor difference `D/P` is not universal: "
            f"it is positive for {ratio_all['positive_ratio_n']} records, "
            f"negative for {ratio_all['negative_ratio_n']}, and zero for "
            f"{ratio_all['zero_ratio_n']}; its median is "
            f"{fmt(ratio_all['median'])} and IQR is "
            f"[{fmt(ratio_all['q25'])}, {fmt(ratio_all['q75'])}]. Its sign "
            "therefore changes across streams, which directly rules out one "
            "automatic high-uptake/low-anchor tradeoff.",
            "",
            "Define `D_t = E_t(0.6) - E_t(0.05)`. For exact endpoint equality, "
            "the absolute latent-log-odds path gap at step `t` is "
            "`|D_t - D_T|`. Under the empirical 0.02 stance-scale criterion, "
            "the signed latent gap additionally contains the terminal latent "
            "residual `g_T`, and the reported stance gap is its "
            "tanh-transformed counterpart. The replay checks recover both "
            "endpoint decompositions to numerical precision.",
            "",
            "- Changing uptake at fixed anchoring changes the anchored initial "
            "state by at most "
            f"{fmt(checks['max_anchored_initial_state_change_when_only_uptake_changes'], 12)}.",
            "- Changing anchoring at fixed uptake changes an evidence log-odds "
            "increment by at most "
            f"{fmt(checks['max_log_odds_increment_change_when_only_anchor_changes'], 12)}.",
            "- Maximum cross- and main-diagonal endpoint decomposition errors "
            "are "
            f"{fmt(checks['max_cross_endpoint_decomposition_error'], 12)} and "
            f"{fmt(checks['max_diagonal_endpoint_decomposition_error'], 12)}.",
            "",
            "## What changes across groups",
            "",
            "The exact direction split materially changes the empirical rate:",
            "",
            "| Exact uptake-leverage direction | Nonsaturated n | Matched | Separated ≥0.20 | Conditional separation | Median interior gap |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in orientation_rows:
        separated = int(row["endpoint_matched_and_separated"])
        conditional = (
            f"{fmt(100 * float(row['conditional_separation_rate']), 1)}%"
            if row["conditional_separation_rate"] is not None
            else "NA"
        )
        lines.append(
            f"| {str(row['level']).replace('_', ' ')} | "
            f"{row['nonsaturated']} | {row['endpoint_matched']} | "
            f"{separated} | {conditional} | "
            f"{fmt(row['median_maximum_interior_trajectory_gap'])} |"
        )
    lines.extend(
        [
            "",
            "The raw-pressure split is a secondary descriptive proxy that "
            "keeps its weak/balanced category visible; it disagrees with the "
            "exact uptake-leverage direction in 12 weak/balanced records:",
            "",
            "| Raw evidence relation | Nonsaturated n | Match rate | Joint rate | Conditional separation |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in relation_rows:
        conditional = (
            f"{fmt(100 * float(row['conditional_separation_rate']), 1)}%"
            if row["conditional_separation_rate"] is not None
            else "NA"
        )
        lines.append(
            f"| {str(row['level']).replace('_', ' ')} | "
            f"{row['nonsaturated']} | "
            f"{fmt(100 * float(row['endpoint_match_rate']), 1)}% | "
            f"{fmt(100 * float(row['unconditional_match_and_separation_rate']), 1)}% | "
            f"{conditional} |"
        )
    lines.extend(
        [
            "",
            "Signed prior cells are also highly heterogeneous:",
            "",
            "| Observed initial stance | Nonsaturated n | Matched | Match rate | Separated / matched |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in initial_rows:
        lines.append(
            f"| {float(row['level']):+.3f} | {row['nonsaturated']} | "
            f"{row['endpoint_matched']} | "
            f"{fmt(100 * float(row['endpoint_match_rate']), 1)}% | "
            f"{row['endpoint_matched_and_separated']}/"
            f"{row['endpoint_matched']} |"
        )
    lines.extend(
        [
            "",
            "Thus the pooled match frequency must not be described as a "
            "population prevalence. The algebraic distinction is invariant; "
            "how often a particular grid happens to compensate is "
            "group-composition dependent.",
            "",
            "The source population comprises 2,495 occurrence-level "
            "participant records, not 2,495 unique evidence streams: there "
            "are 2,482 evidence-stream IDs, 704 group IDs, and 107 topics. "
            "Group-size sensitivities use "
            "both participant-record count and distinct evidence-stream count "
            "per group; they do not label record count as distinct "
            "deliberators.",
            "",
            "## Weighting and sensitivity checks",
            "",
            "| Weighting | Match rate | Joint rate | Conditional separation |",
            "|---|---:|---:|---:|",
        ]
    )
    for weighting in ("participant", "equal_group", "equal_topic"):
        row = primary_weighting[weighting]
        lines.append(
            f"| {weighting.replace('_', ' ')} | "
            f"{fmt(100 * float(row['endpoint_match_rate']), 2)}% | "
            f"{fmt(100 * float(row['unconditional_match_and_separation_rate']), 2)}% | "
            f"{fmt(100 * float(row['conditional_separation_rate']), 1)}% |"
        )
    lines.extend(
        [
            "",
            "Sensitivity results:",
            "",
            f"- At endpoint tolerances 0.01, 0.02, and 0.05, respectively, "
            f"{tolerance_rows[0.01]['endpoint_matched_and_separated_n']}/"
            f"{tolerance_rows[0.01]['endpoint_matched_n']}, "
            f"{tolerance_rows[0.02]['endpoint_matched_and_separated_n']}/"
            f"{tolerance_rows[0.02]['endpoint_matched_n']}, and "
            f"{tolerance_rows[0.05]['endpoint_matched_and_separated_n']}/"
            f"{tolerance_rows[0.05]['endpoint_matched_n']} matches separate.",
            f"- Requiring all four factorial cells, rather than only the "
            f"selected pair, to be nonsaturated leaves "
            f"{all_four_primary['endpoint_matched_and_separated']}/"
            f"{all_four_primary['endpoint_matched']} separated matches.",
            f"- The hierarchical topic-then-group interval is "
            f"[{fmt(100 * float(hierarchical_ci['unconditional_match_and_separation_rate']['ci_low']), 2)}%, "
            f"{fmt(100 * float(hierarchical_ci['unconditional_match_and_separation_rate']['ci_high']), 2)}%] "
            f"for the joint rate and "
            f"[{fmt(100 * float(hierarchical_ci['conditional_separation_rate']['ci_low']), 1)}%, "
            f"{fmt(100 * float(hierarchical_ci['conditional_separation_rate']['ci_high']), 1)}%] "
            "for conditional separation.",
            f"- Keeping one occurrence per evidence-stream ID leaves "
            f"{shared_stream['endpoint_matched']} matches and "
            f"{shared_stream['endpoint_matched_and_separated']} separated; "
            "this is a shared-stream sensitivity, not a claim that repeated "
            "occurrences are duplicate people or duplicate outcomes.",
            f"- Across 107 leave-one-topic-out analyses, conditional separation "
            f"ranges from {fmt(100 * min(leave_conditional), 1)}% to "
            f"{fmt(100 * max(leave_conditional), 1)}%; the joint rate ranges "
            f"from {fmt(100 * min(leave_joint), 2)}% to "
            f"{fmt(100 * max(leave_joint), 2)}%.",
            "",
            "## Outcome-defined groups are secondary",
            "",
            "| Diagnostic observed-response stratum | Total records | Matched | Separated ≥0.20 |",
            "|---|---:|---:|---:|",
        ]
    )
    for stratum in STRATA:
        row = find_summary(
            endpoint_summaries,
            "direction_oriented",
            PRIMARY_ENDPOINT_TOLERANCE,
            stratum,
        )
        lines.append(
            f"| {stratum.replace('_', ' ')} | {strata_counts[stratum]} | "
            f"{row['endpoint_matched_n']} | "
            f"{row['endpoint_matched_and_separated_n']} |"
        )
    lines.extend(
        [
            "",
            "These five groups condition on observed human movement and are "
            "diagnostic, not prospectively assignable types. They cannot "
            "support subgroup prevalence or causal claims.",
            "",
            "## Exploratory full-grid check",
            "",
            "The 13×13 grid search is explicitly exploratory. At tolerance "
            f"0.02, {grid_primary['participants_with_endpoint_equivalent_pairs']}/"
            f"{grid_primary['participants']} eligible records have at least "
            "one nonsaturated endpoint-equivalent pair. After equal weighting "
            "within participant, the mean fraction separated by at least 0.20 "
            "is "
            f"{fmt(100 * float(grid_primary['mean_participant_separated_pair_fraction']), 1)}%.",
            "",
            "## Claim boundary",
            "",
            "Supported: in the externally specified BE update policy, final "
            "stance does not uniquely identify evidence uptake versus prior "
            "anchoring. Fixed received-evidence streams can yield nearly equal "
            "endpoints but different anchored starts, interior paths, and "
            "whether a side switch occurs.",
            "",
            "Not supported: identification of human psychological parameters, "
            "causal human belief trajectories, generated-dialogue effects, or "
            "population-general subgroup prevalence.",
            "",
            "Machine-readable counts, intervals, paired rows, heterogeneity "
            "tables, exact mechanism checks, and the smoke-amended v2 protocol "
            "are retained beside this report.",
            "",
        ]
    )
    return "\n".join(lines)


def build_rebuttal_insert(
    endpoint_summaries: Sequence[Mapping[str, object]],
    rate_intervals: Sequence[Mapping[str, object]],
    heterogeneity_rows: Sequence[Mapping[str, object]],
    weighting_rows: Sequence[Mapping[str, object]],
    bootstrap_replicates: int,
) -> str:
    primary = find_summary(
        endpoint_summaries,
        "direction_oriented",
        PRIMARY_ENDPOINT_TOLERANCE,
        "all",
    )
    conditional_ci = find_rate_interval(
        rate_intervals,
        "fixed_topics_groups_within_topic",
        "conditional_separation_rate",
    )
    joint_ci = find_rate_interval(
        rate_intervals,
        "fixed_topics_groups_within_topic",
        "unconditional_match_and_separation_rate",
    )
    orientation = {
        str(row["level"]): row
        for row in heterogeneity_rows
        if row["dimension"] == "uptake_leverage_orientation"
    }
    consistent = orientation["prior_consistent_uptake_leverage"]
    opposing = orientation["prior_opposing_uptake_leverage"]
    weighting = {
        str(row["weighting"]): row
        for row in weighting_rows
        if row["comparison"] == "direction_oriented"
        and float(row["endpoint_tolerance"]) == PRIMARY_ENDPOINT_TOLERANCE
    }
    lines = [
            "Evidence uptake and prior anchoring are not inverse parameters. "
            "For the externally specified BE update policy, "
            "`L_t(a,u)=aP+E_t(u)`: anchoring `a` scales the prior log odds "
            "`P` once, whereas uptake `u` changes every later evidence "
            "increment. Let `D=E_T(0.6)-E_T(0.05)`. When uptake leverage is "
            "prior-consistent, low-anchor/high-uptake can match "
            "high-anchor/low-uptake if `D≈0.8P`; when uptake leverage is "
            "prior-opposing, the compensating pair flips to low/low versus "
            "high/high if `D≈-0.8P`. Thus even the direction of the apparent "
            "tradeoff depends on the stream-specific uptake leverage. Equal "
            "final stance only constrains the cumulative value at `T`; it does "
            "not constrain the intermediate cumulative path.",
            "",
            "We tested this post hoc by replaying the identical verified "
            "evidence stream for each occurrence through all four factorial "
            "cells. Under the disclosed post-smoke-amended, outcome-free "
            "orientation rule, each record's pair is selected from `P` and `D` "
            "alone, not from that record's endpoint gap. With a locked "
            "final-stance tolerance of 0.02, "
            f"{primary['endpoint_matched_n']}/"
            f"{primary['nonsaturated_n']} eligible records matched endpoints; "
            f"{primary['endpoint_matched_and_separated_n']}/"
            f"{primary['endpoint_matched_n']} "
            f"({fmt(100 * float(primary['separation_rate_ge_0_2']), 1)}%; "
            "fixed-topic groups-within-topic 95% interval "
            f"[{fmt(100 * float(conditional_ci['ci_low']), 1)}%, "
            f"{fmt(100 * float(conditional_ci['ci_high']), 1)}%]) differed by "
            "at least 0.20 at an interior step; the median maximum interior "
            f"gap was {fmt(primary['median_maximum_interior_trajectory_gap'])}. "
            f"The unconditional match-and-separation rate was "
            f"{fmt(100 * float(primary['unconditional_match_and_separation_rate']), 2)}% "
            f"[{fmt(100 * float(joint_ci['ci_low']), 2)}%, "
            f"{fmt(100 * float(joint_ci['ci_high']), 2)}%]. Results are "
            "heterogeneous, as expected: the prior-consistent direction had "
            f"{consistent['endpoint_matched_and_separated']}/"
            f"{consistent['endpoint_matched']} separated matches, whereas the "
            "prior-opposing direction had "
            f"{opposing['endpoint_matched_and_separated']}/"
            f"{opposing['endpoint_matched']}. Participant-, equal-group-, and "
            "equal-topic-weighted conditional estimates were "
            f"{fmt(100 * float(weighting['participant']['conditional_separation_rate']), 1)}%, "
            f"{fmt(100 * float(weighting['equal_group']['conditional_separation_rate']), 1)}%, "
            "and "
            f"{fmt(100 * float(weighting['equal_topic']['conditional_separation_rate']), 1)}%. "
            "We therefore interpret endpoint-match frequency as "
            "group-composition-specific. The analytic mapping establishes "
            "structural endpoint compensation, while these matched records "
            "show operationally that an endpoint observed at resolution 0.02 "
            "cannot distinguish the settings. This is a policy-level result, "
            "not identification of human psychological parameters.",
            "",
        ]
    if bootstrap_replicates != BOOTSTRAP_REPLICATES:
        lines.insert(
            0,
            f"SMOKE OUTPUT ONLY ({bootstrap_replicates}/"
            f"{BOOTSTRAP_REPLICATES} registered bootstrap replicates).",
        )
        lines.insert(1, "")
    return "\n".join(lines)


def write_manifest(
    output_dir: Path,
    input_dir: Path,
    input_manifest: dict,
    spec_path: Path,
) -> None:
    artifact_rows = []
    for path in sorted(output_dir.iterdir()):
        if path.name == "manifest.json" or not path.is_file():
            continue
        artifact_rows.append(
            {
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "schema_version": "uptake-anchor-trajectory-posthoc-v2",
        "study_id": "neurips_2026_uptake_anchor_trajectory_posthoc_v2",
        "classification": "posthoc_robustness_analysis",
        "analysis_spec": {
            "path": spec_path.name,
            "sha256": sha256_file(spec_path),
        },
        "input_bundle": {
            "study_id": input_manifest.get("study_id"),
            "manifest_sha256": sha256_file(input_dir / "manifest.json"),
            "accepted_stream_sha256": EXPECTED_ACCEPTED_SHA256,
            "predictions_sha256": EXPECTED_PREDICTIONS_SHA256,
        },
        "artifacts": artifact_rows,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_analysis(
    input_dir: Path,
    output_dir: Path,
    spec_path: Path,
    bootstrap_replicates: int,
) -> None:
    if output_dir.exists():
        raise AnalysisError(f"Output directory already exists: {output_dir}")
    if not spec_path.is_file():
        raise AnalysisError(f"Analysis specification does not exist: {spec_path}")
    actual_spec_hash = sha256_file(spec_path)
    if actual_spec_hash != EXPECTED_SPEC_SHA256:
        raise AnalysisError(
            "Analysis specification SHA-256 mismatch: "
            f"actual={actual_spec_hash}, expected={EXPECTED_SPEC_SHA256}"
        )

    participants, input_manifest = load_participants(input_dir)
    factorial_rows, trajectories = condition_rows(participants)
    factorial_summary = summarize_factorial_cells(factorial_rows)
    contrast_records = paired_contrast_records(factorial_rows)
    contrast_summary = contrast_bootstrap_summaries(
        contrast_records, bootstrap_replicates
    )
    fixed_comparisons = build_endpoint_comparisons(participants, trajectories)
    direction_oriented = build_direction_oriented_comparison(
        participants, fixed_comparisons
    )
    comparisons = {
        **fixed_comparisons,
        "direction_oriented": direction_oriented,
    }
    endpoint_summaries, primary_endpoint_rows = summarize_endpoint_rows(
        comparisons
    )
    if len(primary_endpoint_rows) != len(
        {row["participant_record_id"] for row in primary_endpoint_rows}
    ):
        raise AnalysisError("Primary matched-participant rows are not unique.")
    endpoint_intervals = endpoint_bootstrap_intervals(
        comparisons, bootstrap_replicates
    )
    rate_intervals = directional_rate_bootstrap_intervals(
        direction_oriented, bootstrap_replicates
    )
    checks = mechanism_checks(participants, trajectories)
    if checks["raw_pressure_vs_exact_orientation_disagreements"] != 12:
        raise AnalysisError("Raw-pressure orientation audit changed.")
    ratio_summary, ratio_participant_rows = compensation_ratio_rows(participants)
    grid_summary, grid_participant_rows = grid_robustness(participants)
    heterogeneity_rows = primary_heterogeneity_summary(
        participants, direction_oriented
    )
    weighting_rows = weighting_summaries(comparisons)
    duplicate_rows = duplicate_stream_sensitivity(
        participants, direction_oriented
    )
    all_four_rows = all_four_cell_nonsaturation_sensitivity(
        participants, direction_oriented, trajectories
    )
    leave_topic_rows = leave_one_topic_out(direction_oriented)
    example_rows, example_svg = trajectory_examples(
        direction_oriented, trajectories
    )
    cluster_rows = [
        *cluster_primary_summary(participants, direction_oriented, "group_id"),
        *cluster_primary_summary(participants, direction_oriented, "topic_id"),
    ]

    stratum_counts = Counter(row.response_stratum for row in participants)
    stratum_rows = []
    for stratum in STRATA:
        selected = [row for row in participants if row.response_stratum == stratum]
        stratum_rows.append(
            {
                "response_stratum": stratum,
                "participant_records": stratum_counts[stratum],
                "group_clusters": len({row.group_id for row in selected}),
                "topic_clusters": len({row.topic_id for row in selected}),
                "mean_evidence_count": statistics.fmean(
                    len(row.evidence) for row in selected
                ),
                "median_evidence_count": statistics.median(
                    len(row.evidence) for row in selected
                ),
            }
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        write_csv(temporary / "stratum_counts.csv", stratum_rows)
        write_csv(temporary / "factorial_condition_summary.csv", factorial_summary)
        write_csv(temporary / "factorial_contrasts.csv", contrast_summary)
        write_csv(temporary / "endpoint_summary.csv", endpoint_summaries)
        write_csv(
            temporary / "matched_record_cluster_bootstrap_intervals.csv",
            endpoint_intervals,
        )
        write_csv(
            temporary / "primary_full_denominator_bootstrap_intervals.csv",
            rate_intervals,
        )
        write_csv(
            temporary / "primary_endpoint_matched_participants.csv",
            primary_endpoint_rows,
        )
        write_csv(temporary / "cluster_heterogeneity.csv", cluster_rows)
        write_csv(temporary / "primary_heterogeneity.csv", heterogeneity_rows)
        write_csv(temporary / "weighting_sensitivity.csv", weighting_rows)
        write_csv(
            temporary / "all_four_cell_nonsaturation_sensitivity.csv",
            all_four_rows,
        )
        write_csv(
            temporary / "duplicate_stream_sensitivity.csv", duplicate_rows
        )
        write_csv(
            temporary / "compensation_ratio_summary.csv", ratio_summary
        )
        write_csv(
            temporary / "compensation_ratio_participants.csv",
            ratio_participant_rows,
        )
        write_csv(temporary / "trajectory_examples.csv", example_rows)
        (temporary / "trajectory_examples.svg").write_text(
            example_svg, encoding="utf-8"
        )
        write_csv(temporary / "leave_one_topic_out.csv", leave_topic_rows)
        write_csv(temporary / "grid_robustness_summary.csv", grid_summary)
        write_csv(
            temporary / "grid_participant_summary.csv", grid_participant_rows
        )
        (temporary / "mechanism_checks.json").write_text(
            json.dumps(checks, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        analysis = {
            "schema_version": "uptake-anchor-trajectory-analysis-v2",
            "classification": "posthoc_robustness_analysis",
            "participant_records": len(participants),
            "distinct_evidence_stream_ids": len(
                {row.evidence_stream_id for row in participants}
            ),
            "group_clusters": len({row.group_id for row in participants}),
            "topic_clusters": len({row.topic_id for row in participants}),
            "direction_oriented_records": len(direction_oriented),
            "factorial": {
                "low_anchor": LOW_ANCHOR,
                "high_anchor": HIGH_ANCHOR,
                "low_uptake": LOW_UPTAKE,
                "high_uptake": HIGH_UPTAKE,
            },
            "endpoint_tolerances": list(ENDPOINT_TOLERANCES),
            "primary_endpoint_tolerance": PRIMARY_ENDPOINT_TOLERANCE,
            "minimum_arguments": MIN_ARGUMENTS,
            "saturation_bound": SATURATION_BOUND,
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_matches_registered_replicates": (
                bootstrap_replicates == BOOTSTRAP_REPLICATES
            ),
            "bootstrap_seed": BOOTSTRAP_SEED,
            "mechanism_checks": checks,
            "primary_endpoint_summary": find_summary(
                endpoint_summaries,
                "direction_oriented",
                PRIMARY_ENDPOINT_TOLERANCE,
                "all",
            ),
        }
        (temporary / "analysis.json").write_text(
            json.dumps(analysis, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "REPORT.md").write_text(
            build_report(
                participants,
                endpoint_summaries,
                rate_intervals,
                grid_summary,
                heterogeneity_rows,
                weighting_rows,
                duplicate_rows,
                leave_topic_rows,
                all_four_rows,
                ratio_summary,
                checks,
                bootstrap_replicates,
            ),
            encoding="utf-8",
        )
        (temporary / "REBUTTAL_INSERT.md").write_text(
            build_rebuttal_insert(
                endpoint_summaries,
                rate_intervals,
                heterogeneity_rows,
                weighting_rows,
                bootstrap_replicates,
            ),
            encoding="utf-8",
        )
        shutil.copyfile(spec_path, temporary / spec_path.name)
        write_manifest(temporary, input_dir, input_manifest, spec_path)
        os.replace(temporary, output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=BOOTSTRAP_REPLICATES,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates <= 0:
        raise AnalysisError("Bootstrap replicates must be positive.")
    run_analysis(
        args.input_dir.resolve(),
        args.output_dir.resolve(),
        args.spec.resolve(),
        args.bootstrap_replicates,
    )


if __name__ == "__main__":
    main()
