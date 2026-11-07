#!/usr/bin/env python3
"""Analyze the preregistered matched-baseline stance trajectories.

The script treats a topic-by-paired-seed debate as the experimental unit.  It
rejects mixed/internal stance sources, incomplete trajectories, and unpaired
condition cells rather than silently falling back to available values.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


PRIMARY_CONDITION = "be_gpt54mini"
COMPARATORS = ("prompt_self_update_gpt54mini", "rag_self_update_gpt54mini")
EXPECTED_CONDITIONS = (PRIMARY_CONDITION, *COMPARATORS)
RUN_KEYS = ["run_id", "topic", "condition", "trial", "seed"]
PAIR_KEYS = ["topic", "trial", "seed"]
METRICS = (
    "final_gap",
    "gap_reduction",
    "mean_absolute_shift",
    "crossing_rate",
    "trajectory_gap_auc",
)
DIRECTIONAL_METRICS = (
    "final_midpoint",
    "center_shift",
    "absolute_center_shift",
    "final_same_nonzero_side_rate",
    "final_shared_positive",
    "final_shared_negative",
    "pro_final_side_switch_rate",
    "con_final_side_switch_rate",
    "pro_any_time_zero_cross_rate",
    "con_any_time_zero_cross_rate",
)
DIRECTIONAL_METRIC_GROUPS = {
    "Center movement": (
        "final_midpoint",
        "center_shift",
        "absolute_center_shift",
    ),
    "Final shared side": (
        "final_same_nonzero_side_rate",
        "final_shared_positive",
        "final_shared_negative",
    ),
    "Agent switching and crossing": (
        "pro_final_side_switch_rate",
        "con_final_side_switch_rate",
        "pro_any_time_zero_cross_rate",
        "con_any_time_zero_cross_rate",
    ),
}
DIRECTIONAL_METRIC_LABELS = {
    "final_midpoint": "Final midpoint",
    "center_shift": "Signed center shift",
    "absolute_center_shift": "Absolute center shift",
    "final_same_nonzero_side_rate": "Same nonzero side rate",
    "final_shared_positive": "Shared positive rate",
    "final_shared_negative": "Shared negative rate",
    "pro_final_side_switch_rate": "Pro final side-switch rate",
    "con_final_side_switch_rate": "Con final side-switch rate",
    "pro_any_time_zero_cross_rate": "Pro any-time zero-cross rate",
    "con_any_time_zero_cross_rate": "Con any-time zero-cross rate",
}
DIRECTIONAL_DIAGNOSTIC_DEFINITIONS = {
    "final_midpoint": (
        "(Pro_T + Con_T) / 2. Signed: positive values indicate a final center "
        "on the proposition-supporting side and negative values the opposing side."
    ),
    "center_shift": (
        "Final midpoint minus initial midpoint. Signed so directional drift is "
        "not hidden by the absolute Pro/Con gap."
    ),
    "absolute_center_shift": (
        "Absolute value of the signed center shift, measuring the magnitude of "
        "joint displacement without direction."
    ),
    "final_same_nonzero_side_rate": (
        "Fraction of runs in which both agents finish strictly positive or both "
        "finish strictly negative. A zero endpoint is not herding."
    ),
    "final_shared_positive": (
        "Fraction of runs in which both agents finish strictly positive."
    ),
    "final_shared_negative": (
        "Fraction of runs in which both agents finish strictly negative. The "
        "positive and negative rates partition the same-nonzero-side rate."
    ),
    "pro_final_side_switch_rate": (
        "Fraction of Pro agents ending strictly negative after a nonzero positive "
        "start. Ending at zero is neutralization, not a side switch."
    ),
    "con_final_side_switch_rate": (
        "Fraction of Con agents ending strictly positive after a nonzero negative "
        "start. Ending at zero is neutralization, not a side switch."
    ),
    "pro_any_time_zero_cross_rate": (
        "Fraction of Pro trajectories touching or crossing zero at any observed "
        "post-start round, including trajectories that later return."
    ),
    "con_any_time_zero_cross_rate": (
        "Fraction of Con trajectories touching or crossing zero at any observed "
        "post-start round, including trajectories that later return."
    ),
}

BUNDLE_MANIFEST_SCHEMA_VERSION = 1
BUNDLE_MANIFEST_NAME = "bundle_manifest.json"
PRIMARY_INPUT_INVENTORY_NAME = "primary_input_inventory.json"


def _require_columns(frame: pd.DataFrame, required: Iterable[str]) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required trajectory columns: {', '.join(missing)}")


def _coerce_finite_numeric(
    values: pd.Series,
    *,
    column: str,
    integer: bool = False,
) -> pd.Series:
    """Coerce real numbers while rejecting booleans, NaN, infinity, and fractions."""
    boolean_mask = values.map(lambda value: isinstance(value, (bool, np.bool_)))
    if boolean_mask.any():
        raise ValueError(f"Trajectory column '{column}' contains boolean values.")
    try:
        numeric = pd.to_numeric(values, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Trajectory column '{column}' must contain numeric values."
        ) from exc
    if np.iscomplexobj(numeric.to_numpy()):
        raise ValueError(
            f"Trajectory column '{column}' must contain real numeric values."
        )
    try:
        as_float = numeric.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Trajectory column '{column}' must contain real numeric values."
        ) from exc
    if not np.isfinite(as_float).all():
        raise ValueError(
            f"Trajectory column '{column}' contains missing or non-finite values."
        )
    if integer:
        if not np.equal(as_float, np.floor(as_float)).all():
            raise ValueError(
                f"Trajectory column '{column}' must contain integer values."
            )
        limits = np.iinfo(np.int64)
        if any(
            value < limits.min or value > limits.max
            for value in numeric.tolist()
        ):
            raise ValueError(
                f"Trajectory column '{column}' exceeds the supported integer range."
            )
        return pd.Series(as_float.astype(np.int64), index=values.index, name=column)
    return pd.Series(as_float, index=values.index, name=column)


def _fingerprint_bytes(raw: bytes, *, label: str) -> dict[str, object]:
    """Return a portable content identity for one byte sequence."""
    return {
        "label": label,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "byte_size": len(raw),
    }


def load_trajectory_input(path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    """Read and fingerprint the exact bytes analyzed without exposing a local path."""
    raw = path.read_bytes()
    return pd.read_csv(io.BytesIO(raw)), _fingerprint_bytes(raw, label=path.name)


def load_primary_input_inventory(
    path: Path,
) -> tuple[str, dict[str, object], dict[str, object]]:
    """Load and validate the verifier inventory that authorizes primary analysis."""
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
        payload = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "Primary input inventory must be valid UTF-8 JSON."
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("Primary input inventory must contain a JSON object.")

    completion = payload.get("completion")
    provenance = payload.get("provenance")
    artifacts = payload.get("artifacts")
    registry = payload.get("registry")
    runs = payload.get("runs")
    if not all(
        isinstance(section, dict)
        for section in (completion, provenance, artifacts, registry, runs)
    ):
        raise ValueError(
            "Primary input inventory is missing completion, provenance, "
            "artifacts, registry, or runs metadata."
        )

    errors = []
    if payload.get("status") != "complete":
        errors.append("status is not complete")
    if payload.get("mode") != "complete":
        errors.append("mode is not complete")
    if payload.get("profile") != "primary":
        errors.append("profile is not primary")
    if completion.get("completion_claimed") is not True:
        errors.append("completion is not claimed")
    if completion.get("design_inventory_complete") is not True:
        errors.append("design inventory is incomplete")
    if completion.get("remaining_runs") != 0:
        errors.append("remaining_runs is not zero")
    completed_runs = completion.get("completed_runs")
    expected_runs = completion.get("expected_runs")
    if (
        not isinstance(completed_runs, int)
        or isinstance(completed_runs, bool)
        or completed_runs < 1
        or completed_runs != expected_runs
    ):
        errors.append("completed_runs does not equal a positive expected_runs")
    if payload.get("warnings") not in (None, []):
        errors.append("inventory contains warnings")
    if runs.get("incomplete_directories") not in (None, []):
        errors.append("inventory contains incomplete run directories")
    if runs.get("unique_result_payloads") != completed_runs:
        errors.append("unique_result_payloads does not equal completed_runs")

    tree_sha256 = artifacts.get("tree_sha256")
    execution_commit = provenance.get("execution_commit")
    registry_sha256 = registry.get("sha256")
    for label, value, expected_length in (
        ("artifact tree SHA-256", tree_sha256, 64),
        ("execution commit", execution_commit, 40),
        ("registry SHA-256", registry_sha256, 64),
    ):
        if not isinstance(value, str) or len(value) != expected_length:
            errors.append(f"{label} is missing or malformed")
    if errors:
        raise ValueError(
            "Primary input inventory failed completion checks: "
            + "; ".join(errors)
            + "."
        )

    summary: dict[str, object] = {
        "schema_version": payload.get("schema_version"),
        "status": payload["status"],
        "mode": payload["mode"],
        "profile": payload["profile"],
        "completed_runs": completed_runs,
        "expected_runs": expected_runs,
        "design_inventory_complete": completion["design_inventory_complete"],
        "campaign_tree_sha256": tree_sha256,
        "execution_commit": execution_commit,
        "registry_sha256": registry_sha256,
        "evaluator_fingerprint_sha256": provenance.get(
            "evaluator_fingerprint_sha256"
        ),
        "extractor_fingerprint_sha256": provenance.get(
            "extractor_fingerprint_sha256"
        ),
    }
    return text, _fingerprint_bytes(raw, label=path.name), summary


def analyzer_provenance(script_path: Path) -> dict[str, object]:
    """Fingerprint this script and bind it to its containing Git commit."""
    resolved_script = script_path.resolve()

    def git(*arguments: str) -> str:
        try:
            result = subprocess.run(
                ["git", "-C", str(resolved_script.parent), *arguments],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(
                "Cannot determine analyzer Git provenance."
            ) from exc
        return result.stdout.strip()

    repository_root = Path(git("rev-parse", "--show-toplevel")).resolve()
    try:
        relative_script = resolved_script.relative_to(repository_root).as_posix()
    except ValueError as exc:
        raise RuntimeError(
            "Analyzer script is outside the detected Git repository."
        ) from exc
    raw = resolved_script.read_bytes()
    script_status = git(
        "status",
        "--porcelain",
        "--untracked-files=no",
        "--",
        relative_script,
    )
    worktree_status = git("status", "--porcelain", "--untracked-files=no")
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_worktree_dirty": bool(worktree_status),
        "script_tracked_dirty": bool(script_status),
        "script": {
            "path": relative_script,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "byte_size": len(raw),
        },
    }


def portable_invocation(args: argparse.Namespace) -> dict[str, object]:
    """Describe all effective CLI arguments without machine-local paths."""
    script = "rebuttal/scripts/analyze_matched_baseline.py"
    argv = [
        "python3.11",
        script,
        "<PRIMARY_CAMPAIGN_DIR>/stance_trajectories.csv",
        "--primary-inventory",
        "<VERIFIED_PRIMARY_INVENTORY_JSON>",
        "--output-dir",
        "<NEW_OUTPUT_DIR>",
        "--expected-final-round",
        str(args.expected_final_round),
        "--bootstrap-replicates",
        str(args.bootstrap_replicates),
        "--random-seed",
        str(args.random_seed),
        "--pair-policy",
        str(args.pair_policy),
        "--analysis-label",
        str(args.analysis_label),
    ]
    return {
        "working_directory": "<REPOSITORY_ROOT>",
        "argv": argv,
        "effective_arguments": {
            "trajectory_csv": (
                "<PRIMARY_CAMPAIGN_DIR>/stance_trajectories.csv"
            ),
            "primary_inventory": "<VERIFIED_PRIMARY_INVENTORY_JSON>",
            "output_dir": "<NEW_OUTPUT_DIR>",
            "expected_final_round": args.expected_final_round,
            "bootstrap_replicates": args.bootstrap_replicates,
            "random_seed": args.random_seed,
            "pair_policy": args.pair_policy,
            "analysis_label": args.analysis_label,
        },
        "path_placeholders": {
            "<REPOSITORY_ROOT>": (
                "checkout containing the recorded analyzer commit"
            ),
            "<PRIMARY_CAMPAIGN_DIR>": (
                "verified matched_baseline_v1 campaign directory"
            ),
            "<VERIFIED_PRIMARY_INVENTORY_JSON>": (
                "verifier output whose content identity is recorded under inputs"
            ),
            "<NEW_OUTPUT_DIR>": (
                "absent destination; the analyzer refuses to overwrite"
            ),
        },
    }


def _artifact_checksums(artifacts: dict[str, str]) -> dict[str, dict[str, object]]:
    """Inventory every non-manifest bundle payload before atomic publication."""
    inventory: dict[str, dict[str, object]] = {}
    for name in sorted(artifacts):
        raw = artifacts[name].encode("utf-8")
        inventory[name] = {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "byte_size": len(raw),
        }
    return inventory


def build_bundle_manifest(
    *,
    args: argparse.Namespace,
    analyzer: dict[str, object],
    trajectory_input: dict[str, object],
    trajectory_rows: int,
    analyzed_runs: int,
    analyzed_topics: int,
    primary_inventory: dict[str, object],
    primary_inventory_summary: dict[str, object],
    artifacts: dict[str, str],
) -> str:
    """Build the deterministic trust-root manifest for an analysis bundle."""
    completed_runs = primary_inventory_summary["completed_runs"]
    if analyzed_runs != completed_runs:
        raise ValueError(
            "Trajectory run count does not match the verified primary inventory: "
            f"{analyzed_runs} != {completed_runs}."
        )
    payload = {
        "schema_version": BUNDLE_MANIFEST_SCHEMA_VERSION,
        "bundle_type": "matched_baseline_direction_analysis",
        "analyzer": analyzer,
        "portable_invocation": portable_invocation(args),
        "inputs": {
            "trajectory_csv": {
                **trajectory_input,
                "rows": trajectory_rows,
                "analyzed_runs": analyzed_runs,
                "analyzed_topics": analyzed_topics,
            },
            "verified_primary_inventory": {
                **primary_inventory,
                "snapshot_output": PRIMARY_INPUT_INVENTORY_NAME,
                "summary": primary_inventory_summary,
            },
        },
        "bootstrap": {
            "method": "topic_then_paired_seed_hierarchical_bootstrap",
            "replicates_per_estimate": args.bootstrap_replicates,
            "condition_metrics": {
                "base_seed": args.random_seed,
                "estimates": len(EXPECTED_CONDITIONS) * len(METRICS),
                "seed_derivation": (
                    "numpy.random.SeedSequence(base_seed).spawn(estimates)"
                ),
            },
            "paired_primary_differences": {
                "base_seed": args.random_seed + 1,
                "estimates": len(COMPARATORS),
                "seed_derivation": (
                    "numpy.random.SeedSequence(base_seed).spawn(estimates)"
                ),
            },
            "directional_condition_metrics": {
                "base_seed": args.random_seed + 2,
                "estimates": len(EXPECTED_CONDITIONS)
                * len(DIRECTIONAL_METRICS),
                "seed_derivation": (
                    "numpy.random.SeedSequence(base_seed).spawn(estimates)"
                ),
            },
        },
        "outputs": _artifact_checksums(artifacts),
        "integrity_scope": (
            "outputs inventories every sibling payload. "
            f"{BUNDLE_MANIFEST_NAME} is the self-referential trust root and "
            "therefore is not listed inside itself."
        ),
    }
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def validate_trajectories(
    frame: pd.DataFrame,
    *,
    conditions: Sequence[str] = EXPECTED_CONDITIONS,
    expected_final_round: int = 15,
) -> pd.DataFrame:
    """Validate and normalize one external-judge trajectory table."""
    _require_columns(
        frame,
        [
            *RUN_KEYS,
            "agent",
            "side",
            "round",
            "stance",
            "stance_source",
        ],
    )
    normalized = frame.copy()
    for column in ("run_id", "topic", "condition", "agent", "side", "stance_source"):
        if normalized[column].isna().any():
            raise ValueError(f"Trajectory column '{column}' contains missing values.")
        normalized[column] = normalized[column].astype(str)
        if normalized[column].str.strip().eq("").any():
            raise ValueError(f"Trajectory column '{column}' contains blank values.")

    for column in ("round", "trial", "seed"):
        normalized[column] = _coerce_finite_numeric(
            normalized[column],
            column=column,
            integer=True,
        )
    normalized["stance"] = _coerce_finite_numeric(
        normalized["stance"],
        column="stance",
    )

    if ((normalized["stance"] < -1.0) | (normalized["stance"] > 1.0)).any():
        raise ValueError("Trajectory table contains stance values outside [-1, 1].")

    source = normalized["stance_source"]
    is_initial = normalized["round"] == 0
    valid_initial = is_initial & (
        source.eq("initial_anchor") | source.eq("external_judge")
    )
    valid_later = (~is_initial) & source.eq("external_judge")
    invalid_sources = sorted(set(source[~(valid_initial | valid_later)]))
    if invalid_sources:
        raise ValueError(
            "Matched analysis requires the designed initial anchor at round 0 "
            "and a direct external judge score at every later round; found: "
            + ", ".join(invalid_sources)
        )

    observed_conditions = set(normalized["condition"].astype(str))
    missing_conditions = sorted(set(conditions) - observed_conditions)
    extra_conditions = sorted(observed_conditions - set(conditions))
    if missing_conditions or extra_conditions:
        raise ValueError(
            "Unexpected condition set. "
            f"Missing={missing_conditions or 'none'}; extra={extra_conditions or 'none'}."
        )

    expected_rounds = set(range(expected_final_round + 1))
    duplicate_mask = normalized.duplicated([*RUN_KEYS, "side", "round"], keep=False)
    if duplicate_mask.any():
        duplicate = normalized.loc[
            duplicate_mask, [*RUN_KEYS, "side", "round"]
        ].head(5)
        raise ValueError(f"Duplicate side/round rows found:\n{duplicate.to_string(index=False)}")

    for keys, group in normalized.groupby(RUN_KEYS, dropna=False, sort=False):
        sides = set(group["side"].astype(str))
        if sides != {"pro", "con"}:
            raise ValueError(f"Run {keys} has sides {sorted(sides)}, expected pro/con.")
        for side, side_group in group.groupby("side", sort=False):
            rounds = set(side_group["round"].tolist())
            if rounds != expected_rounds:
                raise ValueError(
                    f"Run {keys}, side={side} has rounds {sorted(rounds)}; "
                    f"expected 0..{expected_final_round}."
                )

    return normalized


def _crossed_zero(values: pd.Series) -> bool:
    """Return whether a nonzero-start trajectory ever touches/crosses zero."""
    initial = float(values.iloc[0])
    if initial > 0:
        return bool((values <= 0).any())
    if initial < 0:
        return bool((values >= 0).any())
    return False


def _stance_side(value: float) -> str:
    """Map a scalar stance to its exact signed side."""
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "neutral"


def _final_side_switched(values: pd.Series) -> bool:
    """Return whether the final stance is strictly opposite the initial side."""
    initial_side = _stance_side(float(values.iloc[0]))
    final_side = _stance_side(float(values.iloc[-1]))
    return (
        initial_side in {"positive", "negative"}
        and final_side in {"positive", "negative"}
        and initial_side != final_side
    )


def _rate_indicator(value: bool | np.bool_) -> float:
    """Convert a computed boolean event to an explicit numeric rate contribution."""
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"Expected a boolean event, got {type(value).__name__}.")
    return 1.0 if bool(value) else 0.0


def compute_run_metrics(
    trajectories: pd.DataFrame,
    *,
    expected_final_round: int = 15,
) -> pd.DataFrame:
    """Compute preregistered outcomes and direction-aware diagnostics per run."""
    rows: list[dict] = []
    for keys, group in trajectories.groupby(RUN_KEYS, dropna=False, sort=False):
        pivot = (
            group.pivot(index="round", columns="side", values="stance")
            .sort_index()
            .loc[range(expected_final_round + 1), ["pro", "con"]]
        )
        pro = pivot["pro"].astype(float)
        con = pivot["con"].astype(float)
        gap = (pro - con).abs()
        initial_midpoint = float((pro.iloc[0] + con.iloc[0]) / 2.0)
        final_midpoint = float((pro.iloc[-1] + con.iloc[-1]) / 2.0)
        final_pro_side = _stance_side(float(pro.iloc[-1]))
        final_con_side = _stance_side(float(con.iloc[-1]))
        pro_final_side_switched = _final_side_switched(pro)
        con_final_side_switched = _final_side_switched(con)
        pro_any_time_zero_crossed = _crossed_zero(pro)
        con_any_time_zero_crossed = _crossed_zero(con)
        final_same_nonzero_side = bool(
            final_pro_side == final_con_side
            and final_pro_side in {"positive", "negative"}
        )
        final_shared_positive_side = bool(
            final_pro_side == "positive" and final_con_side == "positive"
        )
        final_shared_negative_side = bool(
            final_pro_side == "negative" and final_con_side == "negative"
        )
        duration = float(pivot.index[-1] - pivot.index[0])
        gap_auc = (
            float(np.trapezoid(gap.to_numpy(), x=pivot.index.to_numpy()) / duration)
            if duration > 0
            else float(gap.iloc[-1])
        )

        row = dict(zip(RUN_KEYS, keys))
        row.update(
            {
                "initial_pro": float(pro.iloc[0]),
                "initial_con": float(con.iloc[0]),
                "final_pro": float(pro.iloc[-1]),
                "final_con": float(con.iloc[-1]),
                "initial_midpoint": initial_midpoint,
                "final_midpoint": final_midpoint,
                "center_shift": float(final_midpoint - initial_midpoint),
                "absolute_center_shift": float(
                    abs(final_midpoint - initial_midpoint)
                ),
                "final_pro_side": final_pro_side,
                "final_con_side": final_con_side,
                "pro_final_side_switched": pro_final_side_switched,
                "con_final_side_switched": con_final_side_switched,
                "pro_final_side_switch_rate": _rate_indicator(
                    pro_final_side_switched
                ),
                "con_final_side_switch_rate": _rate_indicator(
                    con_final_side_switched
                ),
                "final_side_switch_rate": float(
                    np.mean(
                        [pro_final_side_switched, con_final_side_switched]
                    )
                ),
                "pro_any_time_zero_crossed": pro_any_time_zero_crossed,
                "con_any_time_zero_crossed": con_any_time_zero_crossed,
                "pro_any_time_zero_cross_rate": _rate_indicator(
                    pro_any_time_zero_crossed
                ),
                "con_any_time_zero_cross_rate": _rate_indicator(
                    con_any_time_zero_crossed
                ),
                "any_time_zero_crossing_rate": float(
                    np.mean(
                        [
                            pro_any_time_zero_crossed,
                            con_any_time_zero_crossed,
                        ]
                    )
                ),
                "final_same_nonzero_side": final_same_nonzero_side,
                "final_same_nonzero_side_rate": _rate_indicator(
                    final_same_nonzero_side
                ),
                "final_shared_positive": _rate_indicator(
                    final_shared_positive_side
                ),
                "final_shared_negative": _rate_indicator(
                    final_shared_negative_side
                ),
                "final_shared_side": (
                    final_pro_side
                    if final_pro_side == final_con_side
                    and final_pro_side in {"positive", "negative"}
                    else "none"
                ),
                "initial_gap": float(gap.iloc[0]),
                "final_gap": float(gap.iloc[-1]),
                "gap_reduction": float(gap.iloc[0] - gap.iloc[-1]),
                "mean_absolute_shift": float(
                    np.mean(
                        [
                            abs(pro.iloc[-1] - pro.iloc[0]),
                            abs(con.iloc[-1] - con.iloc[0]),
                        ]
                    )
                ),
                "crossing_rate": float(
                    np.mean(
                        [
                            pro_any_time_zero_crossed,
                            con_any_time_zero_crossed,
                        ]
                    )
                ),
                "trajectory_gap_auc": gap_auc,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(PAIR_KEYS + ["condition"]).reset_index(drop=True)


def _require_unique_condition_cells(run_metrics: pd.DataFrame) -> None:
    """Reject repeated runs for the same topic/seed/condition cell."""
    _require_columns(run_metrics, [*PAIR_KEYS, "condition", "run_id"])
    duplicate_cells = run_metrics.duplicated(
        [*PAIR_KEYS, "condition"],
        keep=False,
    )
    if duplicate_cells.any():
        duplicate = run_metrics.loc[
            duplicate_cells,
            [*PAIR_KEYS, "condition", "run_id"],
        ].head(10)
        raise ValueError(
            "Each topic/trial/seed/condition cell must contain exactly one run. "
            f"First duplicates:\n{duplicate.to_string(index=False)}"
        )


def require_complete_pairs(
    run_metrics: pd.DataFrame,
    *,
    conditions: Sequence[str] = EXPECTED_CONDITIONS,
) -> None:
    _require_unique_condition_cells(run_metrics)
    counts = (
        run_metrics.groupby(PAIR_KEYS, dropna=False)["condition"]
        .agg(lambda values: tuple(sorted(set(values))))
    )
    expected = tuple(sorted(conditions))
    bad = counts[counts != expected]
    if not bad.empty:
        raise ValueError(
            "Every topic/trial/seed must contain the same condition set. "
            f"First incomplete pairs:\n{bad.head(10).to_string()}"
        )


def select_complete_pairs(
    run_metrics: pd.DataFrame,
    *,
    conditions: Sequence[str] = EXPECTED_CONDITIONS,
) -> tuple[pd.DataFrame, list[dict]]:
    """Explicitly retain only pairs containing every requested condition."""
    _require_unique_condition_cells(run_metrics)
    observed = (
        run_metrics.groupby(PAIR_KEYS, dropna=False)["condition"]
        .agg(lambda values: tuple(sorted(set(values))))
        .rename("observed_conditions")
        .reset_index()
    )
    expected = tuple(sorted(conditions))
    complete = observed[observed["observed_conditions"] == expected]
    incomplete = observed[observed["observed_conditions"] != expected]
    complete_keys = complete[PAIR_KEYS].copy()
    selected = run_metrics.merge(
        complete_keys,
        on=PAIR_KEYS,
        how="inner",
        validate="many_to_one",
    )
    excluded = []
    for row in incomplete.to_dict(orient="records"):
        observed_conditions = list(row.pop("observed_conditions"))
        row["observed_conditions"] = observed_conditions
        row["missing_conditions"] = sorted(set(conditions) - set(observed_conditions))
        excluded.append(row)
    if selected.empty:
        raise ValueError("No complete topic/trial/seed pairs remain.")
    require_complete_pairs(selected, conditions=conditions)
    return selected, excluded


def _equal_topic_mean(frame: pd.DataFrame, value_column: str) -> float:
    return float(frame.groupby("topic", sort=False)[value_column].mean().mean())


def hierarchical_bootstrap(
    frame: pd.DataFrame,
    value_column: str,
    *,
    replicates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Resample topics, then paired seeds within each sampled topic."""
    _require_columns(frame, [*PAIR_KEYS, value_column])
    if replicates < 1:
        raise ValueError("Hierarchical bootstrap replicates must be positive.")
    duplicate_pairs = frame.duplicated(PAIR_KEYS, keep=False)
    if duplicate_pairs.any():
        duplicate = frame.loc[duplicate_pairs, PAIR_KEYS].head(10)
        raise ValueError(
            "Hierarchical bootstrap requires one value per topic/paired seed. "
            f"First duplicates:\n{duplicate.to_string(index=False)}"
        )
    boolean_mask = frame[value_column].map(
        lambda value: isinstance(value, (bool, np.bool_))
    )
    if boolean_mask.any():
        raise ValueError(
            f"Bootstrap metric '{value_column}' contains boolean values; "
            "rate indicators must be explicit 0.0/1.0 numbers."
        )
    try:
        numeric_values = pd.to_numeric(frame[value_column], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Bootstrap metric '{value_column}' must be numeric."
        ) from exc
    if np.iscomplexobj(numeric_values.to_numpy()):
        raise ValueError(
            f"Bootstrap metric '{value_column}' must be real numeric."
        )
    numeric_values = numeric_values.astype(float)
    if not np.isfinite(numeric_values.to_numpy()).all():
        raise ValueError(
            f"Bootstrap metric '{value_column}' contains non-finite values."
        )

    working = frame.copy()
    working[value_column] = numeric_values
    topics = frame["topic"].drop_duplicates().to_numpy()
    if len(topics) < 2:
        raise ValueError("Hierarchical bootstrap requires at least two topics.")
    draws = np.empty(replicates, dtype=float)
    by_topic = {
        topic: working[working["topic"] == topic]
        for topic in topics
    }

    for index in range(replicates):
        sampled_topics = rng.choice(topics, size=len(topics), replace=True)
        topic_means = []
        for topic in sampled_topics:
            values = by_topic[topic][value_column].to_numpy(dtype=float)
            sampled_values = rng.choice(values, size=len(values), replace=True)
            topic_means.append(float(np.mean(sampled_values)))
        draws[index] = float(np.mean(topic_means))
    return draws


def _interval(draws: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    alpha = 1.0 - confidence
    lower, upper = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lower), float(upper)


def _summarize_condition_metrics(
    run_metrics: pd.DataFrame,
    metrics: Sequence[str],
    *,
    replicates: int,
    random_seed: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    seed_sequence = np.random.SeedSequence(random_seed)
    child_seeds = iter(
        seed_sequence.spawn(len(EXPECTED_CONDITIONS) * len(metrics))
    )
    for condition in EXPECTED_CONDITIONS:
        subset = run_metrics[run_metrics["condition"] == condition]
        for metric in metrics:
            rng = np.random.default_rng(next(child_seeds))
            draws = hierarchical_bootstrap(
                subset, metric, replicates=replicates, rng=rng
            )
            lower, upper = _interval(draws)
            rows.append(
                {
                    "condition": condition,
                    "metric": metric,
                    "estimate": _equal_topic_mean(subset, metric),
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "topics": int(subset["topic"].nunique()),
                    "paired_seeds": int(len(subset)),
                }
            )
    return pd.DataFrame(rows)


def summarize_conditions(
    run_metrics: pd.DataFrame,
    *,
    replicates: int,
    random_seed: int,
) -> pd.DataFrame:
    """Aggregate the unchanged preregistered condition metrics."""
    return _summarize_condition_metrics(
        run_metrics,
        METRICS,
        replicates=replicates,
        random_seed=random_seed,
    )


def summarize_directional_conditions(
    run_metrics: pd.DataFrame,
    *,
    replicates: int,
    random_seed: int,
) -> pd.DataFrame:
    """Aggregate secondary directional diagnostics with the same hierarchy."""
    return _summarize_condition_metrics(
        run_metrics,
        DIRECTIONAL_METRICS,
        replicates=replicates,
        random_seed=random_seed,
    )


def paired_differences(
    run_metrics: pd.DataFrame,
    *,
    metric: str = "final_gap",
) -> pd.DataFrame:
    pivot = run_metrics.pivot(index=PAIR_KEYS, columns="condition", values=metric)
    rows = []
    for comparator in COMPARATORS:
        delta_name = f"{PRIMARY_CONDITION}_minus_{comparator}"
        difference = pivot[PRIMARY_CONDITION] - pivot[comparator]
        for keys, value in difference.items():
            row = dict(zip(PAIR_KEYS, keys))
            row.update(
                {
                    "comparison": delta_name,
                    "metric": metric,
                    "difference": float(value),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_pairwise(
    differences: pd.DataFrame,
    *,
    replicates: int,
    random_seed: int,
) -> pd.DataFrame:
    rows = []
    seed_sequence = np.random.SeedSequence(random_seed)
    child_seeds = iter(seed_sequence.spawn(differences["comparison"].nunique()))
    for comparison, subset in differences.groupby("comparison", sort=False):
        rng = np.random.default_rng(next(child_seeds))
        draws = hierarchical_bootstrap(
            subset, "difference", replicates=replicates, rng=rng
        )
        lower, upper = _interval(draws)
        rows.append(
            {
                "comparison": comparison,
                "metric": str(subset["metric"].iloc[0]),
                "estimate": _equal_topic_mean(subset, "difference"),
                "ci_lower": lower,
                "ci_upper": upper,
                "direction": "negative favors BE",
                "topics": int(subset["topic"].nunique()),
                "paired_seeds": int(len(subset)),
            }
        )
    return pd.DataFrame(rows)


def summarize_by_topic(
    run_metrics: pd.DataFrame,
    differences: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    condition_summary = (
        run_metrics.groupby(["topic", "condition"], as_index=False)[list(METRICS)]
        .mean()
        .sort_values(["topic", "condition"])
        .reset_index(drop=True)
    )
    pairwise_summary = (
        differences.groupby(["topic", "comparison"], as_index=False)
        .agg(
            estimate=("difference", "mean"),
            paired_seeds=("difference", "size"),
        )
        .sort_values(["topic", "comparison"])
        .reset_index(drop=True)
    )
    pairwise_summary["direction"] = "negative favors BE"
    return condition_summary, pairwise_summary


def summarize_directional_by_topic(run_metrics: pd.DataFrame) -> pd.DataFrame:
    """Return paired-seed means for every directional diagnostic by topic."""
    grouped = run_metrics.groupby(["topic", "condition"], as_index=False)
    summary = (
        grouped[list(DIRECTIONAL_METRICS)]
        .mean()
        .sort_values(["topic", "condition"])
        .reset_index(drop=True)
    )
    counts = (
        run_metrics.groupby(["topic", "condition"], as_index=False)
        .size()
        .rename(columns={"size": "paired_seeds"})
    )
    summary = summary.merge(
        counts,
        on=["topic", "condition"],
        how="left",
        validate="one_to_one",
    )
    return summary[
        ["topic", "condition", "paired_seeds", *DIRECTIONAL_METRICS]
    ]


def _directional_ci_table(
    summary: pd.DataFrame,
    metrics: Sequence[str],
) -> pd.DataFrame:
    """Render estimates and CIs compactly without discarding signed values."""
    rows = []
    for condition in EXPECTED_CONDITIONS:
        subset = summary[summary["condition"] == condition].set_index("metric")
        row = {"condition": condition}
        for metric in metrics:
            values = subset.loc[metric]
            row[DIRECTIONAL_METRIC_LABELS[metric]] = (
                f"{float(values['estimate']):.4f} "
                f"[{float(values['ci_lower']):.4f}, "
                f"{float(values['ci_upper']):.4f}]"
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _directional_topic_table(
    summary: pd.DataFrame,
    metrics: Sequence[str],
) -> pd.DataFrame:
    columns = ["topic", "condition", "paired_seeds", *metrics]
    return summary[columns].rename(columns=DIRECTIONAL_METRIC_LABELS)


def render_markdown(
    condition_summary: pd.DataFrame,
    pairwise_summary: pd.DataFrame,
    topic_condition_summary: pd.DataFrame,
    topic_pairwise_summary: pd.DataFrame,
    directional_condition_summary: pd.DataFrame,
    directional_topic_summary: pd.DataFrame,
    *,
    analysis_label: str,
    pair_policy: str,
    excluded_pairs: Sequence[dict],
) -> str:
    final_gap = condition_summary[condition_summary["metric"] == "final_gap"].copy()
    columns = ["condition", "estimate", "ci_lower", "ci_upper", "paired_seeds"]
    topic_gaps = topic_condition_summary.pivot(
        index="topic",
        columns="condition",
        values="final_gap",
    ).reset_index()
    lines = [
        f"# {analysis_label}",
        "",
        "All stance values are from the same external evaluator. Estimates and",
        "intervals use the preregistered topic-then-paired-seed hierarchical",
        "bootstrap. The topic × paired seed is the experimental unit.",
        f"Pair policy: `{pair_policy}`. Excluded incomplete pairs: "
        f"`{len(excluded_pairs)}`.",
        "",
        "## Final externally judged Pro/Con gap",
        "",
        final_gap[columns].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Paired BE differences",
        "",
        pairwise_summary[
            [
                "comparison",
                "estimate",
                "ci_lower",
                "ci_upper",
                "direction",
                "paired_seeds",
            ]
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Final gap by topic",
        "",
        topic_gaps.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Paired BE difference by topic",
        "",
        topic_pairwise_summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Direction-aware diagnostics (secondary)",
        "",
        "These diagnostics expose joint directional movement that an absolute",
        "Pro/Con gap cannot distinguish. Their condition-level 95% intervals use",
        "the same topic-then-paired-seed hierarchy. They are descriptive secondary",
        "analyses; no multiplicity-adjusted or preregistered directional claim is",
        "implied.",
        "",
        "### Definitions",
        "",
        pd.DataFrame(
            [
                {
                    "metric": f"`{metric}`",
                    "definition": DIRECTIONAL_DIAGNOSTIC_DEFINITIONS[metric],
                }
                for metric in DIRECTIONAL_METRICS
            ]
        ).to_markdown(index=False),
        "",
    ]
    for group_name, metrics in DIRECTIONAL_METRIC_GROUPS.items():
        lines.extend(
            [
                f"### {group_name}: hierarchical estimates",
                "",
                "Each cell is estimate [95% CI].",
                "",
                _directional_ci_table(
                    directional_condition_summary,
                    metrics,
                ).to_markdown(index=False),
                "",
            ]
        )

    lines.extend(
        [
            "## Direction-aware diagnostics by topic",
            "",
            "Each estimate is the mean across the paired seeds retained for that",
            "topic and condition; these topic rows do not carry separate CIs.",
            "",
        ]
    )
    for group_name, metrics in DIRECTIONAL_METRIC_GROUPS.items():
        lines.extend(
            [
                f"### {group_name}: by topic",
                "",
                _directional_topic_table(
                    directional_topic_summary,
                    metrics,
                ).to_markdown(index=False, floatfmt=".4f"),
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def write_analysis_bundle(output_dir: Path, artifacts: dict[str, str]) -> None:
    """Publish a complete report by atomically renaming an isolated staging dir."""
    if os.path.lexists(output_dir):
        raise FileExistsError(
            f"Refusing to overwrite existing analysis output: {output_dir}"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.tmp-",
            dir=output_dir.parent,
        )
    )
    try:
        for name in sorted(artifacts):
            if Path(name).name != name:
                raise ValueError(f"Analysis artifact name must be portable: {name}")
            (staging_dir / name).write_text(artifacts[name], encoding="utf-8")
        os.replace(staging_dir, output_dir)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectory_csv", type=Path)
    parser.add_argument(
        "--primary-inventory",
        type=Path,
        required=True,
        help=(
            "Complete primary-campaign inventory emitted by "
            "verify_rebuttal_artifacts.py."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New destination directory; existing paths are never overwritten.",
    )
    parser.add_argument("--expected-final-round", type=int, default=15)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--random-seed", type=int, default=20_260_723)
    parser.add_argument(
        "--pair-policy",
        choices=("strict", "complete-only"),
        default="strict",
        help=(
            "strict aborts on any incomplete topic-seed cell; complete-only "
            "explicitly excludes and records incomplete historical cells."
        ),
    )
    parser.add_argument(
        "--analysis-label",
        default="Matched baseline analysis",
        help="Heading used in the generated Markdown report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates < 1:
        raise ValueError("--bootstrap-replicates must be positive.")

    primary_inventory_text, primary_inventory, primary_inventory_summary = (
        load_primary_input_inventory(args.primary_inventory)
    )
    analyzer = analyzer_provenance(Path(__file__))
    trajectories, input_artifact = load_trajectory_input(args.trajectory_csv)
    trajectories = validate_trajectories(
        trajectories, expected_final_round=args.expected_final_round
    )
    run_metrics = compute_run_metrics(
        trajectories, expected_final_round=args.expected_final_round
    )
    excluded_pairs: list[dict] = []
    if args.pair_policy == "strict":
        require_complete_pairs(run_metrics)
    else:
        run_metrics, excluded_pairs = select_complete_pairs(run_metrics)
    condition_summary = summarize_conditions(
        run_metrics,
        replicates=args.bootstrap_replicates,
        random_seed=args.random_seed,
    )
    differences = paired_differences(run_metrics)
    pairwise_summary = summarize_pairwise(
        differences,
        replicates=args.bootstrap_replicates,
        random_seed=args.random_seed + 1,
    )
    topic_condition_summary, topic_pairwise_summary = summarize_by_topic(
        run_metrics,
        differences,
    )
    directional_condition_summary = summarize_directional_conditions(
        run_metrics,
        replicates=args.bootstrap_replicates,
        random_seed=args.random_seed + 2,
    )
    directional_topic_summary = summarize_directional_by_topic(run_metrics)

    markdown = render_markdown(
        condition_summary,
        pairwise_summary,
        topic_condition_summary,
        topic_pairwise_summary,
        directional_condition_summary,
        directional_topic_summary,
        analysis_label=args.analysis_label,
        pair_policy=args.pair_policy,
        excluded_pairs=excluded_pairs,
    )

    payload = {
        "input_artifact": input_artifact,
        "analysis_label": args.analysis_label,
        "expected_final_round": args.expected_final_round,
        "bootstrap_replicates": args.bootstrap_replicates,
        "random_seed": args.random_seed,
        "conditions": list(EXPECTED_CONDITIONS),
        "primary_condition": PRIMARY_CONDITION,
        "comparators": list(COMPARATORS),
        "primary_metric": "final_gap",
        "experimental_unit": "topic_by_paired_seed",
        "pair_policy": args.pair_policy,
        "directional_diagnostic_definitions": DIRECTIONAL_DIAGNOSTIC_DEFINITIONS,
        "excluded_incomplete_pairs": excluded_pairs,
        "condition_summary": condition_summary.to_dict(orient="records"),
        "pairwise_summary": pairwise_summary.to_dict(orient="records"),
        "topic_condition_summary": topic_condition_summary.to_dict(
            orient="records"
        ),
        "topic_pairwise_summary": topic_pairwise_summary.to_dict(
            orient="records"
        ),
        "directional_condition_summary": directional_condition_summary.to_dict(
            orient="records"
        ),
        "directional_topic_summary": directional_topic_summary.to_dict(
            orient="records"
        ),
    }
    analysis_json = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    output_tables = {
        "condition_summary.csv": condition_summary,
        "directional_condition_summary.csv": directional_condition_summary,
        "directional_topic_summary.csv": directional_topic_summary,
        "paired_primary_differences.csv": differences,
        "pairwise_summary.csv": pairwise_summary,
        "run_metrics.csv": run_metrics,
        "topic_condition_summary.csv": topic_condition_summary,
        "topic_pairwise_summary.csv": topic_pairwise_summary,
    }
    artifacts = {
        name: frame.to_csv(index=False, lineterminator="\n")
        for name, frame in output_tables.items()
    }
    artifacts.update(
        {
            "analysis.json": analysis_json,
            "analysis.md": markdown,
            PRIMARY_INPUT_INVENTORY_NAME: primary_inventory_text,
        }
    )
    artifacts[BUNDLE_MANIFEST_NAME] = build_bundle_manifest(
        args=args,
        analyzer=analyzer,
        trajectory_input=input_artifact,
        trajectory_rows=len(trajectories),
        analyzed_runs=int(run_metrics["run_id"].nunique()),
        analyzed_topics=int(run_metrics["topic"].nunique()),
        primary_inventory=primary_inventory,
        primary_inventory_summary=primary_inventory_summary,
        artifacts=artifacts,
    )
    write_analysis_bundle(args.output_dir, artifacts)


if __name__ == "__main__":
    main()
