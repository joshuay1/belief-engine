#!/usr/bin/env python3
"""Post-hoc multiplicity audit of the matched-baseline primary family.

This program reconstructs the two registered absolute-final-gap contrasts
directly from the frozen external-stance trajectory CSV.  It deliberately
excludes every secondary outcome.  The two contrasts share every bootstrap
and exact sign-flip resample so their dependence is retained.

The program is fail closed: the frozen input hashes, campaign completion
inventory, design, evaluator provenance, pairing, trajectory completeness,
and summary/trajectory agreement must all pass before any result is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = ROOT / "MATCHED_BASELINE_PRIMARY_MULTIPLICITY_POSTHOC_SPEC.yaml"
DEFAULT_OUTPUT = (
    ROOT
    / "results"
    / "analyses"
    / "matched_baseline_primary_multiplicity_posthoc_v1"
)

BE = "be_gpt54mini"
COMPARATORS = (
    "prompt_self_update_gpt54mini",
    "rag_self_update_gpt54mini",
)
CONDITIONS = (BE, *COMPARATORS)
COMPARISON_NAMES = tuple(f"{BE}_minus_{name}" for name in COMPARATORS)
AGENT_SIDE = {"Pro": "pro", "Con": "con"}
EXPECTED_ROUNDS = tuple(range(16))
FLOAT_TOLERANCE = 1e-12
TAIL_TOLERANCE = 1e-12
ANALYSIS_OUTPUT_NAMES = (
    "analysis.json",
    "bootstrap_intervals.csv",
    "exact_topic_sign_flip_null.csv",
    "exact_topic_sign_flip_tests.csv",
    "input_audit.json",
    "leave_one_topic_out.csv",
    "leave_one_topic_out_ranges.csv",
    "paired_contrasts.csv",
    "report.md",
    "topic_contrasts.csv",
)


class AnalysisError(RuntimeError):
    """Raised when a locked assumption or input check fails."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AnalysisError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, *, relative_path: str) -> dict[str, Any]:
    return {
        "path": relative_path,
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    write_text(path, canonical_json(payload))


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    require(not frame.empty, f"Refusing to write empty table: {path.name}")
    frame.to_csv(path, index=False, float_format="%.17g", lineterminator="\n")


def resolve_locked_file(
    spec: Mapping[str, Any],
    name: str,
) -> tuple[Path, dict[str, Any]]:
    record = spec["inputs"][name]
    path = ROOT / str(record["path"])
    require(path.is_file(), f"Locked input is missing: {record['path']}")
    observed_hash = sha256(path)
    observed_bytes = path.stat().st_size
    require(
        observed_hash == record["sha256"],
        f"Hash mismatch for locked input {record['path']}: {observed_hash}",
    )
    require(
        observed_bytes == int(record["bytes"]),
        f"Byte-size mismatch for locked input {record['path']}: {observed_bytes}",
    )
    return path, {
        "path": str(record["path"]),
        "sha256": observed_hash,
        "bytes": observed_bytes,
    }


def load_spec(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"Specification is missing: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), "Specification must be a YAML mapping.")
    require(
        payload.get("analysis_id")
        == "neurips_2026_matched_baseline_primary_multiplicity_posthoc_v1",
        "Unexpected analysis_id.",
    )
    require(
        payload.get("classification")
        == "transparent_posthoc_multiplicity_audit_of_registered_primary_family",
        "Unexpected analysis classification.",
    )
    require(
        payload.get("status") == "locked_for_execution",
        "Specification is not locked_for_execution.",
    )

    primary = payload.get("primary_family", {})
    require(
        primary.get("metric") == "absolute_final_pro_con_gap",
        "Primary metric lock changed.",
    )
    require(
        tuple(primary.get("contrasts", ())) == COMPARISON_NAMES,
        "Primary family must contain exactly the two locked BE contrasts.",
    )
    require(
        primary.get("family_size") == 2,
        "Primary family size must be two.",
    )
    require(
        primary.get("excluded_outcomes")
        == ["all_secondary_and_diagnostic_outcomes_including_trajectory_gap_auc"],
        "Secondary-outcome exclusion lock changed.",
    )

    bootstrap = payload.get("bootstrap", {})
    require(
        bootstrap.get("method")
        == "shared_hierarchical_topic_then_paired_seed_bootstrap",
        "Bootstrap method lock changed.",
    )
    require(
        bootstrap.get("replicates") == 100000,
        "Exactly 100,000 bootstrap draws are required.",
    )
    require(
        bootstrap.get("shared_resamples_across_contrasts") is True,
        "Bootstrap resamples must be shared across contrasts.",
    )
    require(
        bootstrap.get("chunk_size") == 2000,
        "Bootstrap chunk-size lock changed.",
    )
    require(
        bootstrap.get("simultaneous_method")
        == "max_absolute_studentized_bootstrap_deviation",
        "Simultaneous-interval method lock changed.",
    )
    require(
        bootstrap.get("bonferroni_percentile_quantiles") == [0.0125, 0.9875],
        "Bonferroni percentile quantiles changed.",
    )

    sign_flip = payload.get("exact_topic_sign_flip", {})
    require(
        sign_flip.get("method")
        == "exhaustive_shared_topic_vector_sign_flip",
        "Exact sign-flip method lock changed.",
    )
    require(sign_flip.get("topics") == 5, "Sign-flip topic count changed.")
    require(
        sign_flip.get("assignments") == 32,
        "Exactly 32 topic-level assignments are required.",
    )
    require(
        sign_flip.get("shared_signs_across_contrasts") is True,
        "Topic signs must be shared across contrasts.",
    )

    expected_script = payload["scripts"]["analyzer"]
    own_path = Path(__file__).resolve()
    require(
        str(expected_script["path"])
        == "scripts/analyze_matched_baseline_primary_multiplicity.py",
        "Analyzer path lock changed.",
    )
    require(
        sha256(own_path) == expected_script["sha256"],
        "Analyzer script does not match the hash frozen in the specification.",
    )
    return payload


def validate_frozen_provenance(
    spec: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> dict[str, Any]:
    preregistration = yaml.safe_load(
        paths["preregistration"].read_text(encoding="utf-8")
    )
    require(
        preregistration["experiment_id"] == "neurips_2026_matched_baseline_v1",
        "Unexpected parent preregistration.",
    )
    require(
        preregistration["outcomes"]["primary"]["name"]
        == "absolute_final_pro_con_gap",
        "Parent registered primary outcome changed.",
    )
    require(
        preregistration["conditions"] == list(CONDITIONS),
        "Parent registered condition order changed.",
    )
    require(
        preregistration["analysis"]["experimental_unit"]
        == "topic_by_paired_seed",
        "Parent experimental-unit lock changed.",
    )
    require(
        preregistration["design"]["stance_source_for_all_conditions"]
        == "external",
        "Parent external-evaluator lock changed.",
    )

    inventory = json.loads(
        paths["verified_primary_inventory"].read_text(encoding="utf-8")
    )
    require(inventory.get("status") == "complete", "Inventory is not complete.")
    require(inventory.get("mode") == "complete", "Inventory mode is not complete.")
    require(inventory.get("profile") == "primary", "Inventory is not primary.")
    completion = inventory.get("completion", {})
    require(
        completion.get("completion_claimed") is True,
        "Inventory does not claim completion.",
    )
    require(
        completion.get("design_inventory_complete") is True,
        "Inventory design is incomplete.",
    )
    require(
        completion.get("completed_runs")
        == completion.get("expected_runs")
        == 150,
        "Inventory does not certify 150/150 runs.",
    )
    require(
        completion.get("remaining_runs") == 0,
        "Inventory reports remaining runs.",
    )
    require(inventory.get("warnings") == [], "Inventory contains warnings.")
    require(
        inventory["runs"].get("incomplete_directories") == [],
        "Inventory contains incomplete run directories.",
    )

    campaign = json.loads(paths["campaign_manifest"].read_text(encoding="utf-8"))
    expected = spec["expected_design"]
    require(
        campaign.get("experiment") == "two_agent_model_comparison",
        "Unexpected campaign experiment.",
    )
    require(
        campaign["git"].get("commit") == expected["execution_commit"],
        "Campaign execution commit changed.",
    )
    require(
        campaign["git"].get("dirty") is False,
        "Campaign manifest records a dirty execution worktree.",
    )
    require(
        campaign.get("selected_be_stance_source") == "external",
        "Campaign selected stance source is not external.",
    )
    require(
        campaign.get("topic_grid_stance_source")
        == "external_judge_for_all_conditions",
        "Campaign does not use the common external evaluator for all conditions.",
    )
    require(
        campaign["initial_stances"] == {"pro": 0.75, "con": -0.75},
        "Campaign initial stances changed.",
    )
    require(
        [entry["name"] for entry in campaign["conditions"]] == list(CONDITIONS),
        "Campaign condition registry changed.",
    )
    require(
        len(campaign.get("runs", [])) == 150,
        "Campaign manifest does not contain 150 runs.",
    )

    evaluator = campaign["evaluator_provenance"]
    for key in (
        "backend",
        "model",
        "reasoning_effort",
        "verbosity",
        "temperature",
        "prompt_sha256",
        "evaluator_code_sha256",
        "runner_code_sha256",
        "runner_commit",
    ):
        require(
            evaluator.get(key) == expected["evaluator"][key],
            f"Evaluator provenance mismatch for {key}.",
        )
    require(
        inventory["provenance"]["execution_commit"]
        == expected["execution_commit"],
        "Inventory execution commit changed.",
    )
    require(
        inventory["artifacts"]["tree_sha256"]
        == expected["campaign_tree_sha256"],
        "Verified campaign tree hash changed.",
    )
    require(
        inventory["registry"]["sha256"] == expected["registry_sha256"],
        "Verified campaign registry hash changed.",
    )
    require(
        inventory["provenance"]["evaluator_fingerprint_sha256"]
        == expected["evaluator_fingerprint_sha256"],
        "Verified evaluator fingerprint changed.",
    )

    parent_bundle = json.loads(
        paths["verified_trajectory_bundle_manifest"].read_text(encoding="utf-8")
    )
    require(
        parent_bundle.get("bundle_type") == "matched_baseline_direction_analysis",
        "Unexpected verified trajectory bundle type.",
    )
    require(
        parent_bundle["inputs"]["trajectory_csv"]["sha256"]
        == spec["inputs"]["trajectory_csv"]["sha256"],
        "Parent bundle trajectory identity changed.",
    )
    require(
        parent_bundle["inputs"]["trajectory_csv"]["rows"] == 4800,
        "Parent bundle does not certify 4,800 trajectory rows.",
    )
    require(
        parent_bundle["inputs"]["verified_primary_inventory"]["sha256"]
        == spec["inputs"]["verified_primary_inventory"]["sha256"],
        "Parent bundle inventory identity changed.",
    )

    return {
        "parent_preregistration": {
            "primary_outcome": preregistration["outcomes"]["primary"]["name"],
            "experimental_unit": preregistration["analysis"]["experimental_unit"],
            "paired_method_differences": preregistration["analysis"][
                "paired_method_differences"
            ],
        },
        "completion_inventory": {
            "status": inventory["status"],
            "completed_runs": completion["completed_runs"],
            "expected_runs": completion["expected_runs"],
            "remaining_runs": completion["remaining_runs"],
            "campaign_tree_sha256": inventory["artifacts"]["tree_sha256"],
            "registry_sha256": inventory["registry"]["sha256"],
            "execution_commit": inventory["provenance"]["execution_commit"],
            "evaluator_fingerprint_sha256": inventory["provenance"][
                "evaluator_fingerprint_sha256"
            ],
        },
        "campaign_manifest": {
            "runs": len(campaign["runs"]),
            "conditions": [entry["name"] for entry in campaign["conditions"]],
            "execution_commit": campaign["git"]["commit"],
            "git_dirty": campaign["git"]["dirty"],
            "selected_stance_source": campaign["selected_be_stance_source"],
            "evaluator": evaluator,
        },
        "verified_parent_bundle": {
            "bundle_type": parent_bundle["bundle_type"],
            "trajectory_rows": parent_bundle["inputs"]["trajectory_csv"]["rows"],
            "trajectory_sha256": parent_bundle["inputs"]["trajectory_csv"][
                "sha256"
            ],
            "inventory_sha256": parent_bundle["inputs"][
                "verified_primary_inventory"
            ]["sha256"],
        },
    }


def finite_numeric(series: pd.Series, name: str) -> np.ndarray:
    require(
        not series.map(lambda value: isinstance(value, (bool, np.bool_))).any(),
        f"{name} contains boolean values.",
    )
    numeric = pd.to_numeric(series, errors="raise").to_numpy(dtype=float)
    require(np.isfinite(numeric).all(), f"{name} contains non-finite values.")
    return numeric


def integer_numeric(series: pd.Series, name: str) -> np.ndarray:
    numeric = finite_numeric(series, name)
    require(np.equal(numeric, np.floor(numeric)).all(), f"{name} is not integral.")
    return numeric.astype(np.int64)


def validate_and_reconstruct(
    spec: Mapping[str, Any],
    trajectory_path: Path,
    summary_path: Path,
    campaign_manifest_path: Path,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, dict[str, Any]]:
    trajectory = pd.read_csv(trajectory_path)
    required_columns = {
        "run_id",
        "topic",
        "topic_slug",
        "condition",
        "condition_label",
        "model",
        "backend_type",
        "use_belief_engine",
        "agent_type",
        "use_memory",
        "trial",
        "seed",
        "agent",
        "side",
        "round",
        "stance",
        "stance_source",
    }
    require(
        set(trajectory.columns) == required_columns,
        "Trajectory schema differs from the frozen 17-column schema.",
    )
    require(len(trajectory) == 4800, "Expected exactly 4,800 trajectory rows.")
    require(not trajectory.isna().any().any(), "Trajectory contains missing values.")

    trajectory["trial"] = integer_numeric(trajectory["trial"], "trial")
    trajectory["seed"] = integer_numeric(trajectory["seed"], "seed")
    trajectory["round"] = integer_numeric(trajectory["round"], "round")
    trajectory["stance"] = finite_numeric(trajectory["stance"], "stance")
    require(
        ((trajectory["stance"] >= -1.0) & (trajectory["stance"] <= 1.0)).all(),
        "Trajectory stance falls outside [-1, 1].",
    )
    require(
        set(trajectory["condition"]) == set(CONDITIONS),
        "Trajectory condition set changed.",
    )
    require(
        set(trajectory["agent"]) == set(AGENT_SIDE),
        "Trajectory agent labels changed.",
    )
    require(
        all(
            (trajectory.loc[trajectory["agent"] == agent, "side"] == side).all()
            for agent, side in AGENT_SIDE.items()
        ),
        "Agent and side labels disagree.",
    )
    require(
        not trajectory.duplicated(["run_id", "agent", "round"]).any(),
        "Duplicate run-agent-round trajectory rows found.",
    )

    topic_rows = spec["expected_design"]["topics"]
    topic_order = [row["slug"] for row in topic_rows]
    topic_map = {row["slug"]: row["text"] for row in topic_rows}
    require(
        set(trajectory["topic_slug"]) == set(topic_order),
        "Trajectory topic slugs changed.",
    )
    require(
        all(
            (
                trajectory.loc[
                    trajectory["topic_slug"] == slug, "topic"
                ]
                == text
            ).all()
            for slug, text in topic_map.items()
        ),
        "Topic slug/text mapping changed.",
    )

    run_group = trajectory.groupby("run_id", sort=False)
    require(run_group.ngroups == 150, "Trajectory does not contain 150 runs.")
    run_sizes = run_group.size()
    require((run_sizes == 32).all(), "Every run must contain 32 stance rows.")
    for run_id, group in run_group:
        require(
            tuple(sorted(group["round"].unique())) == EXPECTED_ROUNDS,
            f"Run {run_id} does not contain rounds 0 through 15.",
        )
        require(
            group.groupby("agent")["round"].nunique().to_dict()
            == {"Con": 16, "Pro": 16},
            f"Run {run_id} does not contain 16 states per agent.",
        )
        round_zero = group[group["round"] == 0].set_index("agent")
        require(
            math.isclose(
                float(round_zero.loc["Pro", "stance"]),
                0.75,
                abs_tol=FLOAT_TOLERANCE,
            )
            and math.isclose(
                float(round_zero.loc["Con", "stance"]),
                -0.75,
                abs_tol=FLOAT_TOLERANCE,
            ),
            f"Run {run_id} has the wrong initial anchors.",
        )
        require(
            set(round_zero["stance_source"]) == {"initial_anchor"},
            f"Run {run_id} has the wrong round-zero stance source.",
        )
        require(
            set(group.loc[group["round"] > 0, "stance_source"])
            == {"external_judge"},
            f"Run {run_id} has non-external post-start stance scores.",
        )
        invariant_columns = [
            "topic",
            "topic_slug",
            "condition",
            "trial",
            "seed",
        ]
        require(
            all(group[column].nunique() == 1 for column in invariant_columns),
            f"Run {run_id} changes run-level metadata.",
        )

    run_registry = (
        trajectory[
            ["run_id", "topic", "topic_slug", "condition", "trial", "seed"]
        ]
        .drop_duplicates()
        .sort_values("run_id")
        .reset_index(drop=True)
    )
    require(len(run_registry) == 150, "Run registry contains duplicate identities.")
    design_counts = (
        run_registry.groupby(["topic_slug", "condition"]).size().unstack()
    )
    require(
        design_counts.shape == (5, 3)
        and (design_counts.loc[topic_order, list(CONDITIONS)] == 10)
        .to_numpy()
        .all(),
        "Topic-condition design is not complete at 10 runs per cell.",
    )
    require(
        set(run_registry["trial"]) == set(range(1, 11)),
        "Trial registry is not 1 through 10.",
    )
    pair_counts = run_registry.groupby(["topic_slug", "trial", "seed"])[
        "condition"
    ].nunique()
    require(
        len(pair_counts) == 50 and (pair_counts == 3).all(),
        "The three conditions do not share all 50 topic-seed blocks.",
    )
    topic_trial_counts = run_registry.groupby(["topic_slug", "trial"])[
        "seed"
    ].nunique()
    require(
        len(topic_trial_counts) == 50 and (topic_trial_counts == 1).all(),
        "A topic/trial maps to multiple paired seeds.",
    )

    campaign = json.loads(campaign_manifest_path.read_text(encoding="utf-8"))
    manifest_registry = pd.DataFrame(
        [
            {
                "run_id": row["run_id"],
                "topic": row["topic"],
                "topic_slug": row["topic_slug"],
                "condition": row["condition_name"],
                "trial": int(row["trial"]),
                "seed": int(row["seed"]),
            }
            for row in campaign["runs"]
        ]
    ).sort_values("run_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(
        run_registry,
        manifest_registry,
        check_dtype=False,
        check_like=False,
    )

    final_rows = trajectory[trajectory["round"] == 15].copy()
    require(len(final_rows) == 300, "Expected 300 final agent scores.")
    final_wide = final_rows.pivot(
        index=[
            "run_id",
            "topic",
            "topic_slug",
            "condition",
            "trial",
            "seed",
        ],
        columns="agent",
        values="stance",
    ).reset_index()
    require(
        len(final_wide) == 150 and {"Pro", "Con"} <= set(final_wide.columns),
        "Could not reconstruct one Pro and one Con final score per run.",
    )
    final_wide["final_gap"] = np.abs(
        final_wide["Pro"].to_numpy(float)
        - final_wide["Con"].to_numpy(float)
    )

    summary = pd.read_csv(summary_path)
    require(len(summary) == 150, "Summary does not contain 150 rows.")
    require(
        not summary.duplicated("run_id").any(),
        "Summary contains duplicate run IDs.",
    )
    require(
        set(summary["condition"]) == set(CONDITIONS),
        "Summary condition set changed.",
    )
    require(
        set(summary["stance_source"]) == {"external_judge"},
        "Summary includes a non-external stance source.",
    )
    merged = final_wide.merge(
        summary[
            [
                "run_id",
                "topic",
                "topic_slug",
                "condition",
                "trial",
                "seed",
                "pro_final",
                "con_final",
                "final_distance",
            ]
        ],
        on=["run_id", "topic", "topic_slug", "condition", "trial", "seed"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    require(
        set(merged["_merge"]) == {"both"},
        "Summary and trajectory run registries differ.",
    )
    for observed, reported, label in (
        (merged["Pro"], merged["pro_final"], "Pro final stance"),
        (merged["Con"], merged["con_final"], "Con final stance"),
        (merged["final_gap"], merged["final_distance"], "final gap"),
    ):
        error = np.max(
            np.abs(
                pd.to_numeric(observed, errors="raise").to_numpy(float)
                - pd.to_numeric(reported, errors="raise").to_numpy(float)
            )
        )
        require(
            error <= FLOAT_TOLERANCE,
            f"Summary/trajectory {label} mismatch: {error}.",
        )

    gap_wide = final_wide.pivot(
        index=["topic", "topic_slug", "trial", "seed"],
        columns="condition",
        values="final_gap",
    ).reset_index()
    require(
        len(gap_wide) == 50
        and not gap_wide[list(CONDITIONS)].isna().any().any(),
        "Could not form all 50 paired final-gap blocks.",
    )
    gap_wide["topic_order"] = gap_wide["topic_slug"].map(
        {slug: index for index, slug in enumerate(topic_order)}
    )
    gap_wide = gap_wide.sort_values(["topic_order", "trial"]).reset_index(
        drop=True
    )

    contrast_rows: list[dict[str, Any]] = []
    for row in gap_wide.itertuples(index=False):
        for comparator, comparison in zip(COMPARATORS, COMPARISON_NAMES):
            be_gap = float(getattr(row, BE))
            comparator_gap = float(getattr(row, comparator))
            contrast_rows.append(
                {
                    "topic_order": int(row.topic_order),
                    "topic": row.topic,
                    "topic_slug": row.topic_slug,
                    "trial": int(row.trial),
                    "seed": int(row.seed),
                    "comparison": comparison,
                    "be_final_gap": be_gap,
                    "comparator_final_gap": comparator_gap,
                    "difference": be_gap - comparator_gap,
                }
            )
    paired = pd.DataFrame(contrast_rows)
    require(len(paired) == 100, "Expected exactly 100 paired contrast rows.")

    cube = np.empty((5, 10, 2), dtype=float)
    for topic_index, slug in enumerate(topic_order):
        block = paired[paired["topic_slug"] == slug]
        for contrast_index, comparison in enumerate(COMPARISON_NAMES):
            values = (
                block[block["comparison"] == comparison]
                .sort_values("trial")["difference"]
                .to_numpy(float)
            )
            require(
                len(values) == 10,
                f"Topic {slug} comparison {comparison} is incomplete.",
            )
            cube[topic_index, :, contrast_index] = values

    design_audit = {
        "trajectory_rows": len(trajectory),
        "debates": run_group.ngroups,
        "agent_trajectories": 300,
        "external_post_start_scores": int((trajectory["round"] > 0).sum()),
        "topics": 5,
        "conditions": list(CONDITIONS),
        "paired_topic_seed_blocks": len(gap_wide),
        "paired_contrast_rows": len(paired),
        "trials_per_topic": 10,
        "rounds_per_agent": 16,
        "final_round": 15,
        "summary_trajectory_max_absolute_error": 0.0,
    }
    return paired, cube, gap_wide, design_audit


def hierarchical_bootstrap(
    cube: np.ndarray,
    *,
    replicates: int,
    seed: int,
    chunk_size: int,
) -> np.ndarray:
    require(cube.shape == (5, 10, 2), "Bootstrap cube must be 5 x 10 x 2.")
    rng = np.random.default_rng(seed)
    draws = np.empty((replicates, 2), dtype=float)
    for start in range(0, replicates, chunk_size):
        stop = min(start + chunk_size, replicates)
        size = stop - start
        sampled_topics = rng.integers(0, 5, size=(size, 5))
        sampled_seeds = rng.integers(0, 10, size=(size, 5, 10))
        sampled = cube[
            sampled_topics[:, :, None],
            sampled_seeds,
            :,
        ]
        draws[start:stop] = sampled.mean(axis=(1, 2))
    require(np.isfinite(draws).all(), "Bootstrap produced non-finite draws.")
    return draws


def bootstrap_table(
    cube: np.ndarray,
    draws: np.ndarray,
    spec: Mapping[str, Any],
) -> pd.DataFrame:
    estimates = cube.mean(axis=(0, 1))
    standard_errors = draws.std(axis=0, ddof=1)
    require(
        np.isfinite(standard_errors).all() and (standard_errors > 0).all(),
        "Every primary contrast must have a positive finite bootstrap SE.",
    )
    studentized = (draws - estimates[None, :]) / standard_errors[None, :]
    maximum = np.max(np.abs(studentized), axis=1)
    critical = float(np.quantile(maximum, 0.95))
    pointwise = np.quantile(draws, [0.025, 0.975], axis=0)
    bonferroni = np.quantile(draws, [0.0125, 0.9875], axis=0)
    rows = []
    for index, comparison in enumerate(COMPARISON_NAMES):
        rows.append(
            {
                "comparison": comparison,
                "estimate": float(estimates[index]),
                "bootstrap_standard_error": float(standard_errors[index]),
                "pointwise_ci_lower": float(pointwise[0, index]),
                "pointwise_ci_upper": float(pointwise[1, index]),
                "familywise_ci_lower": float(
                    estimates[index] - critical * standard_errors[index]
                ),
                "familywise_ci_upper": float(
                    estimates[index] + critical * standard_errors[index]
                ),
                "max_abs_t_critical_value": critical,
                "bonferroni_percentile_ci_lower": float(
                    bonferroni[0, index]
                ),
                "bonferroni_percentile_ci_upper": float(
                    bonferroni[1, index]
                ),
                "pointwise_confidence_level": 0.95,
                "familywise_confidence_level": 0.95,
                "bonferroni_marginal_confidence_level": 0.975,
                "family_size": 2,
                "bootstrap_replicates": int(spec["bootstrap"]["replicates"]),
                "bootstrap_seed": int(spec["bootstrap"]["random_seed"]),
            }
        )
    return pd.DataFrame(rows)


def signed_t(values: np.ndarray) -> np.ndarray:
    n = values.shape[0]
    means = values.mean(axis=0)
    standard_deviations = values.std(axis=0, ddof=1)
    output = np.empty(values.shape[1], dtype=float)
    for index, (mean, standard_deviation) in enumerate(
        zip(means, standard_deviations)
    ):
        if standard_deviation > 0:
            output[index] = mean / (standard_deviation / math.sqrt(n))
        elif abs(mean) <= TAIL_TOLERANCE:
            output[index] = 0.0
        else:
            output[index] = math.copysign(math.inf, mean)
    return output


def sign_label(value: float) -> str:
    if value > TAIL_TOLERANCE:
        return "positive"
    if value < -TAIL_TOLERANCE:
        return "negative"
    return "zero"


def exact_sign_flip_tables(
    cube: np.ndarray,
    topic_rows: Sequence[Mapping[str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    topic_means = cube.mean(axis=1)
    observed_t = signed_t(topic_means)
    null_rows: list[dict[str, Any]] = []
    null_t = np.empty((32, 2), dtype=float)
    for mask in range(32):
        signs = np.array(
            [1.0 if mask & (1 << index) else -1.0 for index in range(5)]
        )
        statistics = signed_t(topic_means * signs[:, None])
        null_t[mask] = statistics
        null_rows.append(
            {
                "assignment_mask": mask,
                "topic_signs": ",".join(
                    "+" if sign > 0 else "-" for sign in signs
                ),
                f"t__{COMPARISON_NAMES[0]}": float(statistics[0]),
                f"t__{COMPARISON_NAMES[1]}": float(statistics[1]),
                "max_abs_t": float(np.max(np.abs(statistics))),
            }
        )
    maximum = np.max(np.abs(null_t), axis=1)

    topic_contrast_rows: list[dict[str, Any]] = []
    for topic_index, topic in enumerate(topic_rows):
        for comparison_index, comparison in enumerate(COMPARISON_NAMES):
            estimate = float(topic_means[topic_index, comparison_index])
            topic_contrast_rows.append(
                {
                    "topic_order": topic_index,
                    "topic_slug": topic["slug"],
                    "topic": topic["text"],
                    "comparison": comparison,
                    "paired_seeds": 10,
                    "topic_mean_difference": estimate,
                    "topic_sign": sign_label(estimate),
                }
            )

    tests: list[dict[str, Any]] = []
    topic_frame = pd.DataFrame(topic_contrast_rows)
    for index, comparison in enumerate(COMPARISON_NAMES):
        values = topic_means[:, index]
        signs = [sign_label(float(value)) for value in values]
        marginal = float(
            np.mean(
                np.abs(null_t[:, index])
                >= abs(observed_t[index]) - TAIL_TOLERANCE
            )
        )
        adjusted = float(
            np.mean(maximum >= abs(observed_t[index]) - TAIL_TOLERANCE)
        )
        tests.append(
            {
                "comparison": comparison,
                "estimate_equal_topic_weighted": float(values.mean()),
                "observed_topic_t": float(observed_t[index]),
                "exact_two_sided_marginal_p": marginal,
                "exact_max_abs_t_adjusted_p": adjusted,
                "negative_topics": signs.count("negative"),
                "zero_topics": signs.count("zero"),
                "positive_topics": signs.count("positive"),
                "topic_sign_pattern": ",".join(
                    f"{topic_rows[row]['slug']}:{signs[row]}"
                    for row in range(5)
                ),
                "topics": 5,
                "assignments": 32,
            }
        )
    return (
        topic_frame,
        pd.DataFrame(tests),
        pd.DataFrame(null_rows),
    )


def leave_one_topic_out_tables(
    cube: np.ndarray,
    topic_rows: Sequence[Mapping[str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    topic_means = cube.mean(axis=1)
    full = topic_means.mean(axis=0)
    detail: list[dict[str, Any]] = []
    for omitted in range(5):
        retained = np.delete(topic_means, omitted, axis=0)
        estimates = retained.mean(axis=0)
        for index, comparison in enumerate(COMPARISON_NAMES):
            detail.append(
                {
                    "comparison": comparison,
                    "omitted_topic_slug": topic_rows[omitted]["slug"],
                    "omitted_topic": topic_rows[omitted]["text"],
                    "estimate": float(estimates[index]),
                    "retained_topics": 4,
                    "retained_paired_blocks": 40,
                }
            )
    detail_frame = pd.DataFrame(detail)
    ranges: list[dict[str, Any]] = []
    for index, comparison in enumerate(COMPARISON_NAMES):
        selected = detail_frame[detail_frame["comparison"] == comparison]
        minimum = float(selected["estimate"].min())
        maximum = float(selected["estimate"].max())
        minimum_topics = ",".join(
            selected.loc[
                np.isclose(
                    selected["estimate"],
                    minimum,
                    atol=FLOAT_TOLERANCE,
                    rtol=0,
                ),
                "omitted_topic_slug",
            ]
        )
        maximum_topics = ",".join(
            selected.loc[
                np.isclose(
                    selected["estimate"],
                    maximum,
                    atol=FLOAT_TOLERANCE,
                    rtol=0,
                ),
                "omitted_topic_slug",
            ]
        )
        ranges.append(
            {
                "comparison": comparison,
                "full_estimate": float(full[index]),
                "leave_one_topic_out_min": minimum,
                "leave_one_topic_out_max": maximum,
                "omission_at_min": minimum_topics,
                "omission_at_max": maximum_topics,
                "sign_stable_across_omissions": bool(
                    (selected["estimate"] < -TAIL_TOLERANCE).all()
                    or (selected["estimate"] > TAIL_TOLERANCE).all()
                ),
                "omissions": 5,
            }
        )
    return detail_frame, pd.DataFrame(ranges)


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records"))


def build_report(
    bootstrap: pd.DataFrame,
    sign_tests: pd.DataFrame,
    topic_contrasts: pd.DataFrame,
    loto_ranges: pd.DataFrame,
) -> str:
    lookup_boot = bootstrap.set_index("comparison")
    lookup_sign = sign_tests.set_index("comparison")
    lookup_loto = loto_ranges.set_index("comparison")
    labels = {
        COMPARISON_NAMES[0]: "BE − Prompt self-update",
        COMPARISON_NAMES[1]: "BE − RAG + self-update",
    }
    lines = [
        "# Matched-baseline primary-family multiplicity audit",
        "",
        "## Scope",
        "",
        "This is a transparent post-hoc multiplicity audit of the completed, "
        "registered matched-baseline primary family. The family contains only "
        "the two Belief Engine minus comparator contrasts in externally scored "
        "absolute final Pro–Con gap. Negative differences mean a smaller final "
        "gap for BE. No trajectory AUC or other secondary/diagnostic outcome is "
        "included.",
        "",
        "The analysis reconstructs 50 topic-by-paired-seed blocks from the "
        "frozen 150-debate trajectory bundle. All conditions use the same "
        "condition-blind external stance evaluator.",
        "",
        "## Results",
        "",
        "| Contrast | Estimate | Pointwise 95% CI | Simultaneous max-|t| 95% CI | Bonferroni percentile 97.5% CI | Exact topic p | Max-|t| adjusted p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for comparison in COMPARISON_NAMES:
        b = lookup_boot.loc[comparison]
        s = lookup_sign.loc[comparison]
        lines.append(
            f"| {labels[comparison]} | {b['estimate']:.4f} | "
            f"[{b['pointwise_ci_lower']:.4f}, {b['pointwise_ci_upper']:.4f}] | "
            f"[{b['familywise_ci_lower']:.4f}, {b['familywise_ci_upper']:.4f}] | "
            f"[{b['bonferroni_percentile_ci_lower']:.4f}, "
            f"{b['bonferroni_percentile_ci_upper']:.4f}] | "
            f"{s['exact_two_sided_marginal_p']:.4f} | "
            f"{s['exact_max_abs_t_adjusted_p']:.4f} |"
        )
    lines.extend(
        [
            "",
            "The simultaneous intervals use 100,000 shared hierarchical "
            "topic-then-seed resamples and the 95th percentile of the maximum "
            "absolute standardized bootstrap deviation across the two contrasts. "
            "The Bonferroni column is a percentile sensitivity using 1.25% and "
            "98.75% quantiles.",
            "",
            "## Topic signs and influence",
            "",
            "| Topic | BE − Prompt | Sign | BE − RAG | Sign |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for topic in topic_contrasts["topic_slug"].drop_duplicates():
        selected = topic_contrasts[
            topic_contrasts["topic_slug"] == topic
        ].set_index("comparison")
        p = selected.loc[COMPARISON_NAMES[0]]
        r = selected.loc[COMPARISON_NAMES[1]]
        lines.append(
            f"| {topic} | {p['topic_mean_difference']:.4f} | "
            f"{p['topic_sign']} | {r['topic_mean_difference']:.4f} | "
            f"{r['topic_sign']} |"
        )
    lines.extend(["", "Leave-one-topic-out ranges:"])
    for comparison in COMPARISON_NAMES:
        row = lookup_loto.loc[comparison]
        lines.append(
            f"- {labels[comparison]}: [{row['leave_one_topic_out_min']:.4f}, "
            f"{row['leave_one_topic_out_max']:.4f}] across the five omissions "
            f"(sign stable: {str(bool(row['sign_stable_across_omissions'])).lower()})."
        )
    lines.extend(
        [
            "",
            "The exact randomization sensitivity first averages the ten paired "
            "seeds within each topic, then exhaustively enumerates all 2^5 = 32 "
            "shared sign assignments to the two-contrast topic vector. It reports "
            "two-sided marginal and single-step max-|t| adjusted p-values. With "
            "only five topics, these p-values are necessarily coarse.",
            "",
            "## Interpretation limits",
            "",
            "- This audit controls multiplicity only for the two registered "
            "primary comparisons; it does not retrospectively promote any "
            "secondary outcome.",
            "- The hierarchical bootstrap quantifies uncertainty over topics and "
            "paired seeds under the five-topic design. Five topics still provide "
            "limited support for topic-general claims.",
            "- The exact sign-flip sensitivity assumes joint sign exchangeability "
            "of the two-contrast topic vectors under the null. Exhaustive "
            "enumeration removes Monte Carlo error, not that assumption.",
            "- A smaller externally scored final disagreement is not by itself "
            "evidence of normatively better deliberation or human-like belief "
            "change.",
            "",
        ]
    )
    return "\n".join(lines)


def analyze(
    spec_path: Path,
    output_dir: Path,
) -> None:
    require(
        not output_dir.exists(),
        f"Refusing to overwrite existing output directory: {output_dir}",
    )
    spec = load_spec(spec_path)

    paths: dict[str, Path] = {}
    input_records: dict[str, Any] = {}
    for name in spec["inputs"]:
        path, record = resolve_locked_file(spec, name)
        paths[name] = path
        input_records[name] = record

    provenance_audit = validate_frozen_provenance(spec, paths)
    paired, cube, _, design_audit = validate_and_reconstruct(
        spec,
        paths["trajectory_csv"],
        paths["summary_csv"],
        paths["campaign_manifest"],
    )

    bootstrap_config = spec["bootstrap"]
    draws = hierarchical_bootstrap(
        cube,
        replicates=int(bootstrap_config["replicates"]),
        seed=int(bootstrap_config["random_seed"]),
        chunk_size=int(bootstrap_config["chunk_size"]),
    )
    bootstrap = bootstrap_table(cube, draws, spec)
    topic_contrasts, sign_tests, sign_null = exact_sign_flip_tables(
        cube,
        spec["expected_design"]["topics"],
    )
    loto, loto_ranges = leave_one_topic_out_tables(
        cube,
        spec["expected_design"]["topics"],
    )
    report = build_report(bootstrap, sign_tests, topic_contrasts, loto_ranges)

    input_audit = {
        "status": "PASS",
        "frozen_inputs": input_records,
        "provenance": provenance_audit,
        "design": design_audit,
        "scope": {
            "primary_metric": "absolute_final_pro_con_gap",
            "contrasts": list(COMPARISON_NAMES),
            "family_size": 2,
            "secondary_outcomes_included": [],
        },
    }
    analysis = {
        "analysis_id": spec["analysis_id"],
        "classification": spec["classification"],
        "status": "PASS",
        "primary_family": {
            "metric": "absolute_final_pro_con_gap",
            "direction": "negative_difference_means_smaller_final_gap_for_be",
            "contrasts": list(COMPARISON_NAMES),
            "secondary_outcomes_included": [],
        },
        "design": design_audit,
        "bootstrap": {
            "method": bootstrap_config["method"],
            "replicates": bootstrap_config["replicates"],
            "random_seed": bootstrap_config["random_seed"],
            "chunk_size": bootstrap_config["chunk_size"],
            "shared_resamples_across_contrasts": True,
            "simultaneous_method": bootstrap_config["simultaneous_method"],
            "intervals": records(bootstrap),
        },
        "exact_topic_sign_flip": {
            "method": spec["exact_topic_sign_flip"]["method"],
            "assignments": 32,
            "shared_signs_across_contrasts": True,
            "tests": records(sign_tests),
        },
        "topic_contrasts": records(topic_contrasts),
        "leave_one_topic_out_ranges": records(loto_ranges),
        "interpretation": {
            "posthoc": True,
            "five_topic_limit": True,
            "smaller_gap_not_normatively_better_by_itself": True,
            "prohibited": spec["interpretation"]["prohibited"],
        },
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.",
            dir=str(output_dir.parent),
        )
    )
    try:
        write_json(staging / "input_audit.json", input_audit)
        write_json(staging / "analysis.json", analysis)
        write_csv(staging / "paired_contrasts.csv", paired)
        write_csv(staging / "topic_contrasts.csv", topic_contrasts)
        write_csv(staging / "bootstrap_intervals.csv", bootstrap)
        write_csv(staging / "exact_topic_sign_flip_tests.csv", sign_tests)
        write_csv(staging / "exact_topic_sign_flip_null.csv", sign_null)
        write_csv(staging / "leave_one_topic_out.csv", loto)
        write_csv(staging / "leave_one_topic_out_ranges.csv", loto_ranges)
        write_text(staging / "report.md", report)

        output_records = {
            name: {
                "sha256": sha256(staging / name),
                "bytes": (staging / name).stat().st_size,
            }
            for name in ANALYSIS_OUTPUT_NAMES
        }
        manifest = {
            "schema_version": 1,
            "analysis_id": spec["analysis_id"],
            "status": "PASS",
            "classification": spec["classification"],
            "integrity_scope": (
                "analysis_outputs inventories every analyzer payload. "
                "manifest.json is the self-referential trust root; "
                "verification.json is independently generated after analysis."
            ),
            "specification": {
                "path": str(spec_path.relative_to(ROOT)),
                "sha256": sha256(spec_path),
                "bytes": spec_path.stat().st_size,
            },
            "analyzer": {
                "path": "scripts/analyze_matched_baseline_primary_multiplicity.py",
                "sha256": sha256(Path(__file__).resolve()),
            },
            "inputs": input_records,
            "design": design_audit,
            "primary_family": {
                "metric": "absolute_final_pro_con_gap",
                "contrasts": list(COMPARISON_NAMES),
                "family_size": 2,
                "secondary_outcomes_included": [],
            },
            "bootstrap": {
                "method": bootstrap_config["method"],
                "replicates": bootstrap_config["replicates"],
                "random_seed": bootstrap_config["random_seed"],
                "chunk_size": bootstrap_config["chunk_size"],
                "shared_resamples_across_contrasts": True,
                "simultaneous_method": bootstrap_config[
                    "simultaneous_method"
                ],
            },
            "exact_topic_sign_flip": {
                "method": spec["exact_topic_sign_flip"]["method"],
                "assignments": 32,
                "shared_signs_across_contrasts": True,
            },
            "runtime": {
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "pyyaml": yaml.__version__,
            },
            "portable_invocation": [
                "python3",
                "scripts/analyze_matched_baseline_primary_multiplicity.py",
                "--spec",
                "MATCHED_BASELINE_PRIMARY_MULTIPLICITY_POSTHOC_SPEC.yaml",
                "--output-dir",
                "results/analyses/matched_baseline_primary_multiplicity_posthoc_v1",
            ],
            "analysis_outputs": output_records,
        }
        write_json(staging / "manifest.json", manifest)
        os.replace(staging, output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print(
        f"PASS: wrote matched-baseline primary multiplicity audit to {output_dir}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyze(args.spec.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
