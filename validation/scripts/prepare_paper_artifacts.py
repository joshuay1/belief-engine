#!/usr/bin/env python3
"""Refresh compact paper artifact metadata and derived fold statistics.

The exact paper-facing artifacts live directly under ``validation/paper_artifacts``.
This script intentionally does not copy duplicate result trees into that bundle.
It refreshes the fold-level DEBATE uncertainty tables from the compact held-out
inputs that remain in ``validation/results`` and then rewrites the artifact
manifests.
"""

from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from validation.scripts.summarize_debate_rmse_uncertainty import (
    build_outputs as build_debate_rmse_uncertainty,
)


ARTIFACT_ROOT = REPO_ROOT / "validation" / "paper_artifacts"
EVIDENCE_ARTIFACT_DIR = ARTIFACT_ROOT / "evidence_alignment"
TWO_AGENT_ARTIFACT_DIR = ARTIFACT_ROOT / "two_agent_focus10"

EVIDENCE_GROUP_SOURCE = (
    REPO_ROOT
    / "validation"
    / "results"
    / "parameter_recommendations_group_response5_evidence13x13_quality"
)
EVIDENCE_TOPIC_SOURCE = (
    REPO_ROOT
    / "validation"
    / "results"
    / "parameter_recommendations_topic_response5_evidence13x13_quality"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def refresh_debate_uncertainty() -> None:
    build_debate_rmse_uncertainty(
        group_dir=EVIDENCE_GROUP_SOURCE,
        topic_dir=EVIDENCE_TOPIC_SOURCE,
        output_dir=EVIDENCE_ARTIFACT_DIR,
    )


def iter_artifacts() -> list[Path]:
    return sorted(
        path
        for path in ARTIFACT_ROOT.rglob("*")
        if path.is_file() and path.name != "MANIFEST.sha256"
    )


def write_checksum_manifest() -> None:
    lines = []
    for path in iter_artifacts():
        rel = path.relative_to(REPO_ROOT).as_posix()
        lines.append(f"{sha256(path)}  {rel}")
    (ARTIFACT_ROOT / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")


def count_run_specs() -> int:
    run_specs_path = TWO_AGENT_ARTIFACT_DIR / "run_specs.csv"
    with run_specs_path.open(newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def write_yaml_manifest() -> None:
    files = iter_artifacts()
    evidence_count = len([p for p in files if "evidence_alignment" in p.parts])
    two_agent_count = len([p for p in files if "two_agent_focus10" in p.parts])
    topic_grid_count = len([p for p in files if "topic_grids" in p.parts and p.suffix == ".pdf"])
    run_spec_count = count_run_specs()
    text = f"""paper_artifacts_version: "2026-05-07"
generated_at_utc: "not timestamped; manifest is deterministic"
artifact_root: "validation/paper_artifacts"

reproducibility_boundary:
  exact_paper_figures: "Use the committed CSV/PDF/TeX files in validation/paper_artifacts."
  full_api_rerun: "Requires external data/API access and is not expected to be byte-identical because hosted LLM behavior can change."

evidence_alignment:
  artifact_dir: "validation/paper_artifacts/evidence_alignment"
  fold_statistics_inputs:
    group_heldout: "validation/results/parameter_recommendations_group_response5_evidence13x13_quality"
    topic_heldout: "validation/results/parameter_recommendations_topic_response5_evidence13x13_quality"
  selected_files: {evidence_count}
  primary_figures:
    - "evidence_alignment/figure_parameter_rmse_surfaces_evidence_alignment_focus.pdf"
    - "evidence_alignment/figure_parameter_rmse_surfaces_evidence_aligned_topic_category.pdf"
  primary_tables:
    - "evidence_alignment/evidence_alignment_focus_baseline_metrics.csv"
    - "evidence_alignment/parameter_rmse_surface_evidence_alignment_focus.csv"
    - "evidence_alignment/parameter_rmse_surface_evidence_aligned_topic_category.csv"
    - "evidence_alignment/debate_rmse_confidence_intervals.csv"
    - "evidence_alignment/debate_rmse_fold_tests.csv"

two_agent_focus10:
  artifact_dir: "validation/paper_artifacts/two_agent_focus10"
  selected_files: {two_agent_count}
  completed_runs: 120
  topics: 10
  pairings: 4
  trials_per_topic_pairing: 3
  rounds: 15
  backend: "openai"
  model: "gpt-4o-mini"
  topic_grid_pdfs: {topic_grid_count}
  run_specs: {run_spec_count}
  primary_tables:
    - "two_agent_focus10/topic_convergence_by_pairing.csv"
    - "two_agent_focus10/topic_convergence_diagnostics.csv"
    - "two_agent_focus10/topic_convergence_by_pairing_table.tex"
  regenerate_configs_command: >-
    python validation/scripts/run_two_agent_controlled.py --preset human_showcase_focus10
    --output-dir validation/results/two_agent_focus10_gpt4omini
    --dry-run --backend-type openai --model gpt-4o-mini --rounds 15

checksums: "MANIFEST.sha256"
"""
    (ARTIFACT_ROOT / "manifest.yaml").write_text(text)


def main() -> int:
    if not ARTIFACT_ROOT.exists():
        raise FileNotFoundError(ARTIFACT_ROOT)
    refresh_debate_uncertainty()
    write_yaml_manifest()
    write_checksum_manifest()
    print(f"Refreshed paper artifact metadata in {ARTIFACT_ROOT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
