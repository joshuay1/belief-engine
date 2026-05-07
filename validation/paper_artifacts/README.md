# Paper Artifacts

This directory is the compact reproducibility bundle for the selected paper
results. For review, this directory is the clean byte-level boundary: it
contains the CSVs, TeX tables, PDFs, and compact rerun metadata needed for the
reported figures and tables.

## Exact Figure/Table Reproduction

Use the committed files in this directory when you need to reproduce the paper
figures exactly. These frozen CSV/PDF/TeX artifacts are the stable
reproducibility boundary.

```bash
python validation/scripts/prepare_paper_artifacts.py
```

The command refreshes fold-level DEBATE uncertainty tables from the compact
held-out inputs that remain in `validation/results/`, then rewrites:

- `manifest.yaml`: human-readable provenance and counts.
- `MANIFEST.sha256`: checksums for every committed artifact.

It does not copy duplicated result trees into this directory.

## Contents

`evidence_alignment/` contains the selected DEBATE validation outputs:

- RMSE surface CSVs for the focused evidence-alignment panels.
- RMSE surface CSVs for the evidence-aligned topic-category split.
- Baseline metrics for the evidence-alignment comparison.
- Fold-level RMSE confidence intervals and paired tests:
  - `debate_rmse_confidence_intervals.csv`
  - `debate_rmse_fold_metrics.csv`
  - `debate_rmse_fold_tests.csv`
  - `debate_rmse_uncertainty_report.md`
- Paper-facing PDF figures, including:
  - `figure_parameter_rmse_surfaces_evidence_alignment_focus.pdf`
  - `figure_parameter_rmse_surfaces_evidence_aligned_topic_category.pdf`

`two_agent_focus10/` contains the selected generated two-agent robustness run:

- `summary.csv`: one row per completed debate.
- `stance_trajectories.csv`: round-by-round belief trajectories.
- `argument_uptake.csv`: argument extraction and internalization log.
- `topic_convergence_by_pairing.csv`: convergence separated by topic and profile pairing.
- `topic_convergence_diagnostics.csv`: topic-level convergence and uptake metrics.
- `topic_pairing_diagnostics.csv`: topic-by-pairing diagnostics.
- `topic_convergence_by_pairing_table.tex`: standalone appendix table.
- `topic_grids/*.pdf`: one paper-facing trajectory grid per topic.
- `run_specs.csv`: compact design table for the 120 API runs.

The 120 full YAML configs are intentionally not committed because they are
mostly repeated boilerplate. They can be regenerated from the preset whenever
needed:

```bash
python validation/scripts/run_two_agent_controlled.py \
  --preset human_showcase_focus10 \
  --output-dir validation/results/two_agent_focus10_gpt4omini \
  --dry-run \
  --backend-type openai \
  --model gpt-4o-mini \
  --rounds 15
```

## Optional Full API Rerun

The two-agent run can be rerun from the experiment preset:

```bash
python validation/scripts/run_two_agent_controlled.py \
  --preset human_showcase_focus10 \
  --output-dir validation/results/two_agent_focus10_gpt4omini \
  --backend-type openai \
  --model gpt-4o-mini \
  --rounds 15 \
  --artifact-mode compact
```

This uses real API calls. It should be treated as a reproducible experimental
procedure, not a byte-identical replay, because hosted LLM behavior may change
over time. For exact paper figures, use the frozen CSVs and PDFs committed in
this directory.
