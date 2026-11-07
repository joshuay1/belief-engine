# Release Allowlist

This file defines what should be kept in the clean anonymous reviewer artifact,
what should be archived externally, and what should be left out.

The rule of thumb: keep code, small canonical tables, paper-facing figures, and
checksums in Git; put large raw logs, model weights, and bulky caches in an
external archive with checksums.

## Keep In Git

### Project Metadata

- `README.md`
- `pyproject.toml`
- `.env.example`
- `.gitignore`
- `paper_repro/**`

### Core Source Code

Keep the full `src/` package. It is small enough to ship as a unit and avoids
accidentally breaking imports by over-pruning.

- `src/**`

Prune generated local files before release:

- `src/**/__pycache__/**`
- `src/**/*.pyc`

### Validation Code Needed By The Paper

Keep these helper packages:

- `validation/data/load_debate.py`
- `validation/data/preprocess.py`
- `validation/data/scale_mapping.py`
- `validation/evaluation/common.py`
- `validation/fitting/*.py`

Keep these paper-facing scripts:

- `validation/scripts/prepare_paper_artifacts.py`
- `validation/scripts/build_6panel_topic_grids_and_metrics.py`
- `validation/scripts/plot_model_sensitivity_comparison.py`
- `validation/scripts/plot_top_matched_movers.py`
- `validation/scripts/summarize_debate_rmse_uncertainty.py`
- `validation/scripts/run_parameter_recommendations.py`
- `validation/scripts/generate_parameter_recommendation_figures.py`
- `validation/scripts/evaluate_judgement_layer_aqr.py`
- `validation/scripts/run_aqr_llm_extraction_eval.py`
- `validation/scripts/run_two_agent_controlled.py`
- `validation/scripts/run_two_agent_model_comparison.py`

### Configs

- `config/defaults.yaml`
- `config/README.md`
- `config/selected_results/canonical_stance_correlation.yaml`
- `config/selected_results/canonical_sensitivity_sweep_deterministic_counter.yaml`
- `experiments/selected_results/canonical_6cond_symm/config/paper_campaign_config.yaml`

### Exact Paper Artifact Bundle

Keep the compact artifact bundle and checksum manifest:

- `validation/paper_artifacts/**`

This is the primary exact-reproduction target. The command
`python paper_repro/scripts/check_artifacts.py` verifies it.

### Curated Canonical Result Files

Keep selected files from these folders because they directly support paper
figures/tables. Do not copy whole result trees unless the folder is already a
compact artifact bundle; most result directories contain old PNG/SVG variants,
draft diagnostics, or repeated run configs that are not part of the paper
release boundary.

- `validation/results/model_sensitivity_comparison/` selected BE CSV/PDF outputs only
- `validation/results/all_topics_u0.2_a0.4_8models/` five-topic 6-panel inputs, metrics, and PDFs only
- `validation/results/individual_opinion_tracking_response5_evidence13x13/` type-showcase CSV/PDF outputs only
- `validation/results/parameter_recommendations_group_response5_evidence13x13_quality/` compact held-out parameter tables needed by RMSE/showcase rebuilds
- `validation/results/parameter_recommendations_topic_response5_evidence13x13_quality/` compact held-out parameter tables needed by RMSE rebuilds
- `validation/results/judgement_layer_aqr/aqr_test_20260426_133422/` compact AQR report, metrics, and PDF only
- `validation/results/judgement_layer_aqr_llm/aqr_llm_gpt-4o-mini_20260426_141615/` compact LLM AQR report, metrics, and PDF only
- `experiments/selected_results/canonical_6cond_symm/`

For the stance-expression mapping scalar claim, keep only the compact CSV and
plot rather than the full raw log folder:

- `experiments/logs/stance_correlation_20260213_165900/data/results.csv`
- `experiments/logs/stance_correlation_20260213_165900/plots/stance_correlation.png`

## Archive Externally

These are too large or too exploratory for Git, but should be open-access if
they are needed for full reruns or audit trails:

- `models/argument_quality_classifier/` especially `model.safetensors`
- `validation/data/cache/`
- `data/Argument-Quality-Ranking-30k/` if redistribution is permitted; otherwise
  document the upstream source and checksum the local copy
- `logs/GPT-4o-mini_canonical_sensitivity_sweep_deterministic_counter_20260401_150045/`
- `logs/QWEN_canonical_sensitivity_sweep_deterministic_counter_20260423_103726/`
- `logs/GEMMA4_canonical_sensitivity_sweep_deterministic_counter_20260427_113651/`
- `validation/results/two_agent_focus10_gpt4omini/` full raw/config tree
- any `validation/results/*/raw_runs/`
- any `validation/results/*/.mplconfig/`
- all non-allowlisted `validation/results/*`
- all non-allowlisted `experiments/logs/*`

For each external archive, publish:

- archive URL or DOI
- SHA256 manifest
- README mapping archive paths back to paper figures/tables
- note on whether hosted LLM reruns are expected to be exact or only
  behaviorally comparable

## Leave Out Of The Release

- `__pycache__/`, `.pytest_cache/`, `.mplconfig/`
- notebooks used only for scratch analysis
- obsolete draft paper assets not referenced by the active manuscript
- exploratory validation folders not named in this allowlist or
  `paper_repro/PAPER_EXPERIMENT_INVENTORY.md`
- local logs containing API traces, timestamps, or private environment details

## Release Hygiene

Before handing the artifact to reviewers, run:

```bash
make -f paper_repro/Makefile reviewer-check
```

That target verifies Python syntax/imports, the compact paper-artifact
checksums, the top-level release manifest, and common anonymity hazards such as
generated files, raw-run folders, local absolute paths, and likely secrets.
