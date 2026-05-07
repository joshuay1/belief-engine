# Paper Repro Release Audit

Audit date: 2026-05-07
Release root: anonymous reviewer artifact checkout

This audit records the practical release boundary for the anonymous reviewer
artifact. The boundary is deliberately narrower than the full development
workspace: it keeps code, compact paper-facing artifacts, selected rebuild
inputs, and checksums; it excludes raw hosted-model traces, large caches, model
weights, private manuscript source, and contributor metadata.

## Current Reviewer Boundary

The reviewer-facing boundary is:

- `README.md`
- `pyproject.toml`
- `.env.example`
- `.gitignore`
- `src/**`
- `paper_repro/**`
- `config/**` selected paper-facing configs
- `data/focus_topics.json`
- `experiments/selected_results/**`
- `experiments/logs/stance_correlation_20260213_165900/` compact CSV/plot only
- `validation/paper_artifacts/**`
- selected `validation/data/`, `validation/evaluation/`, `validation/fitting/`,
  `validation/scripts/`, and compact non-duplicated `validation/results/` paths
  listed in the allowlist

The exact paper-artifact target is the 36-entry checksum manifest at:

- `validation/paper_artifacts/MANIFEST.sha256`

The full release file list is recorded at:

- `REVIEWER_RELEASE_MANIFEST.sha256`

## Verified Commands

These are the cold reviewer commands:

```bash
make -f paper_repro/Makefile reviewer-check
make -f paper_repro/Makefile check-artifacts
make -f paper_repro/Makefile check-release-manifest
```

Expected outputs:

- `reviewer-check` verifies Python 3.11, syntax, key imports, release hygiene,
  the paper-artifact manifest, and the top-level release manifest.
- `check-artifacts` prints `Artifact check passed.` for
  `validation/paper_artifacts/MANIFEST.sha256`.
- `check-release-manifest` prints `Artifact check passed.` for
  `REVIEWER_RELEASE_MANIFEST.sha256`.

None of these commands call hosted models.

## Keep In Git

Keep small, paper-facing material that a reviewer can inspect directly:

- source code under `src/**`;
- reproducibility scripts and manifests under `paper_repro/**`;
- selected configs under `config/**`;
- compact paper artifacts under `validation/paper_artifacts/**`;
- selected compact result CSV/PDF/Markdown files required by optional local
  rebuilds;
- lightweight focus-topic data and selected compact experiment outputs.

## Archive Externally

These should not be folded into the clean reviewer repository just because they
exist locally. Archive them separately with checksums if they are needed for
full reruns or audit trails:

- `validation/data/cache/**`;
- non-allowlisted `validation/results/**`;
- raw run folders under `validation/results/*/raw_runs/`;
- local `.mplconfig` folders under result directories;
- model/data assets such as `models/argument_quality_classifier/**`;
- local copies of `data/Argument-Quality-Ranking-30k/**`;
- full hosted-model logs.

## Leave Out

- `__pycache__/`, `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`, `.mplconfig/`;
- notebooks and scratch configs used only for exploratory analysis;
- draft validation notes under `validation/docs/**`;
- obsolete paper drafts or manuscript-side notes not referenced by the active
  reproducibility manifest;
- `.env`, API keys, local absolute paths, or private manuscript paths.

## Remaining Release Tasks

- Publish the external archive URL or DOI for large raw/intermediate data.
- Add an archive-level SHA256 manifest and a figure/table-to-path mapping once
  the external archive is frozen.
