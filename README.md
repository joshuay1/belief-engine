# Anonymous Reviewer Artifact

This repository is a compact, anonymous artifact for double-blind review. It is
designed to let reviewers inspect the code, verify the selected paper artifacts,
and rerun the self-contained checks without hosted-model calls.

The artifact is intentionally narrower than the full development repository.
Large caches, raw hosted-model traces, model weights, unsettled third-party data
redistribution, manuscript source files, and contributor metadata are excluded.

## Quick Start

From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[analysis]"
make -f paper_repro/Makefile reviewer-check
```

The `reviewer-check` target performs four cold checks:

- verifies Python 3.11+, syntax, and key imports;
- scans for generated files, raw-run folders, local absolute paths, and likely
  secrets;
- verifies `validation/paper_artifacts/MANIFEST.sha256`;
- verifies the top-level `REVIEWER_RELEASE_MANIFEST.sha256`.

These checks do not call OpenAI, OpenRouter, or any other hosted model.

## What Is Included

- `src/`: core agent, memory, belief-update, backend, and simulation code.
- `paper_repro/`: reviewer Makefile, manifests, audit notes, experiment
  inventory, and verification scripts.
- `validation/paper_artifacts/`: the exact compact paper artifact bundle. This
  is the primary byte-level verification target.
- `validation/scripts/`, `validation/fitting/`, `validation/evaluation/`, and
  `validation/data/`: selected paper-facing rebuild and analysis code.
- `validation/results/`: selected compact source tables/figures needed by
  optional rebuild commands. These folders are curated, not complete historical
  result trees.
- `config/`, `data/focus_topics.json`, and `experiments/selected_results/`:
  selected paper-facing configuration and compact supporting artifacts.

## Reproducibility Levels

The release separates three levels of reproducibility:

1. Exact artifact verification: run `make -f paper_repro/Makefile
   reviewer-check`. This is fully self-contained.
2. Local rebuilds from included compact CSV/PDF inputs: use the rebuild targets
   below. These are intended for inspection and may rewrite generated outputs.
3. Full live reruns: require external datasets, excluded caches/model weights,
   and/or hosted-model API keys. They are not expected to be byte-identical
   across time.

Optional local rebuild targets:

```bash
make -f paper_repro/Makefile rebuild-debate-rmse-stats
make -f paper_repro/Makefile rebuild-five-topic-table
make -f paper_repro/Makefile rebuild-paper-artifacts
```

## Main Verification Target

The compact paper bundle lives at:

```text
validation/paper_artifacts/
```

Its checksum manifest is:

```text
validation/paper_artifacts/MANIFEST.sha256
```

To verify only that bundle:

```bash
make -f paper_repro/Makefile check-artifacts
```

## Exclusions

Excluded from this anonymous artifact:

- `.git` history from the development repository, remotes, and contributor
  metadata;
- API keys, `.env`, and hosted-model traces;
- large extraction caches under `validation/data/cache/`;
- raw `validation/results/*/raw_runs/` trees;
- model weights such as `models/argument_quality_classifier/`;
- third-party datasets whose redistribution status is not settled;
- private manuscript source files and local paper-project paths.

After review, this artifact can be de-anonymized and paired with an external
archive for large raw/intermediate data.
