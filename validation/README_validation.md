# Validation Notes For Reviewers

This directory contains the paper-facing validation code and curated outputs
included in the anonymous artifact. It is not the full historical validation
workspace.

## Cold Verification

From the repository root:

```bash
make -f paper_repro/Makefile reviewer-check
```

This verifies code syntax/imports, anonymity hygiene, the compact paper artifact
bundle, and the full release manifest. It does not call hosted models.

## Exact Paper Artifact Bundle

The primary verification target is:

```text
validation/paper_artifacts/
```

The checksum manifest for that bundle is:

```text
validation/paper_artifacts/MANIFEST.sha256
```

To verify only this bundle:

```bash
make -f paper_repro/Makefile check-artifacts
```

## Included Validation Code

- `data/load_debate.py`, `data/preprocess.py`, `data/scale_mapping.py`:
  lightweight DEBATE loading and scale conversion helpers.
- `evaluation/common.py`: shared evaluation utilities.
- `fitting/`: metrics and parameter-sweep helpers.
- `scripts/`: selected paper-facing artifact preparation, plotting, and summary
  scripts.

## Included Result Material

- `paper_artifacts/evidence_alignment/`: compact DEBATE replay tables, fold
  statistics, and paper figures.
- `paper_artifacts/two_agent_focus10/`: compact two-agent topic-grid tables and
  figures.
- `results/`: selected compact source CSVs/PDFs used by optional local rebuild
  targets. These are curated inputs, not complete raw result trees.

## Optional Local Rebuilds

These commands rebuild selected outputs from included compact inputs:

```bash
make -f paper_repro/Makefile rebuild-debate-rmse-stats
make -f paper_repro/Makefile rebuild-five-topic-table
make -f paper_repro/Makefile rebuild-paper-artifacts
```

They may overwrite generated files in `validation/paper_artifacts/` or
`validation/results/`.

## Not Included

The anonymous artifact excludes:

- `validation/data/cache/` extraction caches;
- raw `validation/results/*/raw_runs/` trees;
- model weights such as `models/argument_quality_classifier/`;
- local copies of Argument-Quality-Ranking-30k;
- draft validation notes under `validation/docs/`;
- full hosted-model logs.

Those materials should live in a separate public archive after review if full
rerun auditability is required.
