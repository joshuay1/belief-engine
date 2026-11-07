# Belief Engine

Code candidate for **Belief Engine: Making Agent Stance Auditable in LLM Deliberation**.

This checkout starts from the anonymous reviewer artifact frozen on May 7, 2026.
It adds the updated runtime, regression tests, and integration with the published
argument-quality scorer. The original compact results remain unchanged. The
newer camera-ready result packages still need to be added before publication.

## Install and check

Use Python 3.11. From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[analysis]'
make -f paper_repro/Makefile camera-ready-check
python -B examples/update_modes.py
```

The check runs syntax/import checks, behavioral tests, release hygiene checks,
and both the original paper-artifact checksums and the current code checksums.
Behavioral tests block outbound network connections. They use synthetic evidence,
scripted language-model responses, and a small test tokenizer. No API keys,
model weights, or third-party datasets are required for these checks.

Optional classifier integration checks:

```bash
python -m pip install -e '.[classifier]'
make -f paper_repro/Makefile test-classifier
```

These tests exercise the production scorer using synthetic tensors. They check
score clipping, input routing, and failure handling without downloading weights.
For the published checkpoint's verification examples, use the instructions in the
[Hugging Face model release](https://huggingface.co/joshcyang/belief-engine-argument-quality-deberta-v3-large).

## Update modes

Select `memory.belief_semantics_mode` explicitly for new experiments.

| Mode | Prior and evidence behavior |
|---|---|
| `incremental_v1` | Historical default. Newly admitted arguments accumulate. Replacing a record does not undo its earlier contribution. |
| `active_seed_v1` | Recomputes from active records. Anchoring weights seed arguments and uptake weights later arguments. |
| `active_explicit_prior_v1` | Keeps a fixed initial stance and recomputes active non-seed evidence. Seeds remain retrievable but do not enter the sum, so anchoring has no numerical effect. |

The observed-initial-stance DEBATE replay has a separate prior-scaling formulation.
Do not substitute the fixed-prior generated-agent mode for that replay protocol.

The runtime also supports strict structured extraction, matched response prompts,
active-memory proportional retrieval, and the component-comparison controls.
Legacy extraction and response behavior remain the defaults unless opted in.

## Argument-quality model

Weights remain on Hugging Face. The pinned configuration is
[`config/camera_ready/argument_quality.yaml`](config/camera_ready/argument_quality.yaml).
Merge that memory block into the configuration for a classifier-based experiment.
It selects `original-2025-12-v1`, commit
`139279e9cbecea92a1778318c91547995ea8e0d4`.

The scorer consumes claim text alone, truncates/pads to 128 tokens, and clips raw
regression outputs to [0, 1]. It does not apply a sigmoid. Classifier-based runs
stop if loading fails unless fallback is explicitly enabled. Embedding-based
live deduplication additionally needs `pip install -e '.[embeddings]'` and its
embedding model. The optional exploratory agentic-memory backend uses `.[agentic]`.

## Release status and provenance

See [`paper_repro/CAMERA_READY.md`](paper_repro/CAMERA_READY.md) for the source
revisions, test coverage, and remaining packaging work. The original release is
preserved in the first Git commit. Its manifest and build record are retained in
`paper_repro/provenance/`. Imported source hashes are recorded there as well.

`validation/paper_artifacts/MANIFEST.sha256` still verifies the original 36-file
paper bundle. `CAMERA_READY_MANIFEST.sha256` verifies this code candidate.
The older experiment inventory is historical and does not yet describe all
camera-ready results.

This is a local release candidate, not a published complete paper release. The
root project license and final public repository URL remain to be finalized.
The imported argument-quality scorer retains its MIT notice in
`src/classifiers/LICENSE.argument_quality`.
