# Reproducing the paper

Start with the installation steps in the [README](README.md).
The saved results allow selected analyses to be checked without running language
models. Live experiments additionally require the relevant datasets and models.

## 1. Verify the code and included files

```bash
make check
make example
```

These commands run offline behavioral tests, verify file checksums, and show a
small update example. They do not rerun the paper's experiments. Test inputs use
synthetic evidence, scripted responses, and a small test tokenizer.

## 2. Rebuild analyses from saved results

```bash
make rebuild-replay-stats
```

This rebuilds the DEBATE fold-level RMSE statistics from the included participant
prediction tables. Outputs go to `output/replay/`, leaving the frozen results intact.
The [saved paper artifacts](validation/paper_artifacts/) and
[selected result files](validation/results/) also include the original sweeps,
earlier baseline comparisons, topic grids, and stance-validation outputs.

The ten-topic stance validation, 150-debate common-evaluator comparison,
finite-solver analysis, and component-study result packages are still pending.
The original 100-input AQR sample also needs its exact classifier predictions and
sample mapping. The final release will map each reported result to its input
files, configuration, and rebuild command.

## 3. Run experiments with the correct update mode

The update mode is part of the experiment configuration:

| `memory.belief_semantics_mode` | Behavior |
|---|---|
| `incremental_v1` | Historical default. Newly admitted arguments accumulate. Replacement does not undo earlier contributions. |
| `active_seed_v1` | Recomputes from active seed and later-message records, weighted by anchoring and uptake. |
| `active_explicit_prior_v1` | Adds active non-seed evidence to a fixed initial stance. Seeds remain retrievable but do not enter the numerical sum. |

Use the original mode when reproducing an existing run. The generated-agent
fixed-prior mode differs from DEBATE replay, where anchoring scales the observed
initial log odds. The earlier [experiment inventory](paper_repro/PAPER_EXPERIMENT_INVENTORY.md)
describes the reviewer snapshot and is not yet a complete camera-ready run guide.

Live reruns require access to the named base model or hosted API, the appropriate
dataset, and the experiment's exact configuration and prompts. Hosted-model
outputs can vary across reruns. Use the saved outputs to reproduce the reported
numerical summaries.

### Argument-quality scorer

```bash
pip install -e '.[classifier]'
make test-classifier
```

The integration tests use synthetic tensors and do not download weights. For a
live classifier run, merge the memory block in
[`argument_quality.yaml`](config/camera_ready/argument_quality.yaml) into its
configuration. It pins the published model to release `original-2025-12-v1`.
The full revision is recorded in that file. Follow the
[model release instructions](https://huggingface.co/joshcyang/belief-engine-argument-quality-deberta-v3-large)
to verify downloaded weights against reference scores.

The scorer takes claim text alone, uses 128-token inputs, and clips raw regression
outputs to [0, 1] without a sigmoid. Loading failures stop classifier-based runs
unless fallback is explicitly enabled. Embedding-based deduplication additionally
requires `pip install -e '.[embeddings]'` and its embedding model.

## Environment and provenance

The tested Python 3.11 dependency versions are recorded in
[`paper_repro/requirements-verified-py311.txt`](paper_repro/requirements-verified-py311.txt).
The [maintainer notes](paper_repro/CAMERA_READY.md) retain implementation provenance,
test coverage, and release preparation details. Original results are preserved.
