# Reproducing the paper

Start with the installation steps in the [README](README.md).
The saved results allow the reported numerical analyses to be checked without
running language models. Live experiments additionally require the relevant datasets and models.

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
The [result index](validation/README.md) links to the saved sweeps, baseline
comparisons, replay tables, topic grids, and validation outputs.

The six camera-ready packages can be rebuilt and checked with:

```bash
make rebuild-paper
```

This runs the original numerical analyses on compact saved inputs, checks the
reported estimates and primary confidence intervals, and writes tables and plots
to `output/paper/`. It requires no API keys or model downloads. Allow a few minutes
for the endpoint search and bootstrap calculations. The
[paper result guide](validation/camera_ready/README.md) maps each study to its
inputs, expected results, and a command for rebuilding it separately.

The [final figure index](validation/camera_ready/figures/README.md) connects the
manuscript assets to these result packages and the older saved outputs. Run
`make check-figures` to verify their checksums, the 150 BE sweep trajectories,
and the separately recovered GPT-5.4-mini profile trajectories.

## 3. Run experiments with the correct update mode

The update mode is part of the experiment configuration:

| `memory.belief_semantics_mode` | Behavior |
|---|---|
| `incremental_v1` | Historical default. Newly admitted arguments accumulate. Replacement does not undo earlier contributions. |
| `active_seed_v1` | Recomputes from active seed and later-message records, weighted by anchoring and uptake. |
| `active_explicit_prior_v1` | Adds active non-seed evidence to a fixed initial stance. Seeds remain retrievable but do not enter the numerical sum. |

Use the original mode when reproducing an existing run. The generated-agent
fixed-prior mode differs from DEBATE replay, where anchoring scales the observed
initial log odds. Use the result index to find the included artifacts.

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
test coverage, and release preparation details.
