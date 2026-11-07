# Belief Engine

Belief Engine separates evidence extraction, memory, stance updating, and reply
generation so that an agent's stance changes can be inspected and controlled.

Companion code for **Belief Engine: Making Agent Stance Auditable in LLM Deliberation**.

**Status:** local camera-ready candidate. Updated code and tests are included.
The newer experiment result packages are still being added.

## Quick start

Use Python 3.11:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[analysis]'
make check
make example
```

`make check` runs offline code tests and verifies the included files.
`make example` demonstrates how replacing an argument affects stance under the
three update modes. Neither command needs API keys or model downloads.

## Reproduce the results

The [reproduction guide](REPRODUCING.md) explains which results are included,
how to rebuild them, and what is needed for live experiments. It also describes
which update mode belongs to each protocol.

The [argument-quality scorer](https://huggingface.co/joshcyang/belief-engine-argument-quality-deberta-v3-large)
is available separately on Hugging Face. Its pinned configuration is in
[`config/camera_ready/argument_quality.yaml`](config/camera_ready/argument_quality.yaml).

Source provenance, test details, and remaining release tasks are in the
[maintainer notes](paper_repro/CAMERA_READY.md).
