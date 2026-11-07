# Belief Engine

Belief Engine separates evidence extraction, memory, stance updating, and reply
generation so that an agent's stance changes can be inspected and controlled.

Companion code for **Belief Engine: Making Agent Stance Auditable in LLM Deliberation**.

The code, offline tests, and compact saved results are included. See the
[paper result guide](validation/camera_ready/README.md) for the camera-ready analyses.

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

## Repository layout

| Folder | Contents |
|---|---|
| `src/` | Agents, evidence scoring, memory, and stance updates |
| `config/` | Runtime and experiment settings |
| `examples/` | Small runnable examples |
| `tests/` | Offline regression tests |
| `validation/` | Analysis scripts and saved paper results |
| `paper_repro/` | Verification tools and maintainer records |

Historical stance-calibration outputs remain under `experiments/`. The small
`data/` folder contains topic names, not third-party datasets.

See the [reproduction guide](REPRODUCING.md) for result locations and the
[maintainer notes](paper_repro/CAMERA_READY.md) for release preparation.

## Licence and citation

Original code is released under the [MIT licence](LICENSE). Third-party assets
retain their own terms, as described in [NOTICE.md](NOTICE.md). Use
[CITATION.cff](CITATION.cff) to cite the paper and identify the software version.
The camera-ready release is [v1.0.0](https://github.com/joshuay1/belief-engine/releases/tag/v1.0.0).
