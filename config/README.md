# Configuration

| File | Purpose |
|---|---|
| `camera_ready/argument_quality.yaml` | Pinned Hugging Face scorer settings to merge into an experiment |
| `selected_results/canonical_sensitivity_sweep_deterministic_counter.yaml` | Historical sweep configuration |
| `selected_results/canonical_stance_correlation.yaml` | Earlier stance-expression calibration |
| `defaults.yaml` | General runtime defaults, not a complete paper experiment specification |

Use the [reproduction guide](../REPRODUCING.md) to choose the correct update mode.
The saved historical configurations may include comparison agents outside the
current main results. They are retained to preserve their original meaning.

Live runs require the named model/API and any referenced datasets or weights.
For a first run without those dependencies, use `make example` from the root.
The newer studies' exact configurations will be added with their result packages.
