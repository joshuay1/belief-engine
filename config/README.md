# Configuration Notes

This reviewer artifact includes only the configuration files needed to inspect
the paper-facing runs and selected optional reruns.

## Included Files

- `defaults.yaml`: general runtime defaults for exploratory simulations. Some
  defaults point to external data/model assets that are intentionally excluded
  from the anonymous artifact.
- `selected_results/canonical_stance_correlation.yaml`: configuration for the
  stance-expression mapping claim.
- `selected_results/canonical_sensitivity_sweep_deterministic_counter.yaml`:
  configuration for the model-sensitivity sweep family.

## Reviewer Guidance

For cold verification, use the top-level reviewer command instead of starting
from `defaults.yaml`:

```bash
make -f paper_repro/Makefile reviewer-check
```

Live runs with `backend.type: openai` or `backend.type: openrouter` require
local API keys. Runs that use classifier strength scoring require the excluded
`models/argument_quality_classifier/` directory. Runs that seed from
Argument-Quality-Ranking-30k require an external copy of that dataset.
