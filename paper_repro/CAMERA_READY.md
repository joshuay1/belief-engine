# Camera-ready release v1.0.0

Prepared on November 6, 2026 from the 199-file anonymous reviewer snapshot.
This release updates executable code and tests and includes the six camera-ready
analysis packages. The original reviewer result bundle remains unchanged.

## Source integration

- Anonymous snapshot: `https://anonymous.4open.science/r/belief-engine-684F/`,
  generated May 7, 2026. The first Git commit preserves the downloaded files.
- Updated runtime and tests: frozen `agent_mem_response_factorial_v1` worktree,
  source revision `c23a9d4a4327fe1ed9493555088ff58e2107fc28`. Imports cover the
  updater, memory/retrieval, structured extraction, matched generation, audit
  metadata, component controls, and the two-agent comparison runner.
- Classifier integration: `codex/argument-evaluator-release`, revision
  `24865e47810fe454b3b37f729e11a5f8fd0b3571` (PR #12). Its loading methods were
  merged into the newer extractor rather than replacing structured extraction.
- Exact imported file hashes are in `provenance/code_sources.json`. These identify
  source bytes even where a working tree contained changes beyond its HEAD.

Local integration changes defer optional embedding/agentic imports until needed,
package the runtime JSON/text resources, add the test runner and pinned scorer
configuration, and document explicit mode selection. Two historical expected
failures became positive assertions of the known incremental behavior. The
active-record tests independently assert the new recomputation behavior.

## Behavioral coverage

`make -f paper_repro/Makefile test` checks:

- Historical accumulation, replacement, target initialization, and state schema.
- Active-seed and explicit-prior updates, archived records, prior double-counting,
  initialization, incompatible modes, and persisted mode checks.
- Active-only proportional retrieval, including quotas and archived backfill.
- Structured extraction retries, invalid/empty outputs, ordering, whole-message
  records, and fixed-strength controls.
- Matched generation inputs, recent dialogue, retrieval toggles, token budgets,
  neutral stance, and separation of internal audit metadata from reply prompts.
- Persistent versus turn-local memory, duplicate-gate controls, speaker order,
  and baseline self-update audit counters.

`make -f paper_repro/Makefile test-classifier` checks raw regression clipping,
128-token input formatting, singleton/empty batches, missing weights, LFS pointers,
Hub revision routing, explicit fallback, and classifier strengths reaching records
in both legacy and structured extraction modes.

The tests block network connections and substitute a small BPE tokenizer for
unit-level token-budget checks. They do not validate the real `cl100k_base`
vocabulary or regenerate any hosted-model experiment. Classifier tests use
synthetic tensors, not the 1.74 GB checkpoint. Verify downloaded weights using
the pinned model release's own `scripts/verify_release.py`.

The original 36 paper-artifact checksums are retained unchanged. A separate
candidate manifest covers the updated code and tests. After intended changes:

```bash
git add <reviewed-files>
python -B paper_repro/scripts/freeze_camera_ready_manifest.py
make -f paper_repro/Makefile camera-ready-check
make -f paper_repro/Makefile test-classifier
```

The manifest tool uses the staged Git file list and the current file bytes. It
never adds untracked files, includes model weights automatically, or rewrites the
original artifact checksums. Stage the new manifest before committing.

## Rebuild compatibility

The saved DEBATE fold statistics now have an end-to-end regression test. The
Wilcoxon calculation explicitly retains the original SciPy 1.10 selection rule
(exact for up to 50 nonzero paired differences, otherwise approximation). This
avoids a change in newer SciPy defaults for zero differences. The three rebuilt
tables match the saved outputs to numerical tolerance.

## Camera-ready result packaging

The six missing empirical packages are included under `validation/camera_ready/`.
`make rebuild-paper` checks them from saved inputs without hosted-model calls.
See that folder's README for the study map and numerical verification scope.

The release includes MIT licensing, citation metadata, and an index of all 17
final manuscript assets. Matching older files are referenced rather than copied.
The main GPT-5.4-mini profile panel uses the separately recovered austerity
campaign, not the older appendix topic grids. `make check-figures` verifies the
assets and profile/sweep trajectories. This check does not certify every historical
plotting command or regenerate hosted responses. Figure 1 drawing code remains
in the paper project; its final PDF is included.

## Repository cleanup

Superseded May setup notes, release audit, allowlist, and inventory were removed
from the current tree. The earlier six-condition architecture benchmark is not
reported in the active manuscript and was also removed. All remain recoverable
from the original snapshot commit. Existing paper artifacts and retained result
values were not changed. The current result index is `validation/README.md`.
